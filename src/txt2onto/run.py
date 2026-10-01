"""Pipeline CLI: python -m txt2onto.run --stage prep|embed|split|stage1|stage1_figures|all"""
import argparse
import json

import pandas as pd

from txt2onto.config import load_config


def stage_prep(cfg: dict) -> None:
    from txt2onto.data import prepare

    labeled, unlabeled_texts, unlabeled_rows, stats = prepare(cfg)
    out = cfg["paths"]["processed_dir"]
    labeled.to_parquet(out / "labeled.parquet", index=False)
    unlabeled_texts.to_parquet(out / "unlabeled.parquet", index=False)
    unlabeled_rows.to_parquet(out / "unlabeled_rows.parquet", index=False)
    (out / "prep_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def stage_embed(cfg: dict) -> None:
    from txt2onto.embed import update_cache

    out = cfg["paths"]["processed_dir"]
    texts = pd.concat([
        pd.read_parquet(out / "labeled.parquet", columns=["text_hash", "text"]),
        pd.read_parquet(out / "unlabeled.parquet", columns=["text_hash", "text"]),
    ])
    print(update_cache(texts, cfg))


def stage_split(cfg: dict) -> None:
    """Group, split (frozen), filter the unlabeled pool, and write the leakage report."""
    import numpy as np

    from txt2onto.embed import load_cache, lookup
    from txt2onto.leakage import (check_accessions, check_split_disjoint, max_similarity,
                                  near_dup_pairs, similarity_summary)
    from txt2onto.split import build_groups, components, group_size_summary, grouped_split

    paths, emb_cfg, split_cfg, seed = cfg["paths"], cfg["embedding"], cfg["split"], cfg["seed"]
    thr, device = emb_cfg["near_dup_cosine"], emb_cfg["device"]
    out = paths["processed_dir"]
    labeled = pd.read_parquet(out / "labeled.parquet")
    unlabeled = pd.read_parquet(out / "unlabeled.parquet")
    unlabeled_rows = pd.read_parquet(out / "unlabeled_rows.parquet")
    emb, index = load_cache(paths["embeddings_dir"])
    report = {"seed": seed, "near_dup_cosine": thr, "prep": json.loads((out / "prep_stats.json").read_text())}

    # L1
    report["L1_accessions"] = check_accessions(labeled, unlabeled_rows)
    assert report["L1_accessions"]["passed"], report["L1_accessions"]

    # L2
    prep = report["prep"]
    report["L2_label_noise"] = {k: prep[k] for k in (
        "dropped_placeholder_negatives", "dropped_conflicting_texts", "dropped_conflicting_rows", "dropped_labeled_empty_text")}

    # L4: near-duplicate text components (always kept together)
    texts = labeled["text_hash"].drop_duplicates().reset_index(drop=True)
    x_texts = lookup(texts, emb, index)
    pairs = near_dup_pairs(x_texts, thr, device)
    text_component = pd.Series(components(len(texts), pairs), index=texts.values)
    labels_by_text = labeled.drop_duplicates("text_hash").set_index("text_hash")["label"].loc[texts].to_numpy()
    labeled["text_component"] = labeled["text_hash"].map(text_component).to_numpy()

    # L5: greedy capped BioProject linking
    max_rows = int(split_cfg["max_group_frac"] * len(labeled))
    groups, skipped = build_groups(labeled, text_component, ["bioproject"], max_rows)
    labeled["group_id"] = groups
    report["L5_groups"] = {**group_size_summary(groups, split_cfg["max_group_frac"]),
                           "bioproject_coverage": prep["labeled_bioproject_coverage"],
                           "skipped_bioproject_links": skipped}
    assert report["L5_groups"]["passed"], report["L5_groups"]

    # L6: split, frozen on first run
    split_path = paths["splits_dir"] / "split.parquet"
    if split_path.exists():
        frozen = pd.read_parquet(split_path).set_index("accession")
        assert set(frozen.index) == set(labeled["accession"]), "labeled set changed since the split was frozen"
        labeled["split"] = labeled["accession"].map(frozen["split"]).to_numpy()
        labeled["split_taxon"] = labeled["accession"].map(frozen["split_taxon"]).to_numpy()
        report["L6_split"] = {"status": "reused frozen split"}
    else:
        labeled["split"] = grouped_split(labeled["label"].to_numpy(), groups, split_cfg["fractions"], seed)

        # L9: taxon-held-out robustness split. Taxa too large for one fold go to train; val/test rows
        # whose text component also occurs in train are excluded so text duplicates cannot leak.
        taxon = pd.factorize(labeled["taxon_id"])[0]
        taxon_size = np.bincount(taxon)[taxon]
        split_taxon = grouped_split(labeled["label"].to_numpy(), taxon, split_cfg["fractions"], seed,
                                    force_train=taxon_size > max_rows)
        train_comps = set(labeled.loc[split_taxon == "train", "text_component"])
        leaked = (split_taxon != "train") & labeled["text_component"].isin(train_comps).to_numpy()
        split_taxon[leaked] = "excluded"
        labeled["split_taxon"] = split_taxon
        labeled[["accession", "group_id", "split", "split_taxon"]].to_parquet(split_path, index=False)
        report["L6_split"] = {"status": "created and frozen", "path": "data/splits/split.parquet"}
    report["L6_split"].update({
        "counts": labeled["split"].value_counts().to_dict(),
        "human_rate": labeled.groupby("split")["label"].mean().round(4).to_dict(),
    })

    # L3: no text, near-dup component or group spans two splits
    report["L3_disjoint"] = [check_split_disjoint(labeled, k) for k in ("text_hash", "text_component", "group_id")]
    assert all(c["passed"] for c in report["L3_disjoint"]), report["L3_disjoint"]
    train_proj = set(labeled.loc[labeled["split"] == "train", "bioproject"].dropna())
    test_proj = labeled[(labeled["split"] == "test") & labeled["bioproject"].notna()]
    report["L3_disjoint"].append({"key": "bioproject (diagnostic)",
                                  "test_rows_with_project_in_train": round(float(test_proj["bioproject"].isin(train_proj).mean()), 4)})

    # L4 report: near-duplicate statistics and residual similarity across splits
    x = {s: lookup(labeled.loc[labeled["split"] == s, "text_hash"].unique(), emb, index) for s in ("train", "val", "test")}
    report["L4_near_duplicates"] = {
        "pairs": int(len(pairs)),
        "cross_label_pairs": int((labels_by_text[pairs[:, 0]] != labels_by_text[pairs[:, 1]]).sum()),
        "components": int(text_component.nunique()),
        "largest_component_texts": int(np.bincount(text_component.to_numpy()).max()),
        "test_max_sim_to_train": similarity_summary(max_similarity(x["test"], x["train"], device), thr),
        "val_max_sim_to_train": similarity_summary(max_similarity(x["val"], x["train"], device), thr),
    }
    # Pairs were found with the same fp16 kernel; allow a hair of numerical slack at the threshold.
    residual = max_similarity(x["test"], x["train"], device)
    assert (residual >= thr + 2e-3).sum() == 0, "test text near-duplicates a training text"

    # L7: enforced in stage 1 (scaler + LR in one Pipeline, grouped CV on train, thresholds on val)
    report["L7_fit_on_train_only"] = {"status": "enforced in stage1"}

    # L8: unlabeled pool hygiene
    labeled_hashes = set(labeled["text_hash"])
    in_labeled = unlabeled["text_hash"].isin(labeled_hashes).to_numpy()
    held_out = np.vstack([x["val"], x["test"]])
    pool_sim = max_similarity(lookup(unlabeled["text_hash"], emb, index), held_out, device)
    unlabeled["max_sim_val_test"] = pool_sim
    unlabeled["pool_status"] = np.select([in_labeled, pool_sim >= thr], ["in_labeled", "near_val_test"], "ok")
    unlabeled.to_parquet(out / "unlabeled.parquet", index=False)
    pool_ok = unlabeled["pool_status"] == "ok"
    heldout_hashes = set(labeled.loc[labeled["split"] != "train", "text_hash"])
    assert not unlabeled.loc[pool_ok, "text_hash"].isin(heldout_hashes).any()
    report["L8_unlabeled_pool"] = {
        "unique_texts": len(unlabeled),
        "status_counts": unlabeled["pool_status"].value_counts().to_dict(),
        "usable_rows": int(unlabeled.loc[pool_ok, "n_rows"].sum()),
        "passed": True,
    }

    # L9
    report["L9_taxon_split"] = {
        "counts": labeled["split_taxon"].value_counts().to_dict(),
        "human_rate": labeled.groupby("split_taxon")["label"].mean().round(4).to_dict(),
        "taxa_forced_to_train": int(labeled.loc[
            labeled.groupby("taxon_id")["accession"].transform("size") > max_rows, "taxon_id"].nunique()),
    }

    labeled.to_parquet(out / "labeled.parquet", index=False)
    report_path = paths["reports_dir"] / "leakage_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "prep"}, indent=2, default=str))


def stage_stage1(cfg: dict) -> None:
    """Grid search, fit, validation diagnostics and training-process traces. Test rows are never loaded (L7)."""
    import joblib
    import numpy as np
    import yaml

    from txt2onto.embed import load_cache, lookup
    from txt2onto.evaluate import (best_f1_threshold, calibration_bins, precision_target_thresholds,
                                   ranking_metrics, threshold_metrics)
    from txt2onto.train import cv_results_frame, grid_search, learning_curve, optimization_trace

    paths, s1, seed = cfg["paths"], cfg["stage1"], cfg["seed"]
    out = paths["artifacts_dir"] / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    labeled = pd.read_parquet(paths["processed_dir"] / "labeled.parquet")
    labeled = labeled[labeled["split"].isin(["train", "val"])].reset_index(drop=True)  # L7: no test rows
    assert "test" not in set(labeled["split"])
    emb, index = load_cache(paths["embeddings_dir"])
    tr, va = labeled[labeled["split"] == "train"], labeled[labeled["split"] == "val"]
    X_tr, y_tr, g_tr = lookup(tr["text_hash"], emb, index), tr["label"].to_numpy(), tr["group_id"].to_numpy()
    X_va, y_va = lookup(va["text_hash"], emb, index), va["label"].to_numpy()

    # Model selection: grouped 5-fold CV on train only
    search = grid_search(X_tr, y_tr, g_tr, s1, seed)
    cv = cv_results_frame(search)
    cv.to_csv(out / "cv_results.csv", index=False)
    best = {"C": float(search.best_params_["clf__C"]), "class_weight": search.best_params_["clf__class_weight"]}
    model = search.best_estimator_
    joblib.dump(model, out / "model.joblib")
    (out / "pipeline.yaml").write_text(yaml.safe_dump({**cfg, "paths": {k: str(v) for k, v in paths.items()}}))

    # Validation scores and thresholds
    p_va = model.predict_proba(X_va)[:, 1]
    p_tr = model.predict_proba(X_tr)[:, 1]
    pd.DataFrame({"accession": va["accession"], "text_hash": va["text_hash"], "label": y_va, "p_human": p_va}) \
        .to_parquet(out / "val_scores.parquet", index=False)
    thr = best_f1_threshold(y_va, p_va)
    grid_t = np.round(np.linspace(0.01, 0.99, 99), 2)
    pd.DataFrame([threshold_metrics(y_va, p_va, t) for t in grid_t]).drop(columns="confusion") \
        .to_csv(out / "val_threshold_curve.csv", index=False)

    # Reference points and slices on val
    explicit = va["text_norm"].str.contains(r"\b(?:human|patient|homo sapiens?|homo)\b", regex=True).to_numpy()
    has_host = va["text_norm"].str.startswith("host:").to_numpy()
    slices = {}
    for name, mask in {"explicit_human_terms": explicit, "no_explicit_terms": ~explicit,
                       "has_host_field": has_host, "no_host_field": ~has_host}.items():
        slices[name] = {"n": int(mask.sum()), "human_rate": round(float(y_va[mask].mean()), 4),
                        **{k: v for k, v in threshold_metrics(y_va[mask], p_va[mask], thr).items() if k != "confusion"}}

    best_row = cv[(cv["C"] == best["C"]) & (cv["class_weight"] == str(best["class_weight"] or "none"))].iloc[0]
    metrics = {
        "best_params": best,
        "cv_best": {k: round(float(best_row[k]), 4) for k in best_row.index if k.startswith(("cv_", "train_"))},
        "train": {"n": len(y_tr), **ranking_metrics(y_tr, p_tr), "at_0.5": threshold_metrics(y_tr, p_tr, 0.5)},
        "val": {"n": len(y_va), "human_rate": round(float(y_va.mean()), 4), **ranking_metrics(y_va, p_va),
                "at_0.5": threshold_metrics(y_va, p_va, 0.5), "at_best_f1": threshold_metrics(y_va, p_va, thr)},
        "val_baselines": {
            "all_human": threshold_metrics(y_va, np.ones_like(p_va), 0.5),
            "keyword_rule": threshold_metrics(y_va, explicit.astype(float), 0.5),
        },
        "val_slices_at_best_f1": slices,
        "val_calibration": calibration_bins(y_va, p_va),
        "stage2_preview": precision_target_thresholds(y_va, p_va, cfg["stage2"]["target_precision"]),
        "L7": {"test_rows_loaded": 0, "scaler_in_pipeline": True, "threshold_selected_on": "val"},
    }

    # Training-process diagnostics
    lc = learning_curve(X_tr, y_tr, g_tr, s1, {"C": best["C"], "class_weight": best["class_weight"]},
                        seed, s1["learning_curve_fractions"])
    lc.to_csv(out / "learning_curve.csv", index=False)
    trace, theta = optimization_trace(X_tr, y_tr, X_va, y_va, best["C"], best["class_weight"], s1["max_iter"])
    trace.to_csv(out / "optimization_trace.csv", index=False)
    sk_coef = model.named_steps["clf"].coef_.ravel()
    metrics["trace_check"] = {
        "iterations": int(trace["iteration"].max()),
        "coef_correlation_with_sklearn": round(float(np.corrcoef(sk_coef, theta[:-1])[0, 1]), 6),
        "final_val_log_loss": round(float(trace["val_log_loss"].iloc[-1]), 4),
    }

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: metrics[k] for k in ("best_params", "cv_best", "val", "val_baselines", "stage2_preview", "trace_check")},
                     indent=2, default=str))
    stage_stage1_figures(cfg)


def stage_stage1_figures(cfg: dict) -> None:
    """Draw report figures from saved stage-1 outputs (validation only; no test rows)."""
    from txt2onto.plots import training_figure, validation_figure

    art, rep = cfg["paths"]["artifacts_dir"] / "stage1", cfg["paths"]["reports_dir"]
    metrics = json.loads((art / "metrics.json").read_text())
    val = pd.read_parquet(art / "val_scores.parquet")
    training_figure(pd.read_csv(art / "optimization_trace.csv"), pd.read_csv(art / "cv_results.csv"),
                    pd.read_csv(art / "learning_curve.csv"), metrics["best_params"], rep / "stage1_training.png")
    validation_figure(val["label"].to_numpy(), val["p_human"].to_numpy(), pd.read_csv(art / "val_threshold_curve.csv"),
                      metrics["val_calibration"], metrics["val"]["at_best_f1"]["threshold"], metrics["stage2_preview"],
                      rep / "stage1_validation.png")
    (rep / "stage1_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"figures written to {rep}")


STAGES = {"prep": stage_prep, "embed": stage_embed, "split": stage_split, "stage1": stage_stage1,
          "stage1_figures": stage_stage1_figures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=[*STAGES, "all"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    pipeline = [n for n in STAGES if n != "stage1_figures"]  # stage1 draws its own figures
    for name in pipeline if args.stage == "all" else [args.stage]:
        print(f"== {name} ==")
        STAGES[name](cfg)


if __name__ == "__main__":
    main()
