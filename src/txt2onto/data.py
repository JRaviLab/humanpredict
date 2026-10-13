"""Load the labeled CSV, clean labels and build model text (PLAN.md sections 0 and 2)."""
import hashlib
import json
import re

import pandas as pd

_WS = re.compile(r"\s+")
_PARTS = re.compile(r"[;|]")  # multi-value separators: ';' (older exports) and ' | ' (current)
LABEL = "category"  # 1 = human, -1 = not human, 0 = unknown


def clean_field(value, placeholders: set[str]) -> str:
    """Split on ';' or '|', drop placeholder tokens and blanks, re-join with '; '."""
    if value is None or pd.isna(value):
        return ""
    parts = (p.strip() for p in _PARTS.split(str(value)))
    return "; ".join(p for p in parts if p and p.strip(" \"'.").lower() not in placeholders)


def build_text(df: pd.DataFrame, text_cfg: dict) -> pd.Series:
    """'host: … | isolation: … | comments: …', keeping only non-empty fields."""
    placeholders = {p.lower() for p in text_cfg["placeholder_tokens"]}
    cleaned = {f: df[f].map(lambda v: clean_field(v, placeholders)) for f in text_cfg["fields"]}

    def join(i):
        return text_cfg["separator"].join(
            f"{text_cfg['prefixes'][f]}: {cleaned[f].iat[i]}" for f in text_cfg["fields"] if cleaned[f].iat[i]
        )

    return pd.Series([join(i) for i in range(len(df))], index=df.index, dtype=str)


def normalize(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def text_hash(norm: str) -> str:
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def is_placeholder(value, placeholders: set[str]) -> bool:
    """True if the value is missing or every part of it is a placeholder."""
    return clean_field(value, placeholders) == ""


def load_bioprojects(path) -> dict[str, str]:
    with open(path) as f:
        return {acc: proj for acc, proj in json.load(f).items() if proj}


def prepare(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return (labeled, unlabeled_texts, unlabeled_rows, stats).

    labeled:         one row per sample; label 1 = human, 0 = not human.
    unlabeled_texts: one row per unique normalized text (the stage-2 pool).
    unlabeled_rows:  accession -> text_hash, to map pool predictions back to samples.
    """
    text_cfg = cfg["text"]
    placeholders = {p.lower() for p in text_cfg["placeholder_tokens"]}
    labeled_path = cfg["paths"]["labeled_data"]
    if str(labeled_path).endswith(".parquet"):
        raw = pd.read_parquet(labeled_path)
    else:
        raw = pd.read_csv(labeled_path, low_memory=False)
    stats = {"raw_rows": len(raw), "raw_label_counts": {str(k): int(v) for k, v in raw[LABEL].value_counts().items()}}

    raw["text"] = build_text(raw, text_cfg)
    raw["text_norm"] = raw["text"].map(normalize)
    raw["text_hash"] = raw["text_norm"].map(text_hash)

    # ---- labeled ----
    lab = raw[raw[LABEL] != 0].copy()
    ph_neg = (lab[LABEL] == -1) & lab["host_name"].map(lambda v: is_placeholder(v, placeholders))
    stats["dropped_placeholder_negatives"] = int(ph_neg.sum())
    lab = lab[~ph_neg]

    empty = lab["text_norm"] == ""
    stats["dropped_labeled_empty_text"] = int(empty.sum())
    lab = lab[~empty]

    n_labels = lab.groupby("text_hash")[LABEL].nunique()
    conflicting = lab["text_hash"].isin(n_labels.index[n_labels > 1])
    stats["dropped_conflicting_texts"] = int((n_labels > 1).sum())
    stats["dropped_conflicting_rows"] = int(conflicting.sum())
    lab = lab[~conflicting]

    projects = load_bioprojects(cfg["paths"]["bioproject_map"])
    labeled = pd.DataFrame({
        "accession": lab["biosample_accession"].values,
        "text": lab["text"].values,
        "text_norm": lab["text_norm"].values,
        "text_hash": lab["text_hash"].values,
        "label": (lab[LABEL] == 1).astype("int8").values,
        "taxon_id": lab["sample_taxon_id"].values,
        "taxonomy_name": lab["sample_taxonomy_name"].values,
        "bioproject": lab["biosample_accession"].map(projects).values,
    })
    stats["labeled_rows"] = len(labeled)
    stats["labeled_label_counts"] = {str(k): int(v) for k, v in labeled["label"].value_counts().items()}
    stats["labeled_unique_texts"] = int(labeled["text_hash"].nunique())
    stats["labeled_bioproject_coverage"] = float(labeled["bioproject"].notna().mean())

    # ---- unlabeled ----
    unl = raw[(raw[LABEL] == 0) & (raw["text_norm"] != "")]
    stats["unlabeled_rows"] = int((raw[LABEL] == 0).sum())
    stats["dropped_unlabeled_empty_text"] = stats["unlabeled_rows"] - len(unl)
    unlabeled_rows = pd.DataFrame({"accession": unl["biosample_accession"].values, "text_hash": unl["text_hash"].values})
    unlabeled_texts = (
        unl.groupby("text_hash", sort=False)
        .agg(text=("text", "first"), text_norm=("text_norm", "first"), n_rows=("text", "size"))
        .reset_index()
    )
    stats["unlabeled_unique_texts"] = len(unlabeled_texts)
    return labeled, unlabeled_texts, unlabeled_rows, stats
