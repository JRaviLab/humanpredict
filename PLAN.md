# txt2onto host test: human vs not-human classifier on BioSample metadata

Binary logistic regression (L2) that predicts whether a BioSample's isolation source is a human host.
Features are PubMedBERT sentence embeddings of the concatenated `host_comments`, `isolation_comments`
and `comments` fields. A second training stage adds high-confidence pseudo-labels (self-training) from the
unlabeled rows. Both stages are compared on the same frozen test set by **precision, recall and F1**.

Status legend: `[ ]` todo, `[~]` in progress, `[x]` done.

---

## 0. Data definitions and known issues

Source table: `data/human_labeled_biosamples.csv`, built by `data exploration/data_prep.ipynb`.

| `is_human` | Meaning | Use |
|---|---|---|
| `1`  | host_name matches a human term | labeled positive |
| `-1` | host_name present, not human | labeled negative (mapped to `0` for training) |
| `0`  | **unlabeled** | pool for stage-2 pseudo-labeling |

Current CSV (regenerated 2026-09-29 by `src/data_prep.py`): 171,001 rows = 46,183 (`1`) + 35,498 (`-1`) + 89,320 (`0`).
The labeled accessions match the previous CSV exactly. Rebuild with
`python src/data_prep.py` (needs `data/bacteria_BioSample.csv`, which is not in git).

- [x] **0.1 Regenerate the unlabeled rows (blocking for stage 2).** *Done: fixed in the notebook (cell 20) and in `src/data_prep.py`; the script asserts there is no overlap.*
  In the notebook, `nans = df.dropna(subset='host_name')` keeps the rows that *have* a host_name, so all
  80,658 old `is_human == 0` rows repeated labeled accessions (100% overlap). Pseudo-labeling them
  would put copies of val/test samples back into training. Fix:
  `nans = df[df['host_name'].isna()]`, then dedupe as before and re-export the CSV.
  The raw file has 4.72M rows with a null host_name; deduplicating on isolation_comments (as the notebook does) leaves 89,320 unlabeled rows. The pipeline runs a hard assertion (check L1) that no `is_human == 0` accession
  appears among the labeled rows.
- [ ] **0.2 Placeholder negatives.** 5,105 `-1` rows have host_name "not applicable", "not collected",
  etc. These are missing labels, not "not human". They are dropped from the labeled set.
- [ ] **0.3 Conflicting texts.** 891 normalized texts (3,006 rows) appear with both labels. They are dropped.
- Expected clean labeled set: ~73.4k rows (44.3k human / 29.1k not-human), ~70.9k unique texts.

## 1. Environment *(done 2026-09-29)*

- [x] Conda env `txt2onto` (Python 3.14.7). CUDA torch wheels exist for 3.14, so no separate env was needed.
- [x] Installed (exact versions in `requirements.lock.txt`; direct dependencies in `requirements.txt`):
  ```bash
  conda activate txt2onto
  pip install torch --index-url https://download.pytorch.org/whl/cu128   # driver reports CUDA 13.1; cu128 works on it
  pip install -r requirements.txt
  pip list --format=freeze > requirements.lock.txt   # plain `pip freeze` prints "@ file://" for conda packages
  ```
  torch 2.11.0+cu128, sentence-transformers 6.1.0, transformers 5.17.0, scikit-learn 1.9.1, scipy 1.18.1,
  pandas 3.0.5, numpy 2.5.2, pyarrow 25.0.1.
- [x] GPU verified: `torch.cuda.is_available() == True` on the RTX 4070 Laptop GPU.
- [x] `NeuML/pubmedbert-base-embeddings` pinned to `b79526d6ef3645e0df4530322e266f24c829f5ef` in `configs/pipeline.yaml`.
  Smoke test: BertModel with mean pooling, 768 dimensions (default max_seq_length 512, set to 256). fp16 output
  is unit-normalized, at about 9.2k short texts/s with a 0.45 GiB VRAM peak.
- Implementation notes found during setup:
  - sklearn 1.9 deprecates `LogisticRegression(penalty=...)`. Use `l1_ratio=0.0` (with finite C) for L2.
  - fp16 encoding returns float16 arrays. Convert to float32 before saving `emb.npy`.
  - Raw cosine similarity doesn't separate host types on its own (cos("blood culture", "wound swab") = 0.36 <
    cos("blood culture", "soil") = 0.38). The logistic regression has to learn the boundary.
- Hardware: RTX 4070 Laptop 8 GB (WSL2), 28 cores, 23 GB RAM.
- `txt2onto.venv/` is empty and unused; ignored by git.

## 2. Layout and storage *(done 2026-09-29)*

```
pyproject.toml                     # package `txt2onto` (installed editable: pip install -e .)
configs/pipeline.yaml              # seed, paths, model revision, C grid, precision targets, caps
src/data_prep.py                   # rebuilds data/human_labeled_biosamples.csv from the raw CSV
src/txt2onto/
  config.py      # [x] loads YAML, resolves paths relative to repo root
  data.py        # [x] load CSV, split labeled/unlabeled, placeholder cleanup, text building
  embed.py       # [x] PubMedBERT embeddings, cached by text hash
  leakage.py     # [x] near-duplicate search (GPU), max-similarity, L1/L3 checks
  split.py       # [x] capped greedy grouping, grouped stratified split
  train.py       # [ ] StandardScaler + L2 LogisticRegression, grouped CV for C
  self_train.py  # [ ] pseudo-label selection and stage-2 fit
  evaluate.py    # [ ] precision / recall / F1, bootstrap CIs, McNemar
  run.py         # CLI: python -m txt2onto.run --stage prep|embed|split|all   (stage1|pseudo|stage2|compare to come)
tests/           # [x] test_data.py, test_split_leakage.py (8 tests)
data/processed/labeled.parquet        # accession, text, text_norm, text_hash, label{0,1}, taxon_id, taxonomy_name,
                                      #   bioproject, text_component, group_id, split, split_taxon
data/processed/unlabeled.parquet      # text_hash, text, text_norm, n_rows, max_sim_val_test, pool_status
data/processed/unlabeled_rows.parquet # accession -> text_hash (maps pool predictions back to samples)
data/processed/prep_stats.json
data/splits/split.parquet             # FROZEN split (accession, group_id, split, split_taxon); in git
data/embeddings/emb.npy               # float32 [154,032, 768] (473 MB), memory-mapped; not in git
data/embeddings/emb_index.parquet     # text_hash -> row
artifacts/stage1/, artifacts/stage2/  # model.joblib, config snapshot, metrics.json, val_scores, test_preds
artifacts/pseudo/round_{k}.parquet    # text_hash, pseudo_label, p_human, weight, round
reports/leakage_report.json (in git), reports/comparison.md, reports/pr_curves.png
```

Each stage reads the previous stage's files and skips work that is already done. Duplicate texts share one embedding row.
Running every stage (`--stage all`) takes about 1 minute; embedding all 154,032 unique texts from scratch takes about 40 s.

### Text construction
1. For each field, split on `;`, drop placeholder tokens (missing, not collected, not applicable, unknown,
   na, n/a, none, -, not provided, restricted access) and strip whitespace.
2. `text = "host: … | isolation: … | comments: …"`, keeping only the non-empty fields.
3. Drop rows whose text is empty.
4. `text_norm` = lowercased with whitespace collapsed; `text_hash = sha1(text_norm)`. Used for dedupe and grouping only.
5. **Never** used as features: `host_name`, `host_taxid`, `host_health` (the label comes from host_name).

Result: 73,440 labeled rows (44,345 human / 29,095 not-human; 70,941 unique texts; 60.0% have a BioProject)
and 85,582 unique unlabeled texts (from 89,244 non-empty rows).

## 3. Leakage checks (`reports/leakage_report.json`) *(done 2026-09-29)*

- [x] **L1** Every labeled accession is unique, and no `is_human == 0` accession appears among the labeled ones. *(hard assert)* Duplicates: 0; overlap: 0.
- [x] **L2** Removed 5,108 placeholder-host negatives, 127 empty texts, and 891 conflicting-label texts (3,006 rows).
- [x] **L3** No `text_hash`, near-duplicate component or `group_id` appears in more than one split. *(hard assert)*
- [x] **L4** Near duplicates at cosine ≥ 0.98: 255,052 pairs forming 59,017 components (largest: 320 texts, e.g. the
  "Stavanger University Hospital; blood; see also SAMEA…" template). 7,148 of the pairs have different labels.
  Near-duplicate components are **always** kept in one split. Highest test-to-train similarity: median 0.84,
  99th percentile 0.978, maximum 0.9795, with 0 pairs at or above 0.98. *(hard assert)*
- [x] **L5** *Changed from plan:* an unconstrained union over text and BioProject chained 415 projects (through
  generic texts such as "BAL" and "tracheal aspirate") into a single 6,725-row group (9.2%). BioProjects are now linked
  **greedily, smallest first, skipping any link that would push a group above 5%**. Only PRJNA288601 (1,677 rows) and
  PRJNA514245 (1,409 rows) were skipped. Largest group: 1,923 rows (2.6%). Test rows whose BioProject also appears in
  train: 72.6% when grouping by text only, 6.7% with capped linking.
- [x] **L6** *Changed from plan:* the split is frozen in `data/splits/split.parquet` (committed to git), not in `labeled.parquet`,
  so rerunning `prep` cannot change it. Reruns check that the labeled accessions match and reuse the frozen split.
  Train / val / test = 51,406 / 11,018 / 11,016, each 60.38% human.
- [ ] **L7** Scaler and LR sit inside one sklearn Pipeline. C is chosen by grouped CV on train; thresholds on val.
  The test set is scored only in `compare`. *(Enforced in stage 1.)*
- [x] **L8** Unlabeled pool: 82,774 usable texts (85,635 rows). Excluded: 2,491 texts already labeled and 317 texts with
  cosine ≥ 0.98 to a val/test text (`pool_status` column).
- [x] **L9** Taxon-held-out split (`split_taxon`): 3 taxa larger than 5% are forced into train, and val/test rows whose text component
  also appears in train are marked `excluded` (2,344 rows). Train / val / test = 52,562 / 9,259 / 9,275. The human rate
  differs across splits (63.3% / 55.0% / 53.5%), so compare F1 against each split's own base rate.

## 4. Stage 1: supervised baseline *(done 2026-09-29; report: `reports/stage1_report.md`)*

- [x] Embed with `SentenceTransformer(model, revision=…, device="cuda")`: fp16, `max_seq_length=256`,
  `normalize_embeddings=True`, batch size 256. *(Done in the `embed` stage.)*
- [x] `Pipeline([StandardScaler(), LogisticRegression(l1_ratio=0.0, solver="lbfgs", max_iter=5000)])`, in `train.py`.
- [x] Grid: C ∈ logspace(-3, 2, 11) × class_weight ∈ {None, "balanced"}. 5-fold StratifiedGroupKFold on train (grouped by `group_id`),
  scored by F1. **Selected C=0.01, class_weight=None; CV F1 0.894 ± 0.011.** The F1 curve is flat across C (0.892–0.894).
- [x] Validation: F1 0.889 at 0.5 (P 0.902 / R 0.876), 0.891 at the F1-optimal threshold 0.4595; AP 0.953.
  Val scores, the threshold curve and calibration are saved in `artifacts/stage1/`.
- [x] Training-process diagnostics: L-BFGS loss trace (re-implemented objective; coefficient correlation with sklearn
  0.9997), regularization path, learning curve (plateaus after about 16k rows). Figures: `reports/stage1_training.png`,
  `reports/stage1_validation.png`.
- [x] L7 enforced: stage 1 never loads test rows; the scaler sits inside the Pipeline; thresholds are chosen on val.
- Stage-2 preview on val: τ⁺ = 0.958 (covers 28.4%), τ⁻ = 0.028 (covers 18.4%) at precision ≥ 0.98.
- **Open issues found (decide before stage 2):**
  - [ ] Human host-name regex misses "Homosapiens", "Homo_sapiens", "Hom sapiens", "HumanN", … (about 85 samples labeled not-human).
  - [ ] 370 "reactor sample N" samples (PRJNA606923, PRJNA590676) labeled *Homo sapiens*, 322 of them in val. They cause 200 of the 222
    confident false negatives; without them, val F1 is 0.911.
  - [ ] Stage 2: cap pseudo-labels per near-duplicate component of the pool, so confident-but-wrong templates aren't reinforced.
- CLI: `python -m txt2onto.run --stage stage1` (about 3 min: 110 CV fits, learning curve, trace, figures); `--stage stage1_figures` redraws figures only.

## 5. Stage 2: self-training on high-confidence predictions

- [ ] Score the cleaned unlabeled pool (unique texts) with the stage-1 model.
- [ ] Thresholds from val, not chosen by hand. τ⁺ = lowest threshold where val precision for the human class is ≥ 0.98;
  τ⁻ = highest threshold where val precision for the not-human class is ≥ 0.98.
- [ ] Pseudo-label: p ≥ τ⁺ → 1, p ≤ τ⁻ → 0. Keep the class ratio close to the training ratio (take the most confident items per class)
  and cap the total at the size of the labeled training set.
- [ ] Weights: labeled rows 1.0; pseudo-labeled rows w ∈ {0.25, 0.5, 1.0}, chosen on val.
- [ ] Refit on train ∪ pseudo with `sample_weight`. Pseudo-labeled rows are used for training only, never for scoring CV folds.
- [ ] Optional rounds 2–3: re-score the remaining pool and stop when val F1 improves by less than 0.002.
- [ ] Sanity check: sample 100 pseudo-labels per class with their text into `reports/pseudo_sample.csv` to read by hand.

## 6. Evaluation (before vs after stage 2)

Same frozen test set; run once in `compare`.

- [ ] Precision, recall and F1 for the human class, plus macro-F1 and the confusion matrix, at threshold 0.5 and at the val-selected threshold.
- [ ] 95% CIs from 1,000 bootstrap resamples drawn by group; paired bootstrap on the stage 2 minus stage 1 difference; McNemar test on the thresholded predictions.
- [ ] Slices: texts that name humans outright (human / patient / homo sapiens) vs indirect ones; texts with vs without host_comments;
  (optional) the taxon-grouped split from L9.
- [ ] `reports/comparison.md`: side-by-side table, PR curves and a leakage summary.

## Progress log

- 2026-09-29: Data profiled, plan written, repo pushed to GitHub (private).
- 2026-09-29: Step 0.1 done. CSV regenerated with real unlabeled rows (89,320, no host_name); the export is scripted in `src/data_prep.py`.
- 2026-09-29: Section 1 done. Environment installed and locked, GPU verified, model revision pinned in `configs/pipeline.yaml`.
- 2026-09-29: Sections 2 and 3 done. Package, embedding cache, capped greedy grouping, frozen split and leakage report are in place; all hard checks pass.
- 2026-09-29: Section 4 done. Stage-1 baseline: val F1 0.889 (CV 0.894). Report in `reports/stage1_report.md`. Label-noise issues logged under section 4.
