# txt2onto run on data/human_labeled.csv (2026-10-09)

**Result: completed through stage 1 (training + validation).** Attempt 1 stopped at `split` with a CUDA out-of-memory error (log: `run_attempt1.log`). Attempt 2 fixed that and ran `split` → `stage1` using the saved prep and embedding outputs (log: `run.log`).

| Stage | Wall clock | Peak RAM |
|---|---|---|
| prep + embed (attempt 1) | ~12 min | 13.8 GB |
| split | 18 min | **23.2 GB**, right at the machine's limit |
| stage1 | 46 min | 12.0 GB |

## Stage 1 results (validation set, n = 126,565, 75.4% human)

Best setting: C = 3.16, class_weight = none (chosen by grouped 5-fold CV on train; CV F1 0.958 ± 0.004).

| | Precision | Recall | F1 | Macro F1 | ROC AUC | Avg precision |
|---|---|---|---|---|---|---|
| Model @ 0.5 | 0.948 | 0.958 | 0.953 | 0.902 | 0.956 | 0.978 |
| Model @ best-F1 threshold (0.375) | 0.940 | 0.968 | 0.954 | 0.901 | | |
| Baseline: predict all human | 0.754 | 1.000 | 0.860 | 0.430 | | |
| Baseline: keyword rule (human/patient/homo) | 0.996 | 0.266 | 0.420 | 0.445 | | |

Confusion matrix at 0.5: TP 91,411, FP 5,041, FN 4,020, TN 26,093.

How to read it: with 75% of rows human, F1 on the human class looks high even for the trivial baseline (0.86). **Macro F1, which weights human and not-human equally, is the fairer comparison: 0.90 for the model vs. 0.43 for the baseline.** The keyword rule is almost never wrong when it fires but only catches 27% of human rows. The model's added value is in the rows with no explicit human term.

| Slice (threshold 0.375) | n | Human rate | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Explicit human terms in text | 25,512 | 99.6% | 0.997 | 0.995 | 0.996 |
| No explicit terms | 101,053 | 69.3% | 0.920 | 0.959 | 0.939 |
| Has host field | 48,501 | 80.6% | 0.949 | 0.990 | 0.969 |
| No host field | 78,064 | 72.2% | 0.933 | 0.953 | 0.943 |

- **Overfitting:** mild. Train log loss is 0.114 vs. 0.215 on validation, and ROC AUC is 0.987 vs. 0.956.
- **Stage 2 preview** (pseudo-labeling at 98% precision): a row is labeled human if p ≥ 0.9986 (covers 18.7% of human rows) and not human if p ≤ 0.043 (covers 14.6% of not-human rows).
- **Optimization trace check:** coefficients correlate 0.968 with sklearn's fit, and the final validation log loss (0.2157) matches.
- **Figures:** `stage1_training.png` and `stage1_validation.png`. Full metrics are in `stage1_metrics.json`.

**Not done:** the test set (126,567 rows) and the taxon-held-out split were not scored. By design, stage 1 never loads test rows (rule L7), and `run.py` has no test-evaluation stage yet.

## Split / leakage (all checks passed; see `leakage_report.json`)

- **Split sizes:** train 590,634; val 126,565; test 126,567. Each is 75.4% human.
- **No overlap between splits:** no exact text, near-duplicate cluster or group appears in more than one split.
- **Similarity of test to train:** at most 0.9795 cosine; the near-duplicate cutoff is 0.98.
- **Near-duplicates:** 870M pairs, collapsing into 140,841 clusters. The largest cluster is 3.4% of rows.
- **Cosmetic:** the report records the split path as `data/splits/split.parquet`, a hardcoded string. The file actually lives in `data/splits_updated/`.

## Fixes applied in attempt 2

- `embedding.sim_chunk: 512` in `configs/pipeline.yaml`, passed to every GPU similarity call in `split`. GPU batches drop from 6.4 GiB to about 0.8 GiB, and the results are unchanged.
- `stage1.n_jobs` lowered from 12 to 3 so parallel CV fits fit in 23 GB of RAM.

## What was done before the run

1. **Cleared old outputs.** I deleted `data/processed_updated`, `data/splits_updated`, `data/embeddings_updated`, `artifacts_updated` and `reports_updated`. These are the paths the current config writes to. I left the original baseline outputs (`artifacts/`, `reports/`, `data/processed|splits|embeddings/`) alone.
2. **Compatibility check against `data/human_labeled.csv`:**

   | Check | Result |
   |---|---|
   | Label column | renamed `is_human` → `category`; the code used `is_human` (fixed) |
   | Label values | 1 = human (638,473), -1 = not human (241,544), 0 = unknown (631,053). No human host names in -1 after your notebook fix |
   | `biosample_accession`, `sample_taxon_id`, `sample_taxonomy_name`, `host_name`, `host_comments`, `isolation_comments`, `comments` | present. Accessions are unique and none are null |
   | `host_taxid`, `host_health` | gone, but the pipeline never used them |
   | Multi-value separator | now ` \| ` instead of `;`. The old code only split on `;`, so placeholders such as `N/A \| N/A` stayed in the model text (fixed) |
   | Config path | pointed to `data/updated_human_labeled.csv`, which doesn't exist (fixed) |

3. **Code changes:**
   - `configs/pipeline.yaml`: changed `labeled_csv` to `data/human_labeled.csv`.
   - `src/txt2onto/data.py`:
     - The label column is now `category`.
     - `clean_field` splits on `;` or `|`.
     - `is_placeholder` treats a host name as a placeholder when every part of it is a placeholder (e.g. `missing | not applicable`).
   - `tests/test_data.py`: added cases for pipe-separated values. All 13 tests pass (`PYTHONPATH=src pytest`).

   Your `human_segregation.ipynb` notebook was not touched.

## Attempt 1 stage results (prep, embed)

**prep** (`data/processed_updated/prep_stats.json`)

| | rows |
|---|---|
| raw | 1,511,070 |
| labeled, kept | 843,766: 636,206 human (75.4%), 207,560 not human |
| dropped: negative with placeholder host | 32,239 |
| dropped: same text with conflicting labels | 3,872 rows (1,149 texts) |
| dropped: empty text | 140 |
| unlabeled pool | 631,053 rows → 625,282 unique texts |

BioProject coverage of the labeled rows is only **5.0%**. The previous run had 9.9%. Most of the leakage grouping will therefore come from near-duplicate text components rather than BioProject.

**embed**: 1,463,276 unique texts encoded and cached in `data/embeddings_updated/`. Both of these finished stages can be reused without recomputing.

**split**: failed at L4 near-duplicate detection.

## Why split failed

`near_dup_pairs` in `src/txt2onto/leakage.py` computes the similarity matrix in blocks of `chunk=4096` query rows against all 840,240 labeled texts at once:

- each fp16 block is 4096 × 840,240 × 2 bytes ≈ **6.4 GiB**
- the boolean mask is about 3.2 GiB more
- the full embedding matrix (≈1.2 GiB) also stays on the GPU

That is more than the 8 GB RTX 4070. The previous dataset had about 365k labeled texts, which used roughly 2.8 GiB per block and fit.

I did not change this, as you asked for no major changes while you were away.

