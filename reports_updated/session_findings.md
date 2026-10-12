# Txt2Onto human-host classifier: session findings (2026-10-09)

Retrain and review on `data/human_labeled.csv`, covering data checks, pipeline fixes, validation results, the label-error review and recommendations.

---

## 1. Summary

- **The pipeline now runs end to end on the new labels** through training and validation (`prep` → `embed` → `split` → `stage1`). Four fixes were needed: three for the new CSV format and one for GPU memory. Training parallelism was also lowered to fit RAM.
- **Validation:** F1 0.953, macro F1 0.902, ROC AUC 0.956 on 126,565 rows (75.4% human). This is far better than predicting "all human" (macro F1 0.43) or a keyword rule (recall 0.27).
- **The weak side is ruling out human.** About 16% of not-human samples are called human. Most of these are animal or veterinary samples whose text reads as clinical.
- **Label review:** the model's confident "errors" exposed several patterns your labeling rule misses: subject IDs, specimens or diseases in `host_name`, and human-only pathogens. These are a minority, about 13% of the highest-confidence errors. Most high-confidence errors are real model mistakes.
- **The main risk for imputation is that the rows to impute look different from the training rows.** Rows with an empty `host_name` almost never have `host_comments` (0.7% vs. 48%), so current validation overstates imputation performance.
- **Not done yet:** the test set and the held-out-organism split have not been scored.

---

## 2. Data compatibility and code changes

### 2.1 Compatibility check: `data/human_labeled.csv` against the pipeline

| Check | Finding | Action |
|---|---|---|
| Label column | `is_human` renamed to `category` | Code updated |
| Label encoding | 1 = human, -1 = not human, 0 = unknown, as specified (after your notebook fix; see 2.2) | — |
| Required columns | `biosample_accession`, `sample_taxon_id`, `sample_taxonomy_name`, `host_name`, `host_comments`, `isolation_comments`, `comments` all present. Accessions unique, none null | — |
| Removed columns | `host_taxid`, `host_health` | Not used by the pipeline |
| Multi-value separator | Now ` \| ` instead of `;`. The old code split only on `;`, so placeholders like `N/A \| N/A` stayed in the model text, and pipe-joined placeholder host names weren't recognized | Split on both |
| Config path | Pointed to `data/updated_human_labeled.csv`, which doesn't exist | Fixed |

### 2.2 Labeling bug found before the run (since fixed by you)

The first version of the CSV labeled every `host_name == "Homo sapiens"` row as -1: 628,071 rows, plus "patient", "homo", "9606", etc. Only 8,217 rows were labeled 1. The cause was a case-sensitive match against the human-name list. After your fix:

| | Before fix | After fix |
|---|---|---|
| Human (1) | 8,217 | 638,473 |
| Not human (-1) | 870,674 | 241,544 |
| Unknown (0) | 631,053 | 631,053 |
| Human names inside -1 | 629,819 | 0 |

### 2.3 Code and config changes

| File | Change |
|---|---|
| `configs/pipeline.yaml` | `labeled_csv: data/human_labeled.csv` |
| `configs/pipeline.yaml` | New `embedding.sim_chunk: 512`: GPU similarity block size, previously hardcoded at 4096 |
| `configs/pipeline.yaml` | `stage1.n_jobs: 12 → 3` |
| `src/txt2onto/data.py` | Label column read from `category` (constant `LABEL`) |
| `src/txt2onto/data.py` | `clean_field` splits on `;` or `\|` and rejoins with `; ` |
| `src/txt2onto/data.py` | `is_placeholder` is true when every part is a placeholder (e.g. `missing \| not applicable`) |
| `src/txt2onto/run.py` | All 7 similarity calls in `split` use `sim_chunk` |
| `tests/test_data.py` | Tests for pipe-separated values and placeholders. 13 of 13 pass |

The notebook `human_segregation.ipynb` was not modified.

**Environment notes:**
- `python` isn't on the PATH. Use `~/miniforge3/envs/txt2onto/bin/python`.
- The installed `txt2onto` package doesn't load, so run with `PYTHONPATH=src`.

---

## 3. Run history

| Attempt | Stages | Result | Wall clock | Peak RAM |
|---|---|---|---|---|
| 1 | prep, embed, split | `split` failed: CUDA out of memory | 12.5 min | 13.8 GB |
| 2 | split, stage1 | Succeeded | 18 min + 46 min | **23.2 GB** (split), 12.0 GB (stage1) |

- Before attempt 1, I cleared all previous `_updated` outputs. The original baseline outputs (`artifacts/`, `reports/`, `data/processed|splits|embeddings/`) were kept.
- **Out-of-memory cause:** near-duplicate detection compared 4,096 texts at a time against all 840,240 labeled texts. Each fp16 block was about 6.4 GiB, plus about 3.2 GiB for the mask, on an 8 GB GPU. The previous dataset (~365k texts) fit. With `sim_chunk: 512`, blocks are about 0.8 GiB and the results are identical.
- **RAM warning:** `split` peaked at 23.2 GB, essentially all of this machine's memory. Most of that comes from holding the 870M near-duplicate pairs. A larger dataset will likely be killed by the OS here.
- **stage1:** the previous `_updated` run died in `stage1` without writing output, likely from running out of RAM with `n_jobs: 12`. With `n_jobs: 3` it peaked at 12 GB.

---

## 4. Prep: what happens to the rows

Dropping these rows is this project's own label cleaning (PLAN.md sections 0 and 2), not something Txt2Onto requires in general.

| | Raw CSV | After prep |
|---|---|---|
| Human (1) | 638,473 | 636,206 |
| Not human (-1) | 241,544 | 207,560 |
| **Labeled total** | **880,017** | **843,766** (75.4% human) |
| Unknown (0) | 631,053 | 630,878 rows set aside as the stage-2 unlabeled pool (625,282 unique texts) |

**Labeled rows removed (36,251):**
1. **Not-human rows whose `host_name` is a placeholder (32,239),** e.g. "not applicable". These rows are -1 only because no host was given. "not applicable" is in the pipeline's placeholder list but not in the notebook's `missing_labels`, so the notebook sent these rows to -1.
2. **Empty input text (140).** The model only sees `host_comments`, `isolation_comments` and `comments`. `host_name` is deliberately excluded because the label comes from it.
3. **Identical text with conflicting labels (3,872 rows, 1,149 texts).** These can't be learned, and they would put contradictory copies in train and test.

**Class balance flipped:** the previous `_updated` run was 29% human because of the labeling bug in 2.2. It's now 75% human, so results from the two runs can't be compared.

**BioProject coverage is 5.0%** of labeled rows (previously 9.9%). Leakage grouping relies mostly on near-duplicate text.

---

## 5. Split and leakage checks (all passed)

| Split | Rows | Human rate |
|---|---|---|
| Train | 590,634 | 75.4% |
| Validation | 126,565 | 75.4% |
| Test | 126,567 | 75.4% |

- **Equal human rates are intentional.** The split is stratified by label, and with 117,781 mostly small groups it can match the rate almost exactly. The upside is that metrics are comparable across splits and the validation threshold transfers to test. The downside is that it says nothing about performance at a different human share.
- **No overlap between splits:** no exact text, near-duplicate cluster or group crosses splits, and no test row shares a BioProject with train.
- **Similarity of test to train:** max 0.9795 cosine, median 0.95. The near-duplicate cutoff is 0.98. Test texts are close in style to training texts.
- **Near-duplicates:** 870M pairs in 140,841 clusters. The largest cluster is 28,849 texts (3.4% of rows). 514k pairs are near-identical text with opposite labels, which is a sign of label noise.
- **Held-out-organism split** (whole taxa held out): train 78.4% human, validation 72.3%, test 75.4%; 89,595 rows excluded. This split exists but **hasn't been scored**.
- **Unlabeled pool:** 619,760 of 625,282 texts are usable for stage 2.
- **Cosmetic:** the leakage report records the split path as `data/splits/split.parquet`, a hardcoded string. The file is actually in `data/splits_updated/`.

---

## 6. Stage 1 validation results

The model is StandardScaler + L2 logistic regression on PubMedBERT sentence embeddings. Best setting: C = 3.16, class_weight = none, chosen by grouped 5-fold CV on train (CV F1 0.958 ± 0.004).

### 6.1 Overall

| | Precision | Recall | F1 | Macro F1 | ROC AUC | Avg precision |
|---|---|---|---|---|---|---|
| Model @ 0.5 | 0.948 | 0.958 | 0.953 | 0.902 | 0.956 | 0.978 |
| Model @ best-F1 threshold (0.375) | 0.940 | 0.968 | 0.954 | 0.901 | | |
| Baseline: predict all human | 0.754 | 1.000 | 0.860 | 0.430 | | |
| Baseline: keyword rule | 0.996 | 0.266 | 0.420 | 0.445 | | |

Confusion matrix at 0.5: TP 91,411, FP 5,041, FN 4,020, TN 26,093.

### 6.2 Per class at 0.5

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Human (75%) | 0.948 | 0.958 | 0.953 |
| Not human (25%) | 0.867 | 0.838 | 0.852 |

### 6.3 Slices (threshold 0.375)

"Host field" here means `host_comments`, not `host_name`.

| Slice | n | Human rate | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Explicit human terms | 25,512 | 99.6% | 0.997 | 0.995 | 0.996 |
| No explicit terms | 101,053 | 69.3% | 0.920 | 0.959 | 0.939 |
| Has `host_comments` | 48,501 | 80.6% | 0.949 | 0.990 | 0.969 |
| No `host_comments` | 78,064 | 72.2% | 0.933 | 0.953 | 0.943 |

- **Train vs. validation:** log loss 0.114 vs. 0.215; ROC AUC 0.987 vs. 0.956.
- **Optimization trace check:** coefficients correlate 0.968 with sklearn's fit; final validation log loss matches (0.2157).

### 6.4 Strengths

- **It learns well beyond keywords:** F1 0.939 on the 80% of rows with no explicit human term.
- **Model selection is reliable:** CV F1 (0.958) closely matches validation F1 (0.953).
- **Scores are reasonably calibrated:** the best-F1 threshold (0.375) is close to 0.5, and F1 barely changes between them.
- **It ranks well:** average precision 0.978. Its most confident predictions are very reliable (see section 7).
- **Leakage-controlled:** the score isn't simple memorization.
- **Cheap and transparent:** fast to retrain and inspect.

### 6.5 Weaknesses

1. **Ruling out human is the weak side:** not-human recall is 0.84.
2. **Precision depends on the human share** of the data being labeled (see section 7).
3. **Validation is probably optimistic.** Validation text is close to training text, BioProject coverage is low, and unseen organisms haven't been tested.
4. **Label noise limits the measurable score.** The labels come from `host_name` rules, and there are 514k near-identical pairs with opposite labels.
5. **It misses human terms in non-human contexts.** In the explicit-terms slice, macro F1 is 0.56: the roughly 100 not-human rows there are mostly called human.
6. **Model capacity is limited.** It's a single linear layer on general-purpose embeddings. It handles negation and multi-part phrasing poorly, and text is cut at 256 tokens. Median text is only 65–106 characters, so the cutoff rarely matters.
7. **Tuning targets human-class F1,** which favors the majority class.
8. **Mild overfitting.** C = 3.16 is fairly weak regularization, consistent with picking up study-specific wording.

---

## 7. Use as a high-confidence "is human" filter

The criteria that matter for a filter:
- **Precision of what passes:** false positives carry a "human" label into later stages.
- **Coverage:** the share of human rows labeled. Anything missed just goes on to later stages.
- **Representativeness** of the passed rows.
- **Stability** on new data.

| Cutoff | Rows labeled human | Precision | 95% lower bound | Human rows labeled | Not-human rows let through |
|---|---|---|---|---|---|
| p ≥ 0.90 | 83,826 | 0.974 | 0.972 | 85.5% | 7.1% |
| p ≥ 0.95 | 78,129 | 0.978 | 0.977 | 80.1% | 5.5% |
| p ≥ 0.99 | 60,296 | 0.981 | 0.980 | 62.0% | 3.7% |
| p ≥ 0.995 | 49,657 | 0.983 | 0.981 | 51.1% | 2.8% |
| p ≥ 0.9986 (stage-2 default) | 23,766 | 0.980 | 0.978 | 24.4% | 1.5% |
| p ≥ 0.9995 | 13,928 | 0.987 | 0.985 | 14.4% | 0.6% |
| Keyword rule alone | 25,512 | 0.996 | | 26.6% | |
| **Keyword or p ≥ 0.99** | 63,685 | 0.981 | 0.980 | **65.5%** | |

**Findings:**
- **Measured precision levels off near 98%.** A higher cutoff costs a lot of coverage with almost no measured gain.
- **The stage-2 default cutoff is a poor choice for a filter.** The model alone at 0.9986 does worse than the keyword rule alone. Keyword or p ≥ 0.99 labels 2.7× more rows at the same measured precision.
- **Precision depends on the human share** of the data being filtered. At the 0.9986 cutoff, the same error rates give about 87% precision at 30% human and about 64% at 10% human. These are estimates from the validation error rates.
- **The passed rows are skewed by organism:** *M. tuberculosis* is 15.4% of them vs. 6.3% of all human rows. Later stages would see a skewed organism mix.
- **The cutoffs were picked on validation,** so they need confirming on test.

---

## 8. Label-error review

The file `reports_updated/mislabel_review.csv` holds 30 rows labeled not human with high model scores. There are 10 each in three bands (≥ 0.98, 0.90–0.98, 0.80–0.90), with suspect host names first and a blank `is_human_review` column.

### 8.1 Correction to an earlier claim

Earlier I said the highest-confidence errors looked mostly like label errors. That was based on the text alone. With `host_name` visible, **most of them are animal hosts** (Pig, Swine, Chicken, Dog, Mus musculus), so their labels are correct. The text sounds clinical: rectal swabs, hospital or university names.

| Band | Rows labeled not human | Host name looks like a person, subject ID or specimen |
|---|---|---|
| ≥ 0.98 | 1,358 | 180 (13%) |
| 0.90–0.98 | 863 | 36 (4%) |
| 0.80–0.90 | 669 | 15 (2%) |

So the ~98% precision plateau is mostly real model error, not mainly a labeling artifact.

### 8.2 Patterns the labeling rule misses

| Pattern | Examples | Validation rows (score ≥ 0.80) |
|---|---|---|
| Subject or patient IDs | `CF-002` … `CF-022`, `Subject 12`, `volunteer 1`, `Infant` | ~130 for CF-0xx alone |
| Specimen in the host field | `blood`, `sputum` | 12+ |
| Disease in the host field | `Invasive pneumococcal disease` | 3 |
| Organism's own name as host | *S.* Typhi (26), *S.* Paratyphi A (6). Both infect only humans, so these rows are human. *S.* Typhimurium ATCC 14028 is a lab strain, not human | 32 |
| Ambiguous | `clinical/host-associated`, `w4`/`w5`, `1122` | — |
| Correctly not human, despite the flag | `Hospital environment` (ICU door handle), `dairy cattle (with subclinical mastitis)` | — |

**Root cause:** the notebook marks anything that isn't matched as human and isn't a placeholder as -1.

### 8.3 Typical real model errors

Animal and veterinary samples with clinical-sounding text:
- Thai hospital rectal swabs from pigs and chickens
- Dog anal swabs
- Mouse gut studies
- Dolphin septicemia
- Bovine fecal samples

The model can't separate these from human samples without the host field.

---

## 9. Imputing host for empty `host_name` rows

### 9.1 Should the model see `host_name` at prediction time? No.

- **The model can't use an input it never saw in training.** Adding `host_name` text at prediction time changes the embedding in ways the classifier never learned. The effect is unpredictable and can't be measured.
- **Training with `host_name` doesn't help.** The labels come from `host_name`, so the model would learn to copy the rule. Validation would look excellent while the model got weaker on empty-host rows.
- **When `host_name` is present, imputation isn't needed.** The rule is the precise answer for those rows.

**Suggested setup:**

| Situation | How to label |
|---|---|
| `host_name` present | Labeling rule, with the fixes from section 8. Model disagreements go to review as a quality check |
| `host_name` empty | Model trained without `host_name`, as now. Accept only high-confidence predictions; leave the rest unknown |

### 9.2 The empty-host rows differ from the training rows

| | Labeled rows (`host_name` present) | Empty-host rows (imputation target) |
|---|---|---|
| Has `host_comments` | **48.3%** | **0.7%** |
| Has `isolation_comments` | 100% | 99.8% |
| Has `comments` | 2.0% | 13.9% |
| Contains human/patient/homo | 18.5% | 3.4% |
| Median text length | 65 chars | 106 chars |

Current validation (F1 0.953) overstates imputation performance:
- Nearly half the validation rows carry `host_comments`, which the target rows almost never have.
- Explicit human terms, which make prediction easy, are five times rarer in the target rows.

The "no `host_comments`" slice (F1 0.943) is a better proxy, but still optimistic.

### 9.3 Isolation source as the main signal

Your hypothesis that `isolation_comments` is dense and informative about the host fits what we've seen:
- **It's nearly the only field available where it matters:** present in 100% of labeled rows and 99.8% of target rows.
- **The model already does well without `host_comments`:** F1 0.943, so most of the signal must come from isolation text, given `comments` is rare.
- **The review examples were mostly decided by isolation content:** sputum, blood, rectal swab, stool, hospital and institution names, strain IDs.
- **The new export packs several attributes into this one field,** joined with ` | `: strain ID, institution, sample type, source. That's why it's dense.

It's also where the confusions come from. Institution names and sample types ("rectal swab", "Siriraj Hospital") are shared by human and veterinary studies. So the field is informative but ambiguous for exactly the hard cases.

**Not yet measured directly.** An isolation-only training ablation (section 10) would quantify it.

---

## 10. Recommendations (priority order)

**Labels**
1. **Fix the -1 rule in `human_segregation.ipynb`.** Send unmatched `host_name` values to 0 or to review instead of -1, and add rules for the patterns in section 8.2. Also add "not applicable" to `missing_labels`, so those rows become 0 instead of -1 rows that prep then drops. This likely improves both training and evaluation more than any model change.
2. **Fill in `is_human_review`** in `mislabel_review.csv` to confirm the patterns and estimate how often the rule is wrong.

**Evaluation**

3. **Score the test set and the held-out-organism split.** `run.py` has no test-evaluation stage yet. This is the main missing result.
4. **Build an evaluation that looks like the imputation target.** Re-score validation with `host_comments` removed from the input, and report it next to the standard numbers.
5. **Hand-audit 100–200 high-confidence predictions on empty-host rows.** This is the only direct measure of imputation precision. Set the acceptance cutoff from it.

**Model**

6. **Match training to use:** drop `host_comments` from the input, or blank it at random for a large share of training rows.
7. **Run an isolation-only ablation:** train on `isolation_comments` alone to measure how much signal it carries (section 9.3).
8. **Consider a different tuning target** if missing not-human samples is costly: macro F1 or not-human precision instead of human-class F1.

**Filter**

9. **Use keyword or p ≥ ~0.99 rather than 0.9986,** confirmed by the audit.
10. **Pick the cutoff for the human share of the data being filtered.**
11. **If later stages can see `host_name`, cross-check it** against an animal-name list. This removes most remaining false positives.
12. **Track the organism mix** of the filtered output.

**Infrastructure**

13. **Reduce `split`'s peak RAM (23.2 GB)** before scaling up, for example by streaming near-duplicate pairs into a union-find instead of collecting all 870M.
14. **Minor fixes:** the hardcoded split path string in `run.py`, and the package that doesn't load (`pip install -e .` in the env).

---

## 11. Files

| Path | Contents |
|---|---|
| `reports_updated/session_findings.md` | This document |
| `reports_updated/run_report.md` | Run report (attempts, stage results) |
| `reports_updated/run_attempt1.log`, `run.log` | Full logs |
| `reports_updated/leakage_report.json` | Split and leakage checks |
| `reports_updated/stage1_metrics.json` | All stage-1 metrics |
| `reports_updated/stage1_training.png`, `stage1_validation.png` | Figures |
| `reports_updated/mislabel_review.csv` | 30 rows for label review |
| `artifacts_updated/stage1/model.joblib` | Trained model |
| `artifacts_updated/stage1/val_scores.parquet` | Validation scores per accession |
| `data/processed_updated/`, `data/splits_updated/`, `data/embeddings_updated/` | Prep outputs, frozen split, embedding cache (1,463,276 texts) |
