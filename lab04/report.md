# Lab 4: Data pipeline and quality

Run everything with one command from the repository root:

```
python lab04/src/run_pipeline.py        # about 15 seconds
python -m pytest lab04/tests            # 9 tests
```

Code: `src/lib.py` (building blocks), `src/run_pipeline.py` (orchestration), `src/data_card.py`. Settings: `config.yaml`. Outputs: `results/` (validation report, leakage checks, baseline metrics, label audit, `preprocessor.joblib`, `run_manifest.yaml`). Documents: `problem_definition.md`, `data_card.md`.

## 1. Problem definition (item 1)

See `problem_definition.md` (half a page). In short: inputs are the 14 census features, the output is `income > 50K`, success is judged by F1 of the positive class and by accuracy on a held-out test set (the majority-class baseline is already 76.1% accuracy, which is why F1 comes first), and the constraints are no leakage and a reproducible pipeline.

## 2. Ingestion (item 2)

The manual asks for `fetch_openml(name='adult', version=2)`. **OpenML could not be reached from the environment where I built the pipeline** (the request was blocked by the network proxy), so by default the pipeline reads the same UCI Adult data as a CSV from GitHub (`jbrownlee/Datasets/adult-all.csv`), downloaded once into `data/adult/` and verified by SHA256 (`21e0ea2f...218c2`) on every run. I checked that it matches the properties of the UCI data: 48,842 rows, 15 columns, 37,155 `<=50K` and 11,687 `>50K` labels, 2,799 / 2,809 / 857 missing values in workclass / occupation / native-country, 52 exact duplicates, and value ranges exactly as in the manual's schema.

The OpenML path is implemented (`--source openml`) and maps its two representation differences (NaN for missing, target named `class`) to the CSV convention. **I could only test it with a simulated OpenML-shaped frame**, not against the live service, so please run `python lab04/src/run_pipeline.py --source openml` on your machine. If the row order of the OpenML copy differs from the CSV, the split and therefore the processed hash will differ; in that case record that run's hash with `--update-hash`.

Ingestion reads every cell as a string and changes nothing. It logs the row and column counts and stops with an error if they are not 48,842 and 15:

```
[pipeline] ingested 48842 rows x 15 columns (source=csv, sha256=21e0ea2f925a...)
```

## 3. Validation (item 3)

Schema as in the manual: 6 integer columns and 9 category columns (8 features plus the label). Ranges: age 17-90, hours-per-week 1-99, capital-gain 0-99999, capital-loss 0-4356. `?` is the missing marker and an allowed category. Category lists come from the UCI data dictionary. Full report: `results/validation_report.md` (and `.json`).

| Check | Violations |
|---|---|
| Row and column count, column names/order | 0 |
| Types (integers parse, categories non-empty) | 0 |
| Ranges (age, hours, capital-gain, capital-loss) | 0 |
| Categories within data dictionary (8 features + label) | 0 |
| Null/empty cells other than `?` | 0 |
| Label values `<=50K` / `>50K` | 0 |
| **Total hard-rule violations** | **0** |

Findings that are not rule violations (reported, not acted on during validation):
- **Missing `?`:** workclass 2,799 (5.7%), occupation 2,809 (5.8%), native-country 857 (1.8%). Workclass and occupation are missing together except in 10 rows.
- **Exact duplicate rows:** 52 (57 rows duplicate on the 14 features).
- **Values at the top of their range:** age 90 (55 rows), hours-per-week 99 (137 rows), capital-gain 99999 (244 rows). These look like "or more" codes, which the data dictionary does not mention. They are inside the allowed ranges, so they are not violations.
- **Plausibility warnings:** 4 rows where relationship (Husband/Wife) disagrees with sex, 1 row with a bachelor's degree or higher at age under 19. Education and education-num are one-to-one.

Nothing was dropped or modified in this step. A test (`test_validation_detects_injected_violations`) injects bad values and checks that the validator reports them without dropping rows.

## 4. Cleaning and transformation (item 4)

Order of operations: split, then drop duplicates from train, then fit everything else on the train set only.

1. **Split** 70/30 stratified on income (`random_state=42`) on the raw rows, before any cleaning: 34,189 train, 14,653 test.
2. **Duplicates:** 23 exact duplicates dropped from train (34,166 rows remain). Test is untouched (14,653 rows).
3. **Imputer** (fit on train): `?` replaced by the train mode for the 8 categorical columns (e.g. workclass `Private`, occupation `Prof-specialty`, native-country `United-States`); median for numeric columns. The numeric columns have no missing values in this dataset, so the median step changes nothing.
4. **Clipping** of the 6 numeric columns to the train 1st/99th percentiles (e.g. age [17, 74], hours [8, 80], capital-gain [0, 15,024], capital-loss [0, 1,980], fnlwgt [27,050, 509,161]).
5. **Encoding** for the model: standard scaling of numeric columns and one-hot encoding of categorical columns, both fitted on train (unseen categories are ignored).
6. The fitted pipeline is saved with `joblib.dump` to `results/preprocessor.joblib`.

Two choices of mine, which the manual does not specify: duplicates are dropped *before* fitting the imputer and clipping bounds (so duplicate rows do not influence the statistics), and the derived features of the next section are computed from the cleaned (imputed and clipped) columns.

**Leakage checks** (`results/leakage_checks.json`, run on every pipeline run):
- Train and test indices do not overlap (0 shared rows).
- The test set has the same number of rows before and after cleaning (14,653).
- The clip bounds equal an independent recomputation from the train rows only.
- The checks can detect a leak: had the test rows been included in the fit, the upper clipping bounds of fnlwgt and capital-loss would have been different.
- Unit tests (`test_cleaner_statistics_come_from_fit_data_only` and others) verify the same properties on synthetic data.

## 5. Derived features (item 5)

`capital_net = capital-gain - capital-loss` and `hours_per_age = hours-per-week / (age - 16)`. The minimum age is 17, so the denominator is at least 1. Logistic regression (`max_iter=1000`), same preprocessing and split for every row. Accuracy and F1 are for the positive class on the 14,653 test rows; the CV columns are 5-fold cross-validation on the training set (the whole pipeline is refitted inside each fold, so there is no leakage across folds).

| Derived features | Test accuracy | Test F1 | CV accuracy | CV F1 |
|---|---|---|---|---|
| none | 0.8518 | 0.6570 | 0.8506 +- 0.0034 | 0.6550 +- 0.0096 |
| capital_net | 0.8512 | 0.6553 | 0.8509 +- 0.0035 | 0.6554 +- 0.0097 |
| hours_per_age | 0.8542 | 0.6640 | 0.8534 +- 0.0038 | 0.6624 +- 0.0104 |
| both | 0.8546 | 0.6650 | 0.8532 +- 0.0036 | 0.6617 +- 0.0096 |

Majority-class accuracy on the test set: 0.7607.

How much of the difference is real? The fold-to-fold spread is large compared with the gaps, so I compared each variant with "none" on the *same* folds and with McNemar's exact test on the test predictions (`results/baseline_metrics.csv`):

| Derived features | F1 gain, CV (mean +- s.e.) | Accuracy gain, CV | McNemar p (test) |
|---|---|---|---|
| capital_net | +0.0005 +- 0.0003 | +0.0003 +- 0.0001 | 0.096 |
| hours_per_age | +0.0075 +- 0.0016 | +0.0028 +- 0.0009 | 0.047 |
| both | +0.0068 +- 0.0016 | +0.0025 +- 0.0008 | 0.023 |

So `capital_net` makes no meaningful difference, and `hours_per_age` gives a small but consistent gain (about 0.3 points of accuracy and 0.7 points of F1). The test-set p-values are borderline, and three variants were compared, so I would call the evidence suggestive rather than strong. The likely reason (checked for `capital_net`, not tested for the other): `capital_net` is a linear combination of two columns the logistic regression already has, so it adds no expressive power (gain and loss are never both non-zero: 0 of 48,842 rows), whereas `hours_per_age` is a ratio, a non-linear combination that a linear model cannot form from its two inputs alone.

## 6. Label quality (item 6)

**Duplicate-conflict audit** (`results/label_audit.json`), on all 48,842 rows:
- Rows with identical values on all 14 features but different `income` labels: **5 groups (10 rows)**.
- If the census weight `fnlwgt` is ignored (13 features), **1,029 groups (3,873 rows, 7.9% of the data)** have identical demographics but different labels. This is a ceiling on how well any model can do, and it is not an error in the data: two different people can have the same description and different incomes.
- Train/test: 28 test rows have an identical-feature twin in train; 27 with the same label, 1 with a conflicting label. Scores without those 28 rows are almost unchanged (accuracy 0.8516 vs 0.8518 for the baseline), so this does not materially affect the results above.

**50-row check** (`results/label_audit_sample.csv`, seed 42, drawn from all rows): I read the 50 rows against the data dictionary and applied rule checks (ranges, category lists, education vs education-num, Husband/Wife vs sex, degree-under-19). **No sampled row violates a rule.** Observations: 2 of the 50 rows have `?` in workclass and occupation, and one is age 90 (the top-coded maximum). Across all rows the rule checks find 4 Husband/Wife-sex mismatches and 1 degree-under-19 row, none of which were in the sample. *The manual asks you to do this manual re-check yourself: open `label_audit_sample.csv` and go through it.*

## 7. Hash, configuration and data card (item 7)

- **Processed dataset:** cleaned and feature-engineered train and test tables (before scaling and one-hot encoding), written with fixed formatting (floats rounded to 6 decimals, LF line endings) to `data/processed/`. SHA256 of the two files combined: `39e2779b0e0b4af98b48f00430c67778bc04f6e0c97678a970c94f15efd9d48d` (train and test hashes are in `results/run_manifest.yaml`).
- **Configuration:** `config.yaml` holds every setting and the expected hash; `results/run_manifest.yaml` records the run (command, git commit, library versions, hashes, counts, full config copy).
- **Reproducibility:** rerunning the command prints `processed hash matches config: reproducible`. I checked this from a different working directory, and in a fresh `git clone` with no data and no results: the pipeline downloaded the raw file, passed the raw-file hash check, ran with `--strict` and reproduced the identical processed SHA256 (same machine, same library versions). `--strict` makes a mismatch an error. Floating-point results can differ in the last digits on another operating system or library version; the hash uses 6 decimals to make that unlikely but not impossible, and the run manifest records versions so you can see why.
- **Data card:** `data_card.md` (generated from the run, covering source, licence CC BY 4.0, size, preparation, biases, privacy, intended use).

## 8. Conclusions (draft; rewrite in your own words)

1. The data pass every hard validation rule (0 violations) but have real quality issues that validation reports rather than fixes: missing values concentrated in workclass/occupation, possible top-coded values, 52 duplicates, and about 8% of rows sharing identical demographics with a different label.
2. Because all statistics (modes, medians, clip bounds, scaler, encoder categories) are learned from the training rows only, and the test rows are never touched, the test scores of about 85% accuracy and 0.66 F1 are honest estimates; the leakage checks show they would have differed had the test rows been used.
3. Derived features are only worth keeping if they help on held-out data: `hours_per_age` helps slightly, `capital_net` does not.

## Caveats and things I could not verify

- The default data source is a GitHub CSV copy of the UCI data, not OpenML (see section 2); `--source openml` is untested against the live service.
- Single split and seed; the baseline is plain logistic regression with no tuning.
- Missing values are filled with the mode, which is probably wrong for many rows (missingness is related to the label, see the data card).
- The dataset licence and citation come from the UCI repository page, which I read through a search tool; check them against the page yourself before submitting.
