# Problem definition: income prediction on UCI Adult

**Task.** Predict whether a person's yearly income exceeds 50,000 USD from 1994 US Census attributes. This is a binary classification problem on tabular data with mixed numerical and categorical columns, and the purpose here is to build a trustworthy data pipeline, not to find the best classifier.

**Inputs.** 14 features per person: age, workclass, fnlwgt (census sampling weight), education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week and native-country. Six are integers and eight are categorical. Missing values are written `?` (workclass, occupation, native-country).

**Output.** `income` with two classes, `<=50K` and `>50K`. About 24% of the 48,842 rows belong to the positive class `>50K`, so always predicting `<=50K` already reaches about 76% accuracy.

**Success metric.** F1 of the positive class (`>50K`) and accuracy, both on a held-out test set. Accuracy alone would reward the trivial majority prediction, so F1 is the primary number and accuracy is reported next to it.

**Constraints.**
1. *No leakage.* The 70/30 stratified split happens before any cleaning. Imputation values, clipping bounds, scaling and encoding categories are learned from the training set only. Duplicates are dropped from train only, and the test set stays untouched.
2. *Reproducible pipeline.* One command runs the whole pipeline from raw data to processed tables, with a fixed seed, a configuration file, a verified raw-file hash and a SHA256 of the processed data.
3. *Auditable data.* Every validation rule reports violations instead of silently dropping rows, and the dataset is documented in a data card (source, licence, biases, privacy).

**Out of scope.** The model is not meant to support real decisions about people. The data are from 1994, contain sensitive attributes (sex, race, native country) and reflect historical income gaps.
