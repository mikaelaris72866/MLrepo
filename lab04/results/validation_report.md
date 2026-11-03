# Validation report

Rows: 48842, columns: 15. Hard-rule violations: **0**. No row was dropped or modified.

Missing marker `?` per column: {'workclass': 2799, 'occupation': 2809, 'native-country': 857}

Label counts: {'<=50K': 37155, '>50K': 11687}

| Check | Kind | Result | Count | Detail |
|---|---|---|---|---|
| row count | violation | pass | 0 | 48842 rows |
| column names and order | violation | pass | 0 | 15 columns |
| types (int columns parse as integers, category columns non-empty) | violation | pass | 0 | all 15 columns OK |
| range age in [17, 90] | violation | pass | 0 | observed [17, 90] |
| range hours-per-week in [1, 99] | violation | pass | 0 | observed [1, 99] |
| range capital-gain in [0, 99999] | violation | pass | 0 | observed [0, 99999] |
| range capital-loss in [0, 4356] | violation | pass | 0 | observed [0, 4356] |
| categories of workclass within data dictionary (+'?') | violation | pass | 0 |  |
| categories of education within data dictionary (+'?') | violation | pass | 0 |  |
| categories of marital-status within data dictionary (+'?') | violation | pass | 0 |  |
| categories of occupation within data dictionary (+'?') | violation | pass | 0 |  |
| categories of relationship within data dictionary (+'?') | violation | pass | 0 |  |
| categories of race within data dictionary (+'?') | violation | pass | 0 |  |
| categories of sex within data dictionary (+'?') | violation | pass | 0 |  |
| categories of native-country within data dictionary (+'?') | violation | pass | 0 |  |
| categories of income within data dictionary (+'?') | violation | pass | 0 |  |
| null / empty cells other than the '?' marker | violation | pass | 0 |  |
| exact duplicate rows (all 15 columns) | info | note | 52 | kept in the data; dropped from train only after the split |
| duplicate feature rows (14 features, any label) | info | note | 57 |  |
| top-coded value age == 90 | info | note | 55 | maximum of the allowed range; probably a top-code ('or more'), which the data dictionary does not document |
| top-coded value hours-per-week == 99 | info | note | 137 | maximum of the allowed range; probably a top-code ('or more'), which the data dictionary does not document |
| top-coded value capital-gain == 99999 | info | note | 244 | maximum of the allowed range; probably a top-code ('or more'), which the data dictionary does not document |
| education <-> education-num is one-to-one | warning | pass | 0 |  |
| relationship Husband/Wife agrees with sex | warning | note | 4 |  |
| bachelor's degree or higher at age < 19 | warning | note | 1 |  |
| workclass and occupation missing together | warning | note | 10 | rows where exactly one of the two is '?' |
| label values are '<=50K' or '>50K' | violation | pass | 0 |  |
