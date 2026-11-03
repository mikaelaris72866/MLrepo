"""Building blocks of the Lab 4 pipeline: ingestion, validation, cleaning, features, audit."""
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
INT_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT_COLS = ["workclass", "education", "marital-status", "occupation", "relationship",
            "race", "sex", "native-country"]
TARGET = "income"
FEATURES = COLUMNS[:-1]  # the 14 input features
SCHEMA = {c: ("int" if c in INT_COLS else "category") for c in COLUMNS}

# Allowed categories from the UCI data dictionary ('?' = missing is always allowed).
ALLOWED = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
                  "State-gov", "Without-pay", "Never-worked"],
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm",
                  "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th",
                  "Doctorate", "5th-6th", "Preschool"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                   "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                   "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative",
                     "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": [
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
        "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba",
        "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico",
        "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan",
        "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
        "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
    "income": ["<=50K", ">50K"],
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ ingestion
def ingest(cfg, root, source="csv", log=print):
    """Read the raw data WITHOUT modifying it (everything stays a string).

    source 'csv'    : the UCI Adult CSV (downloaded once, SHA256 verified).
    source 'openml' : fetch_openml('adult', version=2) as in the manual.  OpenML marks missing
                      values as NaN and names the target 'class'; these two representation
                      differences are mapped to the CSV convention ('?' and 'income') so the
                      rest of the pipeline sees the same schema.  Nothing else is changed.
    """
    d = cfg["data"]
    info = {"source": source}
    if source == "openml":
        from sklearn.datasets import fetch_openml

        frame = fetch_openml(name="adult", version=2, as_frame=True, data_home=str(root / "data" / "openml")).frame
        frame = frame.rename(columns={"class": TARGET})
        raw = frame[COLUMNS].astype(object).where(frame[COLUMNS].notna(), d["missing_token"]).astype(str)
        info["raw_sha256"] = hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
    else:
        path = root / d["raw_path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            log(f"downloading raw data -> {path}")
            urllib.request.urlretrieve(d["raw_url"], path)
        digest = sha256_file(path)
        if digest != d["raw_sha256"]:
            raise RuntimeError(f"raw file hash mismatch: {digest} != {d['raw_sha256']}")
        raw = pd.read_csv(path, header=None, names=COLUMNS, dtype=str, keep_default_na=False)
        info["raw_sha256"] = digest
    info["rows"], info["cols"] = raw.shape
    log(f"ingested {info['rows']} rows x {info['cols']} columns (source={source}, sha256={info['raw_sha256'][:12]}...)")
    if (info["rows"], info["cols"]) != (d["expected_rows"], d["expected_cols"]):
        raise RuntimeError(f"unexpected shape {raw.shape}")
    return raw, info


# ------------------------------------------------------------------ validation
def _check(name, kind, n_bad, detail=""):
    """kind: 'violation' (hard rule), 'warning' (plausibility) or 'info'."""
    status = "pass" if n_bad == 0 else ("FAIL" if kind == "violation" else "note")
    return {"check": name, "kind": kind, "status": status, "n": int(n_bad), "detail": detail}


def validate(raw, cfg):
    """Run schema / range / missing / duplicate / label checks.  Never drops a row.

    Returns (typed_frame, report).  typed_frame has the manual's dtypes; a value that
    cannot be parsed becomes NaN (counted, never dropped).
    """
    miss = cfg["data"]["missing_token"]
    checks = []
    checks.append(_check("row count", "violation", raw.shape[0] != cfg["data"]["expected_rows"],
                         f"{raw.shape[0]} rows"))
    checks.append(_check("column names and order", "violation", list(raw.columns) != COLUMNS,
                         f"{raw.shape[1]} columns"))

    typed = pd.DataFrame(index=raw.index)
    bad_parse = {}
    for c in COLUMNS:
        if SCHEMA[c] == "int":
            num = pd.to_numeric(raw[c], errors="coerce")
            frac = (num.dropna() % 1 != 0).sum()
            bad_parse[c] = int(num.isna().sum() + frac)
            typed[c] = num.astype("int64") if bad_parse[c] == 0 else num
        else:
            bad_parse[c] = int((raw[c].str.strip() == "").sum())
            typed[c] = raw[c].astype("category")
    checks.append(_check("types (int columns parse as integers, category columns non-empty)", "violation",
                         sum(bad_parse.values()),
                         "; ".join(f"{c}: {n}" for c, n in bad_parse.items() if n) or "all 15 columns OK"))

    for c, (lo, hi) in cfg["validation"]["ranges"].items():
        n = int(((typed[c] < lo) | (typed[c] > hi)).sum())
        checks.append(_check(f"range {c} in [{lo}, {hi}]", "violation", n,
                             f"observed [{typed[c].min()}, {typed[c].max()}]"))

    for c, allowed in ALLOWED.items():
        ok = set(allowed) | {miss}
        outside = raw.loc[~raw[c].isin(ok), c]
        checks.append(_check(f"categories of {c} within data dictionary (+'{miss}')", "violation",
                             len(outside), ", ".join(sorted(outside.unique())[:8])))

    # missing values: '?' is the missing marker and is an allowed category
    missing = {c: int((raw[c] == miss).sum()) for c in COLUMNS}
    n_nan = int(raw.isna().sum().sum() + (raw == "").sum().sum())
    checks.append(_check("null / empty cells other than the '?' marker", "violation", n_nan))
    miss_nonzero = {c: n for c, n in missing.items() if n}

    # duplicates
    dup_all = int(raw.duplicated().sum())
    dup_feat = int(raw.duplicated(subset=FEATURES).sum())
    checks.append(_check("exact duplicate rows (all 15 columns)", "info", dup_all,
                         "kept in the data; dropped from train only after the split"))
    checks.append(_check("duplicate feature rows (14 features, any label)", "info", dup_feat))

    # top-coded / special values the data dictionary does not explain
    for c, v in [("age", 90), ("hours-per-week", 99), ("capital-gain", 99999)]:
        checks.append(_check(f"top-coded value {c} == {v}", "info", int((typed[c] == v).sum()),
                             "maximum of the allowed range; probably a top-code ('or more'), "
                             "which the data dictionary does not document"))

    # plausibility / consistency (data-dictionary rules)
    edu_map = typed.groupby("education", observed=True)["education-num"].nunique()
    inv_map = typed.groupby("education-num")["education"].nunique()
    checks.append(_check("education <-> education-num is one-to-one", "warning",
                         int((edu_map > 1).sum() + (inv_map > 1).sum())))
    rel_sex = ((typed["relationship"] == "Husband") & (typed["sex"] == "Female")) | \
              ((typed["relationship"] == "Wife") & (typed["sex"] == "Male"))
    checks.append(_check("relationship Husband/Wife agrees with sex", "warning", int(rel_sex.sum())))
    young_deg = (typed["education-num"] >= 13) & (typed["age"] < 19)
    checks.append(_check("bachelor's degree or higher at age < 19", "warning", int(young_deg.sum())))
    wc_occ = ((typed["workclass"] == miss) != (typed["occupation"] == miss))
    checks.append(_check("workclass and occupation missing together", "warning", int(wc_occ.sum()),
                         "rows where exactly one of the two is '?'"))

    n_label = int((~raw[TARGET].isin(ALLOWED[TARGET])).sum())
    checks.append(_check("label values are '<=50K' or '>50K'", "violation", n_label))

    report = {
        "rows": int(raw.shape[0]), "columns": int(raw.shape[1]),
        "missing_marker_counts": miss_nonzero,
        "label_counts": raw[TARGET].value_counts().to_dict(),
        "checks": checks,
        "n_hard_violations": int(sum(c["n"] for c in checks if c["kind"] == "violation")),
    }
    return typed, report


def report_markdown(rep):
    lines = ["# Validation report", "",
             f"Rows: {rep['rows']}, columns: {rep['columns']}. Hard-rule violations: **{rep['n_hard_violations']}**. "
             "No row was dropped or modified.", "",
             f"Missing marker `?` per column: {rep['missing_marker_counts']}", "",
             f"Label counts: {rep['label_counts']}", "",
             "| Check | Kind | Result | Count | Detail |", "|---|---|---|---|---|"]
    for c in rep["checks"]:
        lines.append(f"| {c['check']} | {c['kind']} | {c['status']} | {c['n']} | {c['detail']} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ split and duplicates
def split_and_dedupe(typed, cfg):
    """70/30 stratified split on the raw typed frame, then drop exact duplicates from train only."""
    from sklearn.model_selection import train_test_split

    s = cfg["split"]
    train, test = train_test_split(
        typed, test_size=s["test_size"], stratify=typed[TARGET] if s["stratify"] else None,
        random_state=cfg["seed"])
    n_train_raw = len(train)
    if cfg["cleaning"]["drop_duplicates_train_only"]:
        train = train.drop_duplicates()
    info = {"train_rows_before_dedupe": n_train_raw, "train_rows": len(train),
            "train_duplicates_dropped": n_train_raw - len(train), "test_rows": len(test)}
    return train, test, info


# ------------------------------------------------------------------ transformers (fit on train only)
class Cleaner(BaseEstimator, TransformerMixin):
    """Impute '?' (mode) and clip numeric columns to train percentiles.

    Everything learned in fit() (modes, medians, clip bounds) comes from the data passed
    to fit(), i.e. the training set only.
    """

    def __init__(self, cat_cols=tuple(CAT_COLS), num_cols=tuple(INT_COLS), missing="?",
                 lower=0.01, upper=0.99):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.missing = missing
        self.lower = lower
        self.upper = upper

    def _impute(self, X):
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].astype(object).where(X[c].astype(object) != self.missing, self.modes_[c])
        for c in self.num_cols:
            X[c] = X[c].astype(float).fillna(self.medians_[c])
        return X

    def fit(self, X, y=None):
        self.modes_ = {c: X.loc[X[c].astype(object) != self.missing, c].astype(object).mode().iloc[0]
                       for c in self.cat_cols}
        self.medians_ = {c: float(X[c].astype(float).median()) for c in self.num_cols}
        Xi = self._impute(X)
        q = Xi[list(self.num_cols)].quantile([self.lower, self.upper])
        self.lo_ = q.loc[self.lower].to_dict()
        self.hi_ = q.loc[self.upper].to_dict()
        return self

    def transform(self, X):
        X = self._impute(X)
        for c in self.num_cols:
            X[c] = X[c].clip(self.lo_[c], self.hi_[c])
        for c in self.cat_cols:
            X[c] = X[c].astype(object)
        return X


class FeatureAdder(BaseEstimator, TransformerMixin):
    """capital_net = capital-gain - capital-loss ; hours_per_age = hours-per-week / (age - 16).

    Computed from the cleaned (imputed and clipped) columns.  age >= 17 in the data, so the
    denominator is at least 1.
    """

    def __init__(self, derived=("capital_net", "hours_per_age")):
        self.derived = derived

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "capital_net" in self.derived:
            X["capital_net"] = X["capital-gain"] - X["capital-loss"]
        if "hours_per_age" in self.derived:
            X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
        return X


def make_encoder(derived):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = list(INT_COLS) + list(derived)
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])


def make_preprocessor(cfg, derived):
    from sklearn.pipeline import Pipeline

    lo, hi = cfg["cleaning"]["clip_percentiles"]
    return Pipeline([
        ("clean", Cleaner(missing=cfg["data"]["missing_token"], lower=lo, upper=hi)),
        ("features", FeatureAdder(derived=tuple(derived))),
        ("encode", make_encoder(derived)),
    ])


def make_model(cfg, derived):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pre = make_preprocessor(cfg, derived)
    return Pipeline(pre.steps + [("clf", LogisticRegression(
        max_iter=cfg["model"]["logistic_regression_max_iter"], random_state=cfg["seed"]))])


# ------------------------------------------------------------------ label quality
def duplicate_conflicts(df, columns):
    """Groups of rows with identical `columns` but more than one income label."""
    key = df[columns].astype(str).agg("|".join, axis=1)
    g = pd.DataFrame({"key": key.values, "y": df[TARGET].astype(str).values})
    n_labels = g.groupby("key")["y"].nunique()
    size = g.groupby("key")["y"].size()
    conflict = n_labels[n_labels > 1].index
    return {
        "groups_with_duplicates": int((size > 1).sum()),
        "conflicting_groups": int(len(conflict)),
        "rows_in_conflicting_groups": int(size[conflict].sum()),
        "share_of_all_rows_pct": round(100 * size[conflict].sum() / len(df), 3),
    }


def cross_split_conflicts(train, test):
    """Test rows whose 14 features appear in train with a different label (and with the same)."""
    kt = train[FEATURES].astype(str).agg("|".join, axis=1)
    ks = test[FEATURES].astype(str).agg("|".join, axis=1)
    lab = pd.DataFrame({"k": kt.values, "y": train[TARGET].astype(str).values}).groupby("k")["y"].agg(set)
    seen = ks.isin(lab.index)
    ty = test[TARGET].astype(str).values
    conflict = np.array([(k in lab.index) and (y not in lab[k]) for k, y in zip(ks, ty)])
    same = np.array([(k in lab.index) and (lab[k] == {y}) for k, y in zip(ks, ty)])
    return {"test_rows_with_feature_twin_in_train": int(seen.sum()),
            "twin_with_conflicting_label": int(conflict.sum()),
            "twin_with_same_label": int(same.sum())}, seen.values


def row_rule_violations(df, cfg):
    """Rule-based re-check of rows against the data dictionary (impossible values)."""
    miss = cfg["data"]["missing_token"]
    out = pd.DataFrame(index=df.index)
    for c, (lo, hi) in cfg["validation"]["ranges"].items():
        out[f"{c}_out_of_range"] = (df[c] < lo) | (df[c] > hi)
    for c, allowed in ALLOWED.items():
        out[f"{c}_bad_category"] = ~df[c].astype(str).isin(set(allowed) | {miss})
    out["education_num_mismatch"] = df["education"].astype(str).map(EDU_TO_NUM) != df["education-num"]
    out["husband_wife_vs_sex"] = ((df["relationship"] == "Husband") & (df["sex"] == "Female")) | \
                                 ((df["relationship"] == "Wife") & (df["sex"] == "Male"))
    out["degree_under_19"] = (df["education-num"] >= 13) & (df["age"] < 19)
    return out


EDU_TO_NUM = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
              "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
              "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16}


# ------------------------------------------------------------------ hashing
def write_processed(df, path):
    """Deterministic CSV (fixed float format, LF line endings) so the SHA256 is reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(6)
    out.to_csv(path, index=True, index_label="row_id", float_format="%.6f", lineterminator="\n")
    return sha256_file(path)
