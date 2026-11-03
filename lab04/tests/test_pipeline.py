import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lab04" / "src"))
import lib  # noqa: E402

CFG = yaml.safe_load((ROOT / "lab04" / "config.yaml").read_text())
RAW_PATH = ROOT / CFG["data"]["raw_path"]
needs_data = pytest.mark.skipif(not RAW_PATH.exists(), reason="raw data not downloaded yet")


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(20, 60, n) for c in lib.INT_COLS})
    for c in lib.CAT_COLS:
        df[c] = rng.choice(["a", "b", "c"], n)
    df["workclass"] = df["workclass"].where(rng.random(n) > 0.2, "?")
    df["income"] = rng.choice(["<=50K", ">50K"], n)
    return df


def test_cleaner_statistics_come_from_fit_data_only():
    train, test = toy(200, 0), toy(100, 1)
    test.loc[test.index[:5], "capital-loss"] = 10_000_000  # extreme values only in test
    c = lib.Cleaner().fit(train[lib.FEATURES])
    assert c.hi_["capital-loss"] < 100            # not influenced by the test outliers
    both = lib.Cleaner().fit(pd.concat([train, test])[lib.FEATURES])
    assert both.hi_["capital-loss"] != c.hi_["capital-loss"]  # the check is sensitive to leakage
    out = c.transform(test[lib.FEATURES])
    assert out["capital-loss"].max() <= c.hi_["capital-loss"]


def test_imputation_uses_train_mode_and_removes_missing_marker():
    train, test = toy(300, 0), toy(100, 1)
    c = lib.Cleaner().fit(train[lib.FEATURES])
    non_missing = train.loc[train["workclass"] != "?", "workclass"]
    assert c.modes_["workclass"] == non_missing.mode().iloc[0]
    out = c.transform(test[lib.FEATURES])
    assert not (out[lib.CAT_COLS] == "?").any().any()


def test_derived_features_formulas():
    df = pd.DataFrame({"capital-gain": [100.0, 0.0], "capital-loss": [30.0, 50.0],
                       "hours-per-week": [40.0, 20.0], "age": [26.0, 17.0]})
    out = lib.FeatureAdder().transform(df)
    assert out["capital_net"].tolist() == [70.0, -50.0]
    assert out["hours_per_age"].tolist() == [4.0, 20.0]


def test_split_dedupes_train_only_and_keeps_test_untouched():
    df = toy(400, 0)
    df = pd.concat([df, df.iloc[:60]], ignore_index=True)  # 60 exact duplicates
    cfg = yaml.safe_load((ROOT / "lab04" / "config.yaml").read_text())
    train, test, info = lib.split_and_dedupe(df, cfg)
    assert info["test_rows"] == len(test) == round(0.3 * len(df))
    assert not train.duplicated().any()
    assert set(train.index).isdisjoint(test.index)
    assert info["train_rows"] == len(train)


def test_unseen_category_does_not_break_encoder():
    train, test = toy(200, 0), toy(50, 1)
    test.loc[test.index[0], "race"] = "never-seen"
    cfg = yaml.safe_load((ROOT / "lab04" / "config.yaml").read_text())
    y = (train["income"] == ">50K").astype(int)
    model = lib.make_model(cfg, ["capital_net", "hours_per_age"]).fit(train[lib.FEATURES], y)
    assert len(model.predict(test[lib.FEATURES])) == 50


def test_openml_shaped_frame_maps_to_same_schema(monkeypatch, tmp_path):
    """OpenML marks missing values as NaN and calls the label 'class'; ingestion maps both."""
    base = toy(60, 0)
    om = base.rename(columns={"income": "class"}).copy()
    om["workclass"] = om["workclass"].replace("?", np.nan)
    for c in lib.CAT_COLS + ["class"]:
        om[c] = om[c].astype("category")

    class Bunch:
        frame = om

    import sklearn.datasets as ds

    monkeypatch.setattr(ds, "fetch_openml", lambda **kw: Bunch)
    cfg = yaml.safe_load((ROOT / "lab04" / "config.yaml").read_text())
    cfg["data"]["expected_rows"], cfg["data"]["expected_cols"] = 60, 15
    raw, info = lib.ingest(cfg, tmp_path, source="openml", log=lambda *_: None)
    assert list(raw.columns) == lib.COLUMNS
    assert (raw["workclass"] == "?").sum() == (base["workclass"] == "?").sum()
    assert raw.equals(base[lib.COLUMNS].astype(str))


def test_processed_csv_hash_is_deterministic(tmp_path):
    df = toy(50, 0)
    df["x"] = np.linspace(0, 1, len(df)) / 3
    h1 = lib.write_processed(df, tmp_path / "a.csv")
    h2 = lib.write_processed(df.copy(), tmp_path / "b.csv")
    assert h1 == h2


@needs_data
def test_real_data_shape_schema_and_no_hard_violations():
    raw, info = lib.ingest(CFG, ROOT, log=lambda *_: None)
    assert raw.shape == (48842, 15)
    typed, rep = lib.validate(raw, CFG)
    assert rep["n_hard_violations"] == 0
    assert rep["missing_marker_counts"] == {"workclass": 2799, "occupation": 2809, "native-country": 857}
    assert typed.shape == raw.shape


@needs_data
def test_validation_detects_injected_violations():
    raw, _ = lib.ingest(CFG, ROOT, log=lambda *_: None)
    bad = raw.copy()
    bad.loc[0, "age"] = "5"            # below range
    bad.loc[1, "hours-per-week"] = "120"
    bad.loc[2, "sex"] = "Unknown"      # category outside dictionary
    bad.loc[3, "capital-gain"] = "abc"  # not an integer
    typed, rep = lib.validate(bad, CFG)
    by = {c["check"]: c["n"] for c in rep["checks"]}
    assert by["range age in [17, 90]"] == 1
    assert by["range hours-per-week in [1, 99]"] == 1
    assert by["categories of sex within data dictionary (+'?')"] == 1
    assert rep["n_hard_violations"] >= 4
    assert len(typed) == len(bad)      # nothing dropped
