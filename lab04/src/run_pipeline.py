"""Lab 4: one-command, leakage-free data pipeline for UCI Adult.

    python lab04/src/run_pipeline.py [--source csv|openml] [--strict] [--skip-card] [--update-hash]

Steps: ingest -> validate -> split (70/30 stratified) -> dedupe train -> fit cleaner on train
-> derived features -> processed tables + SHA256 -> logistic-regression baselines (with/without
derived features) -> label-quality audit -> manifest and data card.
"""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

import lib

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "lab04" / "config.yaml"


def log(msg):
    print(f"[pipeline] {msg}", flush=True)


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csv", "openml"], default="csv")
    ap.add_argument("--strict", action="store_true", help="exit 1 if the processed hash differs from config")
    ap.add_argument("--skip-card", action="store_true", help="do not (re)write lab04/data_card.md")
    ap.add_argument("--update-hash", action="store_true", help="store this run's hash as expected_processed_sha256")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    seed = cfg["seed"]
    res = ROOT / cfg["outputs"]["results_dir"]
    proc = ROOT / cfg["outputs"]["processed_dir"]
    res.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # 1. ingestion (raw, unmodified strings) ---------------------------------------------
    raw, ing = lib.ingest(cfg, ROOT, args.source, log)

    # 2. validation (reports, never drops) -----------------------------------------------
    typed, rep = lib.validate(raw, cfg)
    jdump(rep, res / "validation_report.json")
    (res / "validation_report.md").write_text(lib.report_markdown(rep))
    log(f"validation: {rep['n_hard_violations']} hard-rule violations; missing '?': {rep['missing_marker_counts']}")

    # 3. split BEFORE cleaning, then drop exact duplicates from train only ---------------
    train, test, split = lib.split_and_dedupe(typed, cfg)
    log(f"split: train {split['train_rows_before_dedupe']} -> {split['train_rows']} after dropping "
        f"{split['train_duplicates_dropped']} exact duplicates; test {split['test_rows']} (untouched)")
    overlap_info, test_has_twin = lib.cross_split_conflicts(train, test)
    Xtr, ytr = train[lib.FEATURES], (train[lib.TARGET] == cfg["data"]["positive_label"]).astype(int)
    Xte, yte = test[lib.FEATURES], (test[lib.TARGET] == cfg["data"]["positive_label"]).astype(int)

    # 4. fit preprocessing on train only; persist -----------------------------------------
    derived = cfg["features"]["derived"]
    pre = lib.make_preprocessor(cfg, derived).fit(Xtr)
    joblib.dump(pre, res / "preprocessor.joblib")
    cleaner = pre.named_steps["clean"]

    # leakage checks ---------------------------------------------------------------------
    idx_overlap = len(set(train.index) & set(test.index))
    recomputed_lo = lib.Cleaner(missing=cfg["data"]["missing_token"]).fit(Xtr).lo_
    with_test = lib.Cleaner(missing=cfg["data"]["missing_token"]).fit(pd.concat([Xtr, Xte])).hi_
    differs = [c for c in lib.INT_COLS if with_test[c] != cleaner.hi_[c]]
    leak = {
        "train_test_index_overlap": idx_overlap,
        "test_rows_before_and_after": [split["test_rows"], len(test)],
        "cleaner_fit_on_rows": len(Xtr),
        "clip_bounds_equal_train_only_recomputation": recomputed_lo == cleaner.lo_,
        "columns_whose_clip_upper_bound_would_change_if_test_were_included": differs,
        "imputation_modes_from_train": cleaner.modes_,
        "imputation_medians_from_train": cleaner.medians_,
        "clip_lower_from_train": cleaner.lo_,
        "clip_upper_from_train": cleaner.hi_,
        "test_exact_rows_also_in_train_after_dedupe": overlap_info["test_rows_with_feature_twin_in_train"],
    }
    assert idx_overlap == 0 and leak["clip_bounds_equal_train_only_recomputation"]
    jdump(leak, res / "leakage_checks.json")
    log(f"leakage checks passed (train/test index overlap {idx_overlap}); fitted statistics saved")

    # processed tables (cleaned + derived features, human readable) and SHA256 -------------
    from sklearn.pipeline import Pipeline

    tabular = Pipeline(pre.steps[:2])  # clean + derived features (before encoding/scaling)
    ptrain = tabular.transform(Xtr).assign(**{lib.TARGET: train[lib.TARGET].astype(str).values})
    ptest = tabular.transform(Xte).assign(**{lib.TARGET: test[lib.TARGET].astype(str).values})
    h_train = lib.write_processed(ptrain, proc / "processed_train.csv")
    h_test = lib.write_processed(ptest, proc / "processed_test.csv")
    combined = hashlib.sha256(f"{h_train}\n{h_test}\n".encode()).hexdigest()
    log(f"processed dataset SHA256 = {combined}")

    # 5. baselines with / without derived features ------------------------------------------
    cv = StratifiedKFold(cfg["model"]["cv_folds"], shuffle=True, random_state=seed)
    rows, fold_scores, preds = [], {}, {}
    sets = {"none": [], "capital_net": ["capital_net"], "hours_per_age": ["hours_per_age"],
            "both": list(derived)}
    keep = ~test_has_twin
    for name, d in sets.items():
        model = lib.make_model(cfg, d).fit(Xtr, ytr)
        pred = model.predict(Xte)
        scores = cross_validate(lib.make_model(cfg, d), Xtr, ytr, cv=cv, scoring=["accuracy", "f1"])
        fold_scores[name], preds[name] = scores, pred
        rows.append({
            "derived_features": name,
            "test_accuracy": accuracy_score(yte, pred), "test_f1": f1_score(yte, pred),
            "cv_accuracy_mean": scores["test_accuracy"].mean(), "cv_accuracy_std": scores["test_accuracy"].std(),
            "cv_f1_mean": scores["test_f1"].mean(), "cv_f1_std": scores["test_f1"].std(),
            "test_accuracy_without_train_twins": accuracy_score(yte[keep], pred[keep]),
            "test_f1_without_train_twins": f1_score(yte[keep], pred[keep]),
        })
        log(f"baseline [{name:13s}] test acc {rows[-1]['test_accuracy']:.4f}  F1 {rows[-1]['test_f1']:.4f}  "
            f"| 5-fold CV F1 {rows[-1]['cv_f1_mean']:.4f} +- {rows[-1]['cv_f1_std']:.4f}")
    # paired comparison against "none": same CV folds, and McNemar's exact test on the test set
    from scipy.stats import binomtest

    base_f, base_p = fold_scores["none"], preds["none"]
    for r in rows:
        f, p = fold_scores[r["derived_features"]], preds[r["derived_features"]]
        for m in ("f1", "accuracy"):
            d = f[f"test_{m}"] - base_f[f"test_{m}"]
            r[f"cv_{m}_diff_vs_none_mean"] = d.mean()
            r[f"cv_{m}_diff_vs_none_se"] = d.std(ddof=1) / np.sqrt(len(d))
        b = int(((base_p == yte) & (p != yte)).sum())   # right without, wrong with
        c = int(((base_p != yte) & (p == yte)).sum())   # wrong without, right with
        r["mcnemar_only_right_without"], r["mcnemar_only_right_with"] = b, c
        r["mcnemar_p"] = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    majority = max(yte.mean(), 1 - yte.mean())
    metrics = pd.DataFrame(rows)
    metrics.round(5).to_csv(res / "baseline_metrics.csv", index=False)
    jdump({"majority_class_test_accuracy": majority, "n_test_rows": len(test),
           "n_test_rows_without_train_twins": int(keep.sum())}, res / "baseline_reference.json")

    # 6. label-quality audit ----------------------------------------------------------------
    n = cfg["label_audit"]["sample_size"]
    sample = typed.sample(n, random_state=seed)
    flags = lib.row_rule_violations(sample, cfg)
    sample_out = sample.assign(rule_violations=flags.sum(axis=1).values,
                               rules_violated=flags.apply(lambda r: ",".join(r.index[r]), axis=1).values)
    sample_out.to_csv(res / "label_audit_sample.csv", index_label="row_id")
    all_flags = lib.row_rule_violations(typed, cfg)
    audit = {
        "sample_size": n,
        "sample_rows_with_rule_violation": int((flags.sum(axis=1) > 0).sum()),
        "sample_rows_missing_a_value": int((sample[lib.CAT_COLS] == cfg["data"]["missing_token"]).any(axis=1).sum()),
        "rule_violation_counts_all_rows": {c: int(v) for c, v in all_flags.sum().items() if v},
        "duplicate_conflicts_14_features": lib.duplicate_conflicts(typed, lib.FEATURES),
        "duplicate_conflicts_13_features_without_fnlwgt": lib.duplicate_conflicts(
            typed, [c for c in lib.FEATURES if c != "fnlwgt"]),
        "train_test_conflicts": overlap_info,
    }
    jdump(audit, res / "label_audit.json")
    log(f"label audit: {audit['duplicate_conflicts_14_features']['conflicting_groups']} conflicting duplicate "
        f"groups (14 features); {audit['duplicate_conflicts_13_features_without_fnlwgt']['conflicting_groups']} "
        f"without fnlwgt; {audit['sample_rows_with_rule_violation']}/{n} sampled rows violate a rule")

    # 7. manifest, hash verification, data card ---------------------------------------------
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python lab04/src/run_pipeline.py" + (f" --source {args.source}" if args.source != "csv" else ""),
        "git_commit": commit,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "ingestion": ing, "split": {**split, "seed": seed}, "validation_hard_violations": rep["n_hard_violations"],
        "processed": {"train_rows": len(ptrain), "test_rows": len(ptest), "columns": list(ptrain.columns),
                      "sha256_train": h_train, "sha256_test": h_test, "sha256_combined": combined},
        "config": cfg,
    }
    (res / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    if args.update_hash:
        text = CONFIG.read_text()
        text = re.sub(r"^expected_processed_sha256:.*$", f"expected_processed_sha256: {combined}", text, flags=re.M)
        CONFIG.write_text(text)
        log("expected_processed_sha256 written to config.yaml")
    exp = cfg.get("expected_processed_sha256")
    if exp is None:
        log("no expected hash in config (run with --update-hash to record this run's hash)")
    elif exp == combined:
        log("processed hash matches config: reproducible")
    else:
        log(f"WARNING: processed hash differs from config ({exp[:12]}... vs {combined[:12]}...)")
        if args.strict:
            sys.exit(1)

    if not args.skip_card:
        from data_card import write_card

        write_card(ROOT, cfg, typed, rep, split, audit, leak, metrics, manifest)
        log("data card written to lab04/data_card.md")
    log("done")


if __name__ == "__main__":
    main()
