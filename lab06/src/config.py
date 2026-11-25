"""YAML-driven configuration: load, override from the command line, validate, resolve paths."""
import copy
import hashlib
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "lab06" / "configs" / "base.yaml"


def deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def set_path(cfg, dotted, value):
    """cfg['a']['b'] = value for 'a.b'; the key must already exist (catches typos)."""
    keys, node = dotted.split("."), cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"unknown config section in override: {dotted}")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"unknown config key in override: {dotted}")
    node[keys[-1]] = value


def load_config(path=None, overrides=()):
    """Base config, then the YAML at `path` (if any), then 'a.b=value' overrides (values parsed as YAML)."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != BASE_CONFIG.resolve():
        with open(path) as f:
            extra = yaml.safe_load(f) or {}
        check_keys(cfg, extra)
        deep_update(cfg, extra)
    for item in overrides:
        key, _, raw = item.partition("=")
        set_path(cfg, key.strip(), yaml.safe_load(raw))
    validate(cfg)
    return cfg


def check_keys(ref, new, prefix=""):
    for k, v in new.items():
        if k not in ref:
            raise KeyError(f"unknown config key: {prefix}{k}")
        if isinstance(v, dict) and isinstance(ref[k], dict) and ref[k]:
            check_keys(ref[k], v, f"{prefix}{k}.")


def validate(cfg):
    assert cfg["optimizer"]["name"] in ("adam", "sgd"), "optimizer.name must be adam or sgd"
    assert cfg["scheduler"]["name"] in ("none", "step", "cosine", "plateau"), "bad scheduler.name"
    assert cfg["train"]["accum_steps"] >= 1 and cfg["train"]["epochs"] >= 1
    assert cfg["data"]["batch_size"] >= 1 and cfg["data"]["num_workers"] >= 0
    ie = cfg["run"]["interrupt_after_epoch"]
    assert ie is None or 1 <= ie < cfg["train"]["epochs"], "interrupt_after_epoch must be < epochs"


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def flatten(cfg, prefix=""):
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def config_hash(cfg):
    """Hash of everything that affects the training result (not paths / mlflow / interruption)."""
    c = copy.deepcopy(cfg)
    c["run"] = {}
    return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:10]
