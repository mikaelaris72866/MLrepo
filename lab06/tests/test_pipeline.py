import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import REPO_ROOT, config_hash, load_config  # noqa: E402
from data import get_datasets  # noqa: E402
from models import CNN, count_params  # noqa: E402
from train import TrainingInterrupted, Timers, run, train_one_epoch  # noqa: E402

DATA = REPO_ROOT / "data" / "FashionMNIST" / "raw"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="Fashion-MNIST not downloaded")


def tiny_cfg(tmp_path, **extra):
    over = ["train.epochs=4", "train.threads=1", "data.subset_train=640", "data.subset_test=200", "model.c1=4",
            "model.c2=8", f"run.out_dir={tmp_path}/out", f"run.mlflow.tracking_uri={tmp_path}/mlruns"]
    over += [f"{k}={v}" for k, v in extra.items()]
    return load_config(None, over)


def test_overrides_and_typo_detection():
    cfg = load_config(None, ["optimizer.lr=0.01", "data.batch_size=64", "run.resume=null"])
    assert cfg["optimizer"]["lr"] == 0.01 and cfg["data"]["batch_size"] == 64 and cfg["run"]["resume"] is None
    with pytest.raises(KeyError):
        load_config(None, ["optimizer.learning_rate=0.1"])
    with pytest.raises(KeyError):
        load_config(None, ["nosuch.x=1"])


def test_yaml_file_merges_over_base(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"optimizer": {"name": "sgd", "lr": 0.05}}))
    cfg = load_config(p)
    assert cfg["optimizer"]["name"] == "sgd" and cfg["optimizer"]["momentum"] == 0.9 and cfg["seed"] == 42
    p.write_text(yaml.safe_dump({"optimiser": {}}))
    with pytest.raises(KeyError):
        load_config(p)


def test_hash_ignores_run_section_but_not_training_settings():
    a, b = load_config(None), load_config(None, ["run.name=other", "run.out_dir=x"])
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(load_config(None, ["optimizer.lr=0.5"]))


def test_model_size_matches_lab3_best():
    assert count_params(CNN(64, 128)) == 137_226     # cnn_matched in Lab 3


def test_gradient_accumulation_equals_big_batch():
    """4 micro-batches of 16 with accumulation must give the same step as one batch of 64 (no BN/dropout),
    including a ragged last group."""
    torch.manual_seed(0)
    x, y = torch.randn(200, 1, 28, 28), torch.randint(0, 10, (200,))
    ds = torch.utils.data.TensorDataset(x, y)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def one_epoch(bs, accum):
        torch.manual_seed(1)
        m = CNN(4, 8)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
        train_one_epoch(m, dl, opt, loss_fn, len(ds), accum)
        return torch.cat([p.detach().flatten() for p in m.parameters()])

    big, acc = one_epoch(64, 1), one_epoch(16, 4)
    assert torch.allclose(big, acc, atol=1e-6), (big - acc).abs().max()
    assert not torch.allclose(big, one_epoch(16, 1), atol=1e-6)      # without accumulation it differs


@needs_data
def test_resume_reproduces_uninterrupted_run(tmp_path):
    full = run(tiny_cfg(tmp_path / "a"))
    cfg_i = tiny_cfg(tmp_path / "b", **{"run.interrupt_after_epoch": 2})
    with pytest.raises(TrainingInterrupted):
        run(cfg_i)
    cfg_r = tiny_cfg(tmp_path / "b", **{"run.resume": f"{tmp_path}/b/out/last.pt"})
    res = run(cfg_r)
    assert [h["epoch"] for h in res["history"]] == [1, 2, 3, 4]
    for a, b in zip(full["history"], res["history"]):
        assert a["train_loss"] == pytest.approx(b["train_loss"], abs=1e-6)
        assert a["val_acc"] == b["val_acc"]
    assert full["test_acc"] == res["test_acc"]


@needs_data
def test_resume_refuses_changed_config(tmp_path):
    with pytest.raises(TrainingInterrupted):
        run(tiny_cfg(tmp_path, **{"run.interrupt_after_epoch": 2}))
    bad = tiny_cfg(tmp_path, **{"run.resume": f"{tmp_path}/out/last.pt", "optimizer.lr": 0.5})
    with pytest.raises(ValueError):
        run(bad)


@needs_data
def test_mlflow_logs_params_metrics_artifacts(tmp_path):
    import mlflow
    cfg = tiny_cfg(tmp_path, **{"train.epochs": 2})
    s = run(cfg)
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    r = mlflow.get_run(s["mlflow_run_id"])
    assert r.data.params["optimizer.lr"] == "0.001" and r.data.params["seed"] == "42"
    assert "val_acc" in r.data.metrics and "test_acc" in r.data.metrics
    client = mlflow.tracking.MlflowClient()
    assert {a.path for a in client.list_artifacts(s["mlflow_run_id"])} >= {"config.yaml", "history.csv", "last.pt"}
    assert len(client.get_metric_history(s["mlflow_run_id"], "val_acc")) == 2
    assert r.info.status == "FINISHED"


@needs_data
def test_split_is_55k_5k_and_disjoint():
    cfg = load_config(None)
    tr, va, te = get_datasets(cfg)
    assert (len(tr), len(va), len(te)) == (55_000, 5_000, 10_000)
    assert not set(tr.indices) & set(va.indices)
    x, _ = tr[0]
    assert x.shape == (1, 28, 28) and abs(float(x.mean())) < 5
