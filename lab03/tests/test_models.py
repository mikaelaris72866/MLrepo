import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import build_models  # noqa: E402

# Parameter counts worked out by hand (see report.md, section 3).
HAND_PARAMS = {
    "mlp": 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10,                      # 235,146
    "cnn": (1 * 16 * 9 + 16) + (16 * 32 * 9 + 32) + (32 * 49 * 10 + 10),             # 20,490
    "attention": (49 * 64 + 64) + 2 * (12480 + 4160 + 8320 + 8256 + 256) + (64 * 10 + 10),  # 70,794
}


@pytest.mark.parametrize("name", list(HAND_PARAMS))
def test_hand_param_counts(name):
    model, _ = build_models()[name]
    assert sum(p.numel() for p in model.parameters()) == HAND_PARAMS[name]


def test_all_models_output_shape():
    x = torch.randn(4, 1, 28, 28)
    for name, (m, _) in build_models().items():
        assert m(x).shape == (4, 10), name


def test_matched_models_in_target_range():
    for name, (m, _) in build_models().items():
        if name.endswith("_matched"):
            n = sum(p.numel() for p in m.parameters())
            assert 100_000 <= n <= 150_000, (name, n)
