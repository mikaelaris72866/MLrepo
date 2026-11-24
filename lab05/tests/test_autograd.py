import math
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from autograd import Value, graph_nodes, value_sum, zero_grad  # noqa: E402


def test_forward_builds_graph():
    a, b = Value(2.0), Value(3.0)
    prod = a * b
    c = prod + a
    assert c.data == 8.0
    assert c._op == "+" and c._prev == (prod, a)       # parents of the addition
    assert prod._op == "*" and prod._prev == (a, b)
    assert a._prev == () and a._op == ""               # leaves have no parents
    assert len(graph_nodes(c)) == 4                    # a, b, a*b, (a*b)+a


def test_simple_gradients_and_accumulation():
    x, y = Value(3.0), Value(4.0)
    z = x * y + x                              # x is used twice
    z.backward()
    assert x.grad == y.data + 1                # dz/dx = y + 1  (accumulated from two paths)
    assert y.grad == x.data


def test_same_node_used_twice():
    x = Value(5.0)
    (x * x).backward()
    assert x.grad == 10.0                      # d(x^2)/dx
    y = Value(5.0)
    (y + y).backward()
    assert y.grad == 2.0                       # not 1: contributions are accumulated, not overwritten


def test_diamond_graph():
    x = Value(2.0)
    a, b = x * 3.0, x ** 2
    out = a * b                                # 3x * x^2 = 3x^3  ->  9x^2
    out.backward()
    assert out.data == 24.0 and x.grad == 36.0


def test_output_gradient_is_set_to_one():
    x = Value(1.5)
    out = x * x
    out.grad = 123.0                           # stale value must be overwritten by the seed
    out.backward()
    assert out.grad == 1.0 and x.grad == 3.0


def test_second_backward_accumulates_on_leaves():
    x = Value(2.0)
    out = x * x
    out.backward()
    out.backward()
    assert x.grad == 8.0                       # 4 + 4, as in PyTorch without zero_grad
    zero_grad([x])
    assert x.grad == 0.0


@pytest.mark.parametrize("x,expected", [(2.0, 1.0), (-2.0, 0.0), (0.0, 0.0)])
def test_relu_derivative(x, expected):
    v = Value(x)
    v.relu().backward()
    assert v.grad == expected


def test_pow_and_errors():
    x = Value(3.0)
    (x ** 3).backward()
    assert x.grad == 27.0
    with pytest.raises(TypeError):
        Value(2.0) ** Value(2.0)
    with pytest.raises(ValueError):
        Value(-2.0) ** 0.5


def test_derived_ops_against_closed_forms():
    a, b = Value(6.0), Value(4.0)
    out = (a - b) / b + 1.0 / a - 2.0 * b      # exercises sub, div, rdiv, rmul
    out.backward()
    assert math.isclose(out.data, 0.5 + 1 / 6 - 8.0)
    assert math.isclose(a.grad, 1 / 4 - 1 / 36)
    assert math.isclose(b.grad, -a.data / 16 - 2.0)


@pytest.mark.parametrize("x", [-20.0, -2.0, 0.0, 0.7, 12.0])
def test_sigmoid_log_exp_match_torch(x):
    v = Value(x)
    (v.sigmoid().log() + v.exp() * 0.01).backward()
    t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    (torch.sigmoid(t).log() + t.exp() * 0.01).backward()
    assert abs(v.grad - t.grad.item()) < 1e-9 * max(1.0, abs(t.grad.item()))


def test_log_clamps_its_argument_at_log_eps():
    """Documented deviation from torch.log: log(x) is evaluated at max(x, 1e-12)."""
    v = Value(1e-14)
    out = v.log()
    out.backward()
    assert math.isclose(out.data, math.log(1e-12)) and math.isclose(v.grad, 1e12)


def test_sigmoid_is_stable_in_both_tails():
    assert Value(-800.0).sigmoid().data == 0.0 or Value(-800.0).sigmoid().data < 1e-300
    assert Value(800.0).sigmoid().data == 1.0


def test_deep_chain_does_not_hit_recursion_limit():
    x = Value(1.0)
    y = x
    for _ in range(20_000):
        y = y + x
    y.backward()
    assert y.data == 20_001.0 and x.grad == 20_001.0


def _random_dag(rng, n_leaves=4, n_ops=40):
    """Same random expression built with Values and with torch tensors (shared sub-expressions)."""
    leaves = [rng.uniform(-2, 2) for _ in range(n_leaves)]
    v = [Value(x) for x in leaves]
    t = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in leaves]
    vn, tn = list(v), list(t)
    for _ in range(n_ops):
        op = rng.choice(["add", "mul", "relu", "sq"])
        i, j = rng.randrange(len(vn)), rng.randrange(len(vn))
        if op == "add":
            vn.append(vn[i] + vn[j]); tn.append(tn[i] + tn[j])
        elif op == "mul":
            vn.append(vn[i] * vn[j] * 0.3); tn.append(tn[i] * tn[j] * 0.3)
        elif op == "relu":
            vn.append(vn[i].relu()); tn.append(tn[i].relu())
        else:
            vn.append((vn[i] * 0.5) ** 2); tn.append((tn[i] * 0.5) ** 2)
    out_v = value_sum(vn[-5:])
    out_t = sum(tn[-5:])
    return v, t, out_v, out_t


@pytest.mark.parametrize("seed", range(25))
def test_random_dags_match_torch(seed):
    rng = random.Random(seed)
    v, t, out_v, out_t = _random_dag(rng)
    out_v.backward()
    out_t.backward()
    assert abs(out_v.data - out_t.item()) < 1e-9 * max(1.0, abs(out_t.item()))
    for a, b in zip(v, t):
        expected = 0.0 if b.grad is None else b.grad.item()   # torch: None for leaves the output ignores
        assert abs(a.grad - expected) < 1e-9 * max(1.0, abs(expected))
