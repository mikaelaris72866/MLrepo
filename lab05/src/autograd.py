"""A scalar reverse-mode automatic-differentiation engine.

Every arithmetic operation on a `Value` creates a NEW `Value` that remembers
    * its parents  (`_prev`: the Values the operation was applied to),
    * the operation name (`_op`, for printing / debugging),
    * its local derivative rule (`_backward`): a closure that, given the gradient of
      this node, adds the corresponding contribution to the gradient of each parent.
Running the forward pass therefore builds the computation graph (a DAG) as a side effect.

`backward()` topologically sorts the graph reachable from the output, sets the output
gradient to 1, visits the nodes in reverse topological order and **accumulates**
(`+=`) into the parents' gradients.  Accumulation is what makes a node that is used more than once
(`x * x`, `x + x`, shared sub-expressions) receive the sum of all its contributions.

Required operations: addition, multiplication, power (constant exponent) and ReLU.
Convenience operations built on them: negation, subtraction, division.
Extra primitive operations needed for the logistic model of the lab: exp, log, sigmoid.

Note: `backward()` does not clear existing gradients.  Calling it twice adds the second pass
on top of the first, exactly like PyTorch with retained graphs; use `zero_grad` between
training steps.
"""
import math

LOG_EPS = 1e-12  # log(x) is evaluated at max(x, LOG_EPS) so that log(0) does not blow up


def _noop():
    pass


class Value:
    __slots__ = ("data", "grad", "_prev", "_backward", "_op")

    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)      # forward value
        self.grad = 0.0              # d(output)/d(this node), filled in by backward()
        self._prev = tuple(_prev)    # parent nodes (inputs of the op that produced this node)
        self._backward = _noop       # local derivative rule (set by the op that creates the node)
        self._op = _op               # operation label

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # ------------------------------------------------------------ required operations
    def __add__(self, other):
        other = _as_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad          # d(a+b)/da = 1
            other.grad += out.grad         # d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = _as_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad    # d(ab)/da = b
            other.grad += self.data * out.grad    # d(ab)/db = a
        out._backward = _backward
        return out

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("only constant int/float exponents are supported")
        if self.data < 0 and exponent != int(exponent):
            raise ValueError("negative base with a fractional exponent is not real-valued")
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            if exponent != 0:
                self.grad += exponent * self.data ** (exponent - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            if out.data > 0:               # derivative is 1 for x > 0 and 0 otherwise (also at 0)
                self.grad += out.grad
        out._backward = _backward
        return out

    # ------------------------------------------------------------ extra primitives (logistic model)
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad       # d exp(x)/dx = exp(x)
        out._backward = _backward
        return out

    def log(self):
        x = max(self.data, LOG_EPS)
        out = Value(math.log(x), (self,), "log")

        def _backward():
            self.grad += out.grad / x              # d log(x)/dx = 1/x
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        if x >= 0:                                  # numerically stable in both tails
            s = 1.0 / (1.0 + math.exp(-x))
        else:
            e = math.exp(x)
            s = e / (1.0 + e)
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad  # d sigma/dx = sigma (1 - sigma)
        out._backward = _backward
        return out

    # ------------------------------------------------------------ derived operations
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_as_value(other))

    def __truediv__(self, other):
        return self * _as_value(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return _as_value(other) + (-self)

    def __rtruediv__(self, other):
        return _as_value(other) * self ** -1

    # ------------------------------------------------------------ backward pass
    def topological_order(self):
        """Nodes reachable from `self`, parents before children (iterative DFS, post-order)."""
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))        # appended to `order` after all its parents
            for parent in node._prev:
                if id(parent) not in visited:
                    stack.append((parent, False))
        return order

    def backward(self):
        order = self.topological_order()
        self.grad = 1.0                        # d(output)/d(output) = 1
        for node in reversed(order):           # children before parents
            node._backward()                   # accumulates into the parents' grads


def _as_value(x):
    return x if isinstance(x, Value) else Value(x)


def value_sum(values):
    """Sum a list of Values as a chain of additions (a graph node per addition)."""
    it = iter(values)
    total = next(it)
    for v in it:
        total = total + v
    return total


def zero_grad(values):
    for v in values:
        v.grad = 0.0


def graph_nodes(root):
    """All nodes of the graph rooted at `root` (for counting / inspection)."""
    return root.topological_order()
