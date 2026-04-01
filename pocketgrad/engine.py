import math


class Scalar:
    """Represents a scalar node and its gradient in the computation graph."""

    def __init__(self, data, label="", _ancestors=(), _op=""):
        assert isinstance(data, (int, float)), "node data must be an int or float"
        self.data = data               # scalar value stored in this node
        self.label = str(label)        # optional: used for debugging/visualization
        self._prev = set(_ancestors)   # ancestor nodes in the computation graph
        self._op = _op                 # operation that created this node
        self.grad = 0.0                # stored gradient value
        self._backward = lambda: None  # for leaf nodes
        
    # Called by: self + other
    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(data=other, label=other)
        output = Scalar(data=self.data + other.data, _ancestors=(self, other), _op="+")

        def _backward():
            self.grad += output.grad * 1.0
            other.grad += output.grad * 1.0

        output._backward = _backward
        return output

    # Called by: self * other
    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(data=other, label=other)
        output = Scalar(data=self.data * other.data, _ancestors=(self, other), _op="*")

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward
        return output

    # Called by: self ** other
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "power must be an int or float"

        output = Scalar(self.data**other, _ancestors=(self,), _op=f"^{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * output.grad

        output._backward = _backward
        return output

    # Called by: self.relu()
    def relu(self):
        output = Scalar(0 if self.data < 0 else self.data, _ancestors=(self,), _op="ReLU")

        def _backward():
            self.grad += (output.data > 0) * output.grad

        output._backward = _backward
        return output

    # Called by: self.sigmoid()
    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        output = Scalar(s, _ancestors=(self,), _op="sigmoid")

        def _backward():
            self.grad += (s * (1.0 - s)) * output.grad

        output._backward = _backward
        return output

    # Called by: self.tanh()
    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        output = Scalar(t, _ancestors=(self,), _op="tanh")

        def _backward():
            self.grad += (1 - t**2) * output.grad

        output._backward = _backward
        return output

    def backward(self):
        """
        Traverse the directed acyclic graph (DAG) in reverse topological order
        to compute gradients via reverse-mode automatic differentiation.
        """
        topo = []
        visited = set()

        # Recursively build the topological order 
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for prev in node._prev:
                    build_topo(prev)
                topo.append(node)

        # Start from the root node
        build_topo(self)

        # Manually inserting ∂output/∂output = 1
        self.grad = 1.0

        # Traverse in reverse order and do backpropagation
        for node in reversed(topo):
            node._backward()

    # Called by: -self
    def __neg__(self):
        return self * -1

    # Called by: other + self
    def __radd__(self, other):
        return self + other

    # Called by: self - other
    def __sub__(self, other):
        return self + (-other)

    # Called by: other - self
    def __rsub__(self, other):
        return other + (-self)

    # Called by: other * self
    def __rmul__(self, other):
        return self * other

    # Called by: self / other
    def __truediv__(self, other):
        return self * other**-1

    # Called by: other / self
    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self):
        return f"Scalar(data={self.data}, label={self.label})"