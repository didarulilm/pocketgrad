import random
from pocketgrad.engine import Scalar


class Module:
    """Base class for all components."""

    def zero_grad(self):
        # Clear out gradients of all parameters
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        # Return an empty list, will be overridden by subclasses
        return []


class Neuron(Module):
    """A single neuron in the network."""

    def __init__(self, nin, non_linear=True):
        self.weights = [Scalar(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Scalar(random.uniform(-1, 1))
        self.non_linear = non_linear

    def __call__(self, x):
        # Compute the dot product of the weights and inputs and add the bias
        act = sum((wi * xi for wi, xi in zip(self.weights, x, strict=True)), self.bias)

        # Optionally pass through a non-linear activation function
        return act.relu() if self.non_linear else act

    def parameters(self):
        # Return a list of all parameters
        return self.weights + [self.bias]

    def __repr__(self):
        return f"{'ReLU' if self.non_linear else 'Linear'}Neuron({len(self.weights)})"


class Layer(Module):
    """A layer of neurons in the network."""

    def __init__(self, nin, nouts, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nouts)]

    def __call__(self, x):
        # Apply input x to each neuron in the layer
        out = [n(x) for n in self.neurons]

        # Return single output if one neuron else a list of scalars
        return out[0] if len(out) == 1 else out

    def parameters(self):
        # Return a list of all parameters from all neurons in the layer
        return [p for neuron in self.neurons for p in neuron.parameters()]
        # params = []
        # for neuron in self.neurons:
        #     for p in neuron.parameters():
        #         params.append(p)

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """
    Multi-layer perceptron composed of fully connected layers.
    
    Example:
    
    model = MLP(2, [16, 16, 1])

    - 2 input features
    - 2 hidden layers of 16 neurons each
    - 1 output neuron
    """

    def __init__(self, nin, nouts):
        # Combine the number of inputs and the number of outputs in a single list,
        # and create all layers with a non-linear activation except for the last layer.
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], non_linear=(i != len(nouts) - 1)) for i in range(len(nouts))]

    def __call__(self, x):
        # Forward propagation
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # Return a list of all parameters from all layers in the MLP
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"