# pocketgrad

<a href="https://github.com/didarulilm/pocketgrad/actions/workflows/ci.yml"><img src="https://github.com/didarulilm/pocketgrad/actions/workflows/ci.yml/badge.svg" alt="CI"></a>

A minimal, pedagogical implementation of an autograd engine and neural network library written in Python. Built to understand from first principles how frameworks like PyTorch implement reverse-mode automatic differentiation (backpropagation) under the hood. 

Thanks to Andrej Karpathy for [micrograd](https://github.com/karpathy/micrograd), which served as the primary reference for this project.

## Installation

```bash
pip install pocketgrad
```

## Example with Computational Graph
Each operation dynamically adds a node to the computation graph, forming a [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph). Calling `.backward()` traverses this graph in reverse topological order, accumulating gradients at each node via the [chain rule](https://en.wikipedia.org/wiki/Chain_rule).

The example below demonstrates this by building a simple graph, running backpropagation, and visualizing the result:

```python
from pocketgrad.engine import Scalar
from pocketgrad.visualize import draw_graph

a = Scalar(3.0, label="a")
b = Scalar(5.5, label="b")

c = a + b;    c.label = "c"
d = c / 2;    d.label = "d"
e = d.relu(); e.label = "e"

e.backward()
draw_graph(e)
```

<p align="left">
  <img src="docs/graph.svg" alt="Computational Graph" width="300">
</p> 

## Training a Neural Network

The notebook `demo_mlp.ipynb` provides an end-to-end example of training a simple 2-layer feed-forward MLP with the `pocketgrad.nn` module on the classic two-moons dataset, achieving 100% accuracy. The plot below visualizes the decision boundary learned by the model:

 ![Decision Boundary](docs/decision_boundary.png)

## Architecture

```text
pocketgrad/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI workflow
├── docs/
│   ├── decision_boundary.png   
│   └── graph.svg              
├── pocketgrad/
│   ├── __init__.py             # Package exports
│   ├── engine.py               # Core autograd engine
│   ├── nn.py                   # Neural network modules
│   └── visualize.py            # Graph rendering utilities
├── test/
│   └── test_engine.py          # Unit tests
├── demo_graph.ipynb            # Graph demo notebook
├── demo_mlp.ipynb              # MLP demo notebook
└── pyproject.toml              # Build configuration
```

## Design Decisions

- **Scalar-valued by design:** Each node in the computation graph is represented as a `Scalar` rather than a tensor. This makes the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) traceable at every step, with each gradient reduced to a single number you can verify by hand.
- **Batteries included for learning:** Features a small neural-network library `nn.py` for building and training models on top of the core engine, and `visualize.py` for rendering computation graphs and tracing gradients step by step.
- **Readability over performance:** `pocketgrad` avoids abstraction that reduces its educational value without offering the benefits of a production-grade framework.

## Out of Scope
As a pedagogical tool, `pocketgrad` does not cover:

- Vectorization or tensor abstractions
- GPU acceleration, CUDA, or low-level kernel optimizations
- PyTorch-level API breadth


## Tests

If you are using [uv](https://docs.astral.sh/uv/), you can sync the dependencies and run the test suite with:

```bash
uv sync
uv run -m pytest
```

Or from an active Python environment:

```bash
python -m pytest
```

## License

MIT 
