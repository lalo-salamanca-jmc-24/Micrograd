# Micrograd - Autograd Engine

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Graphviz](https://img.shields.io/badge/Graphviz-000000?style=for-the-badge&logo=graphviz&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## Project Overview

**Micrograd** is an educational implementation inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd). It demonstrates:

- How automatic differentiation works at a fundamental level
- Building computational graphs dynamically
- Implementing the chain rule for backpropagation
- Constructing neural networks from first principles

---

## Project Structure

```
Micrograd/
├── engine.py          # Core Value class with autograd
├── nn.py              # Neural network components (Neuron, Layer, MLP)
├── micrograd.ipynb    # Interactive notebook with visualizations
└── Digraph.gv         # Graphviz output file
```

---

## Core Components

### 1. Value Class (`engine.py`)

The `Value` class wraps scalar values and tracks operations for automatic differentiation:

```python
from engine import Value

a = Value(2.0)
b = Value(3.0)
c = a * b + a**2  # Builds computation graph
c.backward()       # Computes gradients
print(a.grad)      # ∂c/∂a
```

**Supported Operations:**
| Operation | Symbol | Description |
|-----------|--------|-------------|
| Addition | `+` | a + b |
| Multiplication | `*` | a \* b |
| Power | `**` | a \*\* n |
| Negation | `-` | -a |
| Subtraction | `-` | a - b |
| Division | `/` | a / b |
| ReLU | `.relu()` | max(0, a) |

### 2. Neural Network (`nn.py`)

Built on top of the autograd engine:

| Class    | Description                                         |
| -------- | --------------------------------------------------- |
| `Neuron` | Single neuron with weights, bias, and optional ReLU |
| `Layer`  | Collection of neurons (fully connected layer)       |
| `MLP`    | Multi-Layer Perceptron (stacked layers)             |

```python
from nn import MLP

# 2-layer MLP: 3 inputs → 4 hidden → 1 output
model = MLP(3, [4, 1])
x = [Value(1.0), Value(2.0), Value(3.0)]
output = model(x)
```

---

## Visualization

The notebook includes computational graph visualization using Graphviz:

```
       a ──┐
           ├──[*]── d ──┐
       b ──┘            ├──[+]── e ──[/]── L
       f ──────────────┘         c ──┘
```

Each node displays:

- Variable label
- Current value (`data`)
- Computed gradient (`grad`)

---

## Getting Started

### Prerequisites

```bash
pip install graphviz
```

### Usage

```python
# Example: Simple expression and backprop
from engine import Value

x = Value(2.0)
y = Value(3.0)
z = x * y + x**2
z.backward()

print(f"z = {z.data}")      # z = 10.0
print(f"∂z/∂x = {x.grad}")  # ∂z/∂x = 7.0 (y + 2x)
print(f"∂z/∂y = {y.grad}")  # ∂z/∂y = 2.0 (x)
```

---

## Key Concepts Demonstrated

1. **Computational Graphs**: Operations build a DAG of Value nodes
2. **Reverse-Mode Autodiff**: Gradients flow backward from output to inputs
3. **Chain Rule**: `∂L/∂a = ∂L/∂c × ∂c/∂a` applied recursively
4. **Topological Sort**: Ensures correct gradient computation order

---

## Tech Stack

- **Python 3.8+**
- **Graphviz** (for computation graph visualization)
- **Jupyter Notebook** (for interactive exploration)

---

## References

- [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [Video: The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)

---

## License

This project is for educational purposes.

---

_Built as part of GDSC AIML learning track_
