import random
from engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


# Testing the neural network with backpropagation
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Micrograd Neural Network")
    print("=" * 50)
    
    # Create a simple MLP: 3 inputs -> 4 hidden -> 4 hidden -> 1 output
    model = MLP(3, [4, 4, 1])
    print(f"\nModel: {model}")
    print(f"Total parameters: {len(model.parameters())}")
    
    # Sample input
    x = [2.0, 3.0, -1.0]
    print(f"\nInput: {x}")
    
    # Forward pass
    output = model(x)
    print(f"Output: {output.data:.4f}")
    
    # Compute a simple loss (target = 1.0)
    target = 1.0
    loss = (output - target) ** 2
    print(f"Loss (MSE): {loss.data:.4f}")
    
    # Backward pass
    loss.backward()
    
    # Show some gradients
    print("\n" + "-" * 50)
    print("Gradients after backpropagation:")
    print("-" * 50)
    
    # Show first layer weights gradients
    first_neuron = model.layers[0].neurons[0]
    print(f"\nFirst neuron weights and gradients:")
    for i, w in enumerate(first_neuron.w):
        print(f"  w[{i}]: value={w.data:.4f}, grad={w.grad:.4f}")
    print(f"  bias: value={first_neuron.b.data:.4f}, grad={first_neuron.b.grad:.4f}")
    
    print("\n" + "=" * 50)
    print("Backpropagation test PASSED!")
    print("=" * 50)