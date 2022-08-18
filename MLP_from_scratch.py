import random
import numpy as np


class Scalar:
    def __init__(self, data, _children=()) -> None:
        self.data = data
        self.grad = 0
        self.backward = lambda: None
        self.prev = set(_children)
    
    def __repr__(self) -> str:
        return f"Scalar(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other))

        def backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out.backward = backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other))

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out.backward = backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Scalar(float(np.power(self.data, other.data)), (self,))

        def backward():
            self.grad += (other * float(np.power(self.data, other.data - 1))) * out.grad
        out.backward = backward

        return out
    
    def relu(self):
        out = Scalar(max(0, self.data), (self,))

        def backward():
            self.grad += (out.data > 0) * out.grad
        out.backward = backward

        return out
    
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self - other
    
    def __rsub__(self, other):
        return other - self
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * 1 / other
    
    def __rtruediv__(self, other):
        return other * 1 / self


class SubModule:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []


class Neuron(SubModule):
    def __init__(self, in_features, nonlinearity=True) -> None:
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(in_features)]
        self.b = Scalar(0)
        self.nonlinearity = nonlinearity
    
    def __call__(self, x):
        activation = sum((ww * xx for ww, xx in zip(self.w, x)), self.b)
        return activation.relu() if self.nonlinearity else activation
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self) -> str:
        return f"{'ReLU' if self.nonlinearity else 'Linear'}Neuron({len(self.w)})"


class Layer(SubModule):
    def __init__(self, in_features, out_features, **kwargs) -> None:
        self.neurons = [Neuron(in_features, **kwargs) for _ in range(out_features)]
    
    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        if len(out) == 1:
            return out[0]
        return out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(SubModule):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlinearity=i!=len(nouts)-1) for i in range(len(nouts))]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
