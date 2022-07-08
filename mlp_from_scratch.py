import numpy as np
import matplotlib.pyplot as plt


class MSE:
    def __call__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

        return np.square((y_pred - y_true)).mean()
    
    def backward(self):
        n = self.y_true.shape[0]
        self.gradient = 2. * (self.y_pred - self.y_true) / n
        return self.gradient


class Linear:
    def __init__(self, input_dim, num_hidden=1):
        self.weights = np.random.randn(input_dim, num_hidden) * np.sqrt(2. / input_dim)
        self.bias = np.zeros(num_hidden)
    
    def __call__(self, x):
        self.x = x
        output = self.x @ self.weights + self.bias
        return output
    
    def backward(self, gradient):
        self.w_grad = self.x.T @ gradient
        self.b_grad = gradient.sum(axis=0)
        self.x_grad = gradient @ self.weights.T

        return self.x_grad
    
    def update(self, lr):
        self.weights = self.weights - lr * self.w_grad
        self.bias = self.bias - lr * self.b_grad


class ReLU:
    def __call__(self, inp):
        self.input = inp
        self.output = np.clip(self.input, 0, None)
        return self.output
    
    def backward(self, output_grad):
        self.input_grad = (self.input > 0) * output_grad
        return self.input_grad



'''
class MLP:
    def __init__(self, input_dim, num_hidden, num_layers=2):
        self.model = []
        
        self.model.append(Linear(input_dim, num_hidden))
        self.model.append(ReLU())

        if num_layers > 2:
            for _ in range(num_layers - 2):
                self.model.append(Linear(num_hidden, num_hidden))
                self.model.append(ReLU())
        
        self.model.append(Linear(num_hidden, 1))
    
    def __call__(self, x):
        t = x

        for block in self.model:
            t = block(t)
        return t
    
    def backward(self, output_grad):
        grad = output_grad

        for block in self.model[::-1]:
            grad = block.backward(grad)
        return grad
    
    def update(self, lr):
        for block in self.model[::-1]:
            block.update(lr)
'''

class Model:
  def __init__(self, input_dim, num_hidden):
    self.linear1 = Linear(input_dim, num_hidden)
    self.relu = ReLU()
    self.linear2 = Linear(num_hidden, 1)
  
  def __call__(self, x):
    return self.linear2(self.relu(self.linear1(x)))
  
  def backward(self, output_grad):
    linear2_grad = self.linear2.backward(output_grad)
    relu_grad = self.relu.backward(linear2_grad)
    linear1_grad = self.linear1.backward(relu_grad)
    return linear1_grad
  
  def update(self, lr):
    self.linear2.update(lr)
    self.linear1.update(lr)


n = 200
d = 1

x = np.random.uniform(-1, 1, (n, d))
weights_true = np.array([[5],])
bias_true = np.array([10])

y_true = np.power(x, 3) @ weights_true + np.square(x) @ weights_true + x @ weights_true+ bias_true + np.random.randn(n, d)
plt.scatter(x, y_true, marker='x')

loss = MSE()
model = Model(d, 10)
lr = 0.1
epochs = 50

for epoch in range(1, epochs + 1):
    y_pred = model(x)
    loss_value = loss(y_pred, y_true)

    if not epoch % 5:
        print(epoch, loss_value)
        plt.scatter(x, y_pred.squeeze())
    
    model.backward(loss.backward())
    model.update(lr)

plt.show()
