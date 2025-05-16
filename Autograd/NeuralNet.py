from Autograd import Value
import random

class Neuron:

    def __init__(self, numInputs):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(numInputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        assert len(x) == len(self.weights), "Dimension mismatch for inputs"
    
        summedValue = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        activation = summedValue.tanh()
        
        return activation
    
    def parameters(self):
        return self.weights + [self.bias]

class Layer:

    def __init__(self, numInputs, numOutputs):
        self.neurons = [Neuron(numInputs) for _ in range(numOutputs)]

    def __call__(self, x):
        output = [neuron(x) for neuron in self.neurons]
        return output[0] if len(output) == 1 else output
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MultiLayerPerceptron:
    def __init__(self, numInputs, numOutputs):
        size = [numInputs] + numOutputs
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(numOutputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        

def train(mlp, inputs, target, epochs, lr):
    
    for epoch in range(epochs):
        predictions = [mlp(x) for x in inputs]
        loss = sum((targetVal - predictionsVal) ** 2 for targetVal, predictionsVal in zip(target, predictions))

        for param in mlp.parameters():
            param.grad = 0.0

        loss.backward()
    
        for param in mlp.parameters():
            param.data += -1 * lr * param.grad

        print(f"Epoch: {epoch}, Cumulative Loss: {loss.data}")

inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
]
target = [-0.5, 0.73]

nn = MultiLayerPerceptron(3, [4, 4, 1])

train(nn, inputs, target, 20, 0.05)
print("\n=== Predictions after training ===")

for x, t in zip(inputs, target):
    pred = nn(x)
    print(f"Input: {x} → Predicted: {pred.data:.4f}, Target: {t}")
