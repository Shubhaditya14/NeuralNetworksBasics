import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self,w,b):
        self.w = w
        self.b = b

    def feedforward(self,inputs):
        t = np.dot(self.w,inputs) + self.b;
        return sigmoid(t)
    

weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = Neuron(weights, bias)

x = np.array([2, 3])       # x1 = 2, x2 = 3
print(n.feedforward(x))    # 0.9990889488055994

class NeuralNetwork:
    def __init__(self):
        w = np.array([0, 1])
        b = 0
        self.h1 = Neuron(w,b)
        self.h2 = Neuron(w,b)
        self.o1 = Neuron(w,b)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1
    
network = NeuralNetwork()
x = np.array([2,3])
print(network.feedforward(x))

def MSE(y_true,y_pred):
    return ((y_true - y_pred) ** 2).mean()
 
