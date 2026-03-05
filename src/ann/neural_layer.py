import numpy as np
from ann.activations import Activation

class NeuralLayer:
    def __init__(self, in_size, out_size, activation_name, weight_init):
        if weight_init == "xavier":
            self.W = np.random.randn(in_size, out_size) * np.sqrt(2.0 / (in_size + out_size))
        else:
            self.W = np.random.randn(in_size, out_size) * 0.01

        self.b = np.zeros((1, out_size))
        self.activation = Activation(activation_name)
        self.gradW = None
        self.gradb = None
        self.input = None
        self.pre_act = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.pre_act = X @ self.W + self.b
        self.output = self.activation.forward(self.pre_act)
        return self.output