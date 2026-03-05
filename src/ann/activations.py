import numpy as np
class Activation:
    def __init__(self, name):
        self.name = name

    def forward(self, x):
        if self.name == "relu":
            return np.maximum(0, x)
        elif self.name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif self.name == "tanh":
            return np.tanh(x)
        elif self.name == "softmax":
            e = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e / np.sum(e, axis=1, keepdims=True)

    def derivative(self, x):
        if self.name == "relu":
            return (x > 0).astype(float)
        elif self.name == "sigmoid":
            s = self.forward(x)
            return s * (1 - s)
        elif self.name == "tanh":
            return 1 - np.tanh(x) ** 2