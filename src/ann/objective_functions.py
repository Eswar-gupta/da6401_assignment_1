import numpy as np
class Loss:
    def __init__(self, name):
        self.name = name

    def compute(self, y_true, y_pred):
        if self.name == "crossentropy":
            p = np.clip(y_pred, 1e-12, 1.0)
            return -np.mean(np.sum(y_true * np.log(p), axis=1))
        elif self.name == "meansquarederror":
            return np.mean(np.sum((y_pred - y_true) ** 2, axis=1)) / 2

    def output_delta(self, y_true, y_pred):
        N = y_true.shape[0]
        if self.name == "crossentropy":
            return (y_pred - y_true) / N
        elif self.name == "meansquarederror":
            err = y_pred - y_true
            return y_pred * (err - np.sum(y_pred * err, axis=1, keepdims=True)) / N