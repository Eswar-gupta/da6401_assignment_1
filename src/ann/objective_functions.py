import numpy as np
class Loss:
    def __init__(self, name):
        self.name = name

    def compute(self, y_true, y_pred):
        if self.name == "crossentropy":
            # y_pred is logits, compute softmax for loss
            logits = y_pred
            logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
            log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=1, keepdims=True))
            log_probs = logits_shifted - log_sum_exp
            return -np.mean(np.sum(y_true * log_probs, axis=1))
        elif self.name == "meansquarederror":
            return np.mean(np.sum((y_pred - y_true) ** 2, axis=1)) / 2

    def output_delta(self, y_true, y_pred):
        N = y_true.shape[0]
        if self.name == "crossentropy":
            # y_pred is logits, compute softmax then subtract y_true
            logits = y_pred
            logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits_shifted)
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return (softmax_probs - y_true) / N
        elif self.name == "meansquarederror":
            err = y_pred - y_true
            return y_pred * (err - np.sum(y_pred * err, axis=1, keepdims=True)) / N