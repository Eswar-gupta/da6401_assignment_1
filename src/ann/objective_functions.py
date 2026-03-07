import numpy as np

def _softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Loss:
    def __init__(self, name):
        self.name = name.lower().replace('_', '')

    def compute(self, y_true, y_pred):
        if self.name == "crossentropy":
            logits = y_pred
            logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
            log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=1, keepdims=True))
            log_probs = logits_shifted - log_sum_exp
            return -np.mean(np.sum(y_true * log_probs, axis=1))
        elif self.name == "meansquarederror":
            probs = _softmax(y_pred)                         # ← apply softmax first
            return np.mean((probs - y_true) ** 2)            # ← MSE on probabilities

    def output_delta(self, y_true, y_pred):
        N = y_true.shape[0]
        if self.name == "crossentropy":
            logits = y_pred
            logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits_shifted)
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return (softmax_probs - y_true) / N
        elif self.name == "meansquarederror":
            K = y_true.shape[1]
            probs = _softmax(y_pred)                                          # ← softmax of logits
            grad_prob = 2.0 * (probs - y_true) / (N * K)                     # ← upstream gradient
            dot = np.sum(grad_prob * probs, axis=1, keepdims=True)            # ← softmax Jacobian
            return probs * (grad_prob - dot)                                  # ← chain rule through softmax