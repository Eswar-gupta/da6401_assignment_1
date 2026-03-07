"""
Loss Functions
"""

import numpy as np


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


class Cross_Entropy:

    def __init__(self):
        self.probabilities = None
        self.y_true = None

    def forward(self, logits, y_true):
        # Compute CE from logits via log-softmax for stable and exact gradients.
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - logsumexp
        self.probabilities = np.exp(log_probs)
        self.y_true = y_true
        loss = -np.sum(y_true * log_probs) / logits.shape[0]
        return loss

    def backward(self):
        return (self.probabilities - self.y_true) / self.y_true.shape[0]



class MSE:

    def __init__(self):
        self.probabilities = None
        self.y_true = None

    def forward(self, logits, y_true):
        self.probabilities = softmax(logits)
        self.y_true = y_true
        loss = np.mean((self.probabilities - y_true) ** 2)
        return loss

    def backward(self):
        batch = self.y_true.shape[0]
        num_classes = self.y_true.shape[1]
        grad_prob = 2.0 * (self.probabilities - self.y_true) / (batch * num_classes)
        dot = np.sum(grad_prob * self.probabilities, axis=1, keepdims=True)
        return self.probabilities * (grad_prob - dot)