import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import Loss
from ann.optimizers import get_optimizer


class NeuralNetwork:

    def __init__(self, cli_args):
        self.args = cli_args
        self.loss_fn = Loss(cli_args.loss)
        self.layers = []

        hidden = cli_args.hidden_size
        if isinstance(hidden, list):
            sizes = [784] + hidden + [10]
        else:
            sizes = [784] + [hidden] * cli_args.num_layers + [10]

        for i in range(len(sizes) - 1):
            act = cli_args.activation if i < len(sizes) - 2 else "softmax"
            self.layers.append(NeuralLayer(sizes[i], sizes[i + 1], act, cli_args.weight_init))

        self.optimizer = get_optimizer(cli_args, self.layers)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true, y_pred):
        delta = self.loss_fn.output_delta(y_true, y_pred)

        grad_W_list = []
        grad_b_list = []

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            layer.gradW = layer.input.T @ delta
            layer.gradb = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                prev = self.layers[i - 1]
                delta = (delta @ layer.W.T) * prev.activation.derivative(prev.pre_act)

            grad_W_list.append(layer.gradW)
            grad_b_list.append(layer.gradb)

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step()

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        n = X_train.shape[0]
        for epoch in range(epochs):
            idx = np.arange(n)
            np.random.shuffle(idx)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = X_train[idx[start:end]]
                yb = y_train[idx[start:end]]

                y_pred = self.forward(xb)
                self.backward(yb, y_pred)
                self.update_weights()

            train_pred = self.forward(X_train)
            loss = self.loss_fn.compute(y_train, train_pred)
            acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
            print(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f} - acc: {acc:.4f}")

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        loss = self.loss_fn.compute(y, y_pred)
        acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return y_pred, loss, acc

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

