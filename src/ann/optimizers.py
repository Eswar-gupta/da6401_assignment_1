import numpy as np
class SGD:
    def __init__(self, layers, lr, wd):
        self.layers = layers
        self.lr = lr
        self.wd = wd

    def step(self):
        for layer in self.layers:
            gW = layer.gradW + self.wd * layer.W
            layer.W -= self.lr * gW
            layer.b -= self.lr * layer.gradb

class Momentum:
    def __init__(self, layers, lr, wd, beta=0.9):
        self.layers = layers
        self.lr = lr
        self.wd = wd
        self.beta = beta
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            gW = layer.gradW + self.wd * layer.W
            self.vW[i] = self.beta * self.vW[i] + gW
            self.vb[i] = self.beta * self.vb[i] + layer.gradb
            layer.W -= self.lr * self.vW[i]
            layer.b -= self.lr * self.vb[i]

class NAG:
    def __init__(self, layers, lr, wd, beta=0.9):
        self.layers = layers
        self.lr = lr
        self.wd = wd
        self.beta = beta
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            gW = layer.gradW + self.wd * layer.W
            self.vW[i] = self.beta * self.vW[i] + gW
            self.vb[i] = self.beta * self.vb[i] + layer.gradb
            layer.W -= self.lr * (self.beta * self.vW[i] + gW)
            layer.b -= self.lr * (self.beta * self.vb[i] + layer.gradb)

class RMSProp:
    def __init__(self, layers, lr, wd, beta=0.9, eps=1e-8):
        self.layers = layers
        self.lr = lr
        self.wd = wd
        self.beta = beta
        self.eps = eps
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            gW = layer.gradW + self.wd * layer.W
            self.vW[i] = self.beta * self.vW[i] + (1 - self.beta) * gW ** 2
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * layer.gradb ** 2
            layer.W -= self.lr * gW / (np.sqrt(self.vW[i]) + self.eps)
            layer.b -= self.lr * layer.gradb / (np.sqrt(self.vb[i]) + self.eps)

class Adam:
    def __init__(self, layers, lr, wd, beta1=0.9, beta2=0.999, eps=1e-8):
        self.layers = layers
        self.lr = lr
        self.wd = wd
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.mW = [np.zeros_like(l.W) for l in layers]
        self.mb = [np.zeros_like(l.b) for l in layers]
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            gW = layer.gradW + self.wd * layer.W
            gb = layer.gradb

            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * gW
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * gb
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * gW ** 2
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * gb ** 2

            mW_hat = self.mW[i] / (1 - self.beta1 ** self.t)
            mb_hat = self.mb[i] / (1 - self.beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - self.beta2 ** self.t)
            vb_hat = self.vb[i] / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

class Nadam:
    def __init__(self, layers, lr, wd, beta1=0.9, beta2=0.999, eps=1e-8):
        self.layers = layers
        self.lr = lr
        self.wd = wd
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.mW = [np.zeros_like(l.W) for l in layers]
        self.mb = [np.zeros_like(l.b) for l in layers]
        self.vW = [np.zeros_like(l.W) for l in layers]
        self.vb = [np.zeros_like(l.b) for l in layers]

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            gW = layer.gradW + self.wd * layer.W
            gb = layer.gradb

            self.mW[i] = self.beta1 * self.mW[i] + (1 - self.beta1) * gW
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * gb
            self.vW[i] = self.beta2 * self.vW[i] + (1 - self.beta2) * gW ** 2
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * gb ** 2

            mW_hat = self.mW[i] / (1 - self.beta1 ** self.t)
            mb_hat = self.mb[i] / (1 - self.beta1 ** self.t)
            vW_hat = self.vW[i] / (1 - self.beta2 ** self.t)
            vb_hat = self.vb[i] / (1 - self.beta2 ** self.t)

            mW_bar = self.beta1 * mW_hat + (1 - self.beta1) * gW / (1 - self.beta1 ** self.t)
            mb_bar = self.beta1 * mb_hat + (1 - self.beta1) * gb / (1 - self.beta1 ** self.t)

            layer.W -= self.lr * mW_bar / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_bar / (np.sqrt(vb_hat) + self.eps)

def get_optimizer(args, layers):
    lr = args.learningrate
    wd = args.weightdecay

    if args.optimizer == "sgd":
        return SGD(layers, lr, wd)
    elif args.optimizer == "momentum":
        return Momentum(layers, lr, wd)
    elif args.optimizer == "nag":
        return NAG(layers, lr, wd)
    elif args.optimizer == "rmsprop":
        return RMSProp(layers, lr, wd)
    elif args.optimizer == "adam":
        return Adam(layers, lr, wd)
    elif args.optimizer == "nadam":
        return Nadam(layers, lr, wd)