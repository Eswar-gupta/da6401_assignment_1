import numpy as np
from keras.datasets import mnist, fashion_mnist


def load_data(dataset_name):
    if dataset_name in ["fashionmnist", "fashion_mnist"]:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0

    return X_train, np.eye(10)[y_train], X_test, np.eye(10)[y_test]
