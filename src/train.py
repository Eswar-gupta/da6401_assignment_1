import argparse
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('-d', '--dataset', type=str, default='mnist')
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-b', '--batchsize', type=int, default=32)
    parser.add_argument('-l', '--loss', type=str, default='crossentropy')
    parser.add_argument('-o', '--optimizer', type=str, default='adam')
    parser.add_argument('-lr', '--learningrate', type=float, default=0.001)
    parser.add_argument('-wd', '--weightdecay', type=float, default=0.0)
    parser.add_argument('-nhl', '--numlayers', type=int, default=2)
    parser.add_argument('-sz', '--hiddensize', type=int, default=64)
    parser.add_argument('-a', '--activation', type=str, default='relu')
    parser.add_argument('-wi', '--weightinit', type=str, default='xavier')
    return parser.parse_args()


def main():
    args = parse_arguments()

    X_train, y_train, X_test, y_test = load_data(args.dataset)

    net = NeuralNetwork(args)
    net.train(X_train, y_train, epochs=args.epochs, batch_size=args.batchsize)

    _, test_loss, test_acc = net.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.4f} - Test acc: {test_acc:.4f}")

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    np.save(os.path.join(models_dir, 'bestmodel.npy'), net.get_weights(), allow_pickle=True)

    config = {
        'dataset': args.dataset,
        'optimizer': args.optimizer,
        'activation': args.activation,
        'loss': args.loss,
        'numlayers': args.numlayers,
        'hiddensize': args.hiddensize,
        'learningrate': args.learningrate,
        'batchsize': args.batchsize,
        'weightinit': args.weightinit,
        'weightdecay': args.weightdecay,
        'epochs': args.epochs,
    }
    with open(os.path.join(models_dir, 'bestconfig.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("Training complete!")


if __name__ == '__main__':
    main()
