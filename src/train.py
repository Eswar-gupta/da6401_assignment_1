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
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-l', '--loss', type=str, default='crossentropy')
    parser.add_argument('-o', '--optimizer', type=str, default='adam')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-nhl', '--num_layers', type=int, default=2)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='*', default=[128,64])
    parser.add_argument('-a', '--activation', type=str, default='relu')
    parser.add_argument('-wi', '--weight_init', type=str, default='xavier')
    return parser.parse_args()


def main():
    args = parse_arguments()

    X_train, y_train, X_test, y_test = load_data(args.dataset)

    net = NeuralNetwork(args)
    net.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

    _, test_loss, test_acc = net.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.4f} - Test acc: {test_acc:.4f}")

    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    np.save(os.path.join(ROOT_DIR,'src','best_model.npy'), net.get_weights(), allow_pickle=True)

    config = {
        'dataset': args.dataset,
        'optimizer': args.optimizer,
        'activation': args.activation,
        'loss': args.loss,
        'num_layers': args.num_layers,
        'hidden_size': args.hidden_size,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'weight_init': args.weight_init,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
    }
    with open(os.path.join(ROOT_DIR,'src', 'best_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("Training complete!")


if __name__ == '__main__':
    main()
