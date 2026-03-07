import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('--model_path', type=str,default='src\best_model.npy')
    parser.add_argument('-d', '--dataset', type=str, default='mnist')
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-nhl', '--num_layers', type=int, default=2)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='*', default=64)
    parser.add_argument('-a', '--activation', type=str, default='relu')
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-l', '--loss', type=str, default='crossentropy')
    parser.add_argument('-o', '--optimizer', type=str, default='adam')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-wi', '--weight_init', type=str, default='xavier')

    return parser.parse_args()


def main():
    args = parse_arguments()

    args.weight_init = 'xavier'
    args.loss = 'crossentropy'
    args.optimizer = 'sgd'
    args.learning_rate = 0.001
    args.weight_decay = 0.0
    args.epochs = 1
    args.model_path = r"C:\Users\gurra\OneDrive\Desktop\1Acads\sem-6\DA6401-DL\Final_DL_Ass1\da6401_assignment_1\src\best_model.npy"

    _, _, X_test, y_test = load_data(args.dataset)

    net = NeuralNetwork(args)
    weights = np.load(args.model_path, allow_pickle=True).item()
    net.set_weights(weights)

    y_pred = net.forward(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true_labels, y_pred_labels)
    prec = precision_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
    rec = recall_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
    f1 = f1_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
