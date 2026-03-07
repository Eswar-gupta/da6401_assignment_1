import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
import numpy as np
import matplotlib.pyplot as plt
import wandb
import random

run = wandb.init(
    entity="eswar433-indian-institute-of-technology-madras",
    project="DA6401_assignement1_EE23B085",
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "MNIST",
        "epochs": 10,
    },
    name="Q1"
)

from keras.datasets import mnist
import numpy as np

(X_train, y_train), (_, _) = mnist.load_data()

class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']

table = wandb.Table(columns=["Image", "Class Name", "Label"])
for label in range(10):
    indices = np.where(y_train == label)[0]
    samples = indices[:5]
    
    for idx in samples:
        img = X_train[idx]
        table.add_data(wandb.Image(img), class_names[label], label)
        
wandb.log({"MNIST Samples": table})
run.finish()