import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
import numpy as np
import matplotlib.pyplot as plt
import wandb

import random

import wandb
wandb.login("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")
# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="eswar433-indian-institute-of-technology-madras",
    # Set the wandb project where this run will be logged.
    project="DA6401_assignement1_EE23B085",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
    name="data_exploration"
)

# Simulate training.
import wandb
from keras.datasets import fashion_mnist # or mnist
import numpy as np

(X_train, y_train), (_, _) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

table = wandb.Table(columns=["Image", "Class Name", "Label"])
for label in range(10):
    indices = np.where(y_train == label)[0]
    samples = indices[:5]
    
    for idx in samples:
        img = X_train[idx]
        table.add_data(wandb.Image(img), class_names[label], label)
        
wandb.log({"Fashion-MNIST Samples": table})
run.finish()