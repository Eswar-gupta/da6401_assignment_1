import wandb
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.keras.utils import to_categorical

EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001

(X_train, y_train), (X_val, y_val) = mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_val   = X_val.astype("float32")   / 255.0
y_train = to_categorical(y_train, 10)
y_val   = to_categorical(y_val,   10)

optimizers = {
    "sgd":      SGD(learning_rate=LR),
    "momentum": SGD(learning_rate=LR, momentum=0.9, nesterov=False),
    "nag":      SGD(learning_rate=LR, momentum=0.9, nesterov=True),
    "rmsprop":  RMSprop(learning_rate=LR),
    "adam":     Adam(learning_rate=LR),
    "nadam":    Nadam(learning_rate=LR),
}

def build_model():
    return Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu", kernel_initializer="glorot_uniform"),
        Dense(128, activation="relu", kernel_initializer="glorot_uniform"),
        Dense(128, activation="relu", kernel_initializer="glorot_uniform"),
        Dense(10,  activation="softmax")
    ])

for opt_name, opt in optimizers.items():
    run = wandb.init(
        entity="eswar433-indian-institute-of-technology-madras",
        project="DA6401_assignement1_EE23B085",
        group="Q3_Optimizer_Compare",  # ← ALL 6 share this group
        name=f"Q3_{opt_name}",
        config={
            "optimizer": opt_name,
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "layers": 3,
            "neurons": 128,
            "activation": "relu"
        }
    )

    model = build_model()
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    for epoch in range(EPOCHS):
        h = model.fit(X_train, y_train, epochs=1, batch_size=BATCH_SIZE,
                      validation_data=(X_val, y_val), verbose=0)

        wandb.log({
            "epoch":      epoch + 1,
            "train_loss": h.history["loss"][0],
            "train_acc":  h.history["accuracy"][0],
            "val_loss":   h.history["val_loss"][0],
            "val_acc":    h.history["val_accuracy"][0],
        })

    _, test_acc = model.evaluate(X_val, y_val, verbose=0)
    wandb.log({"test_accuracy": test_acc})
    run.finish()
