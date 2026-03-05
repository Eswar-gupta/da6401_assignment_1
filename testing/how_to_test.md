# DA6401 Assignment-1 — TA Testing Guide
### Complete Line-by-Line Testing Walkthrough

> **Who is this for?**  
> This guide is for a Teaching Assistant (TA) who may be new to grading this assignment. It explains **what the assignment asks students to build**, **why each test exists**, and **what you are looking for** at every step. Follow parts in order. Do NOT skip sections.

---

## What is This Assignment?

Students were asked to build a **Multi-Layer Perceptron (MLP) from scratch using only NumPy** — no PyTorch, no TensorFlow. The network must:

- Accept images from **MNIST** (handwritten digits) and **Fashion-MNIST** (clothing items) — 28×28 = 784-pixel inputs, 10 output classes.
- Support **6 optimizers**: SGD, Momentum, NAG, RMSProp, Adam, Nadam.
- Support **3 activations**: ReLU, Sigmoid, Tanh.
- Support **2 loss functions**: Cross-Entropy and Mean Squared Error.
- Be fully controllable via **command-line arguments** (so an autograder can drive it).
- Log all experiments to **Weights & Biases (W&B)**.
- Submit a `bestmodel.npy` (serialized NumPy weights) and a `bestconfig.json`.

**Grading split:**
- **50 marks** — Code implementation (auto-tested by scripts in this folder)
- **50 marks** — W&B experiment report (manually reviewed by you)

---

## Repository Structure — What Students Were Given vs. What They Had to Code

The course provided a **GitHub skeleton** at `https://github.com/MiRL-IITM/da6401_assignment_1`. This skeleton had empty files with docstrings and `pass` / `TODO` placeholders. Students had to **fill in every function themselves using only NumPy**.

The final submitted repo must have this exact structure:

```
<student_repo>/
│
├── README.md                    ← Student must write this
├── requirements.txt             ← Provided; student may add to it
│
├── src/
│   ├── train.py                 ← Student must implement fully
│   ├── inference.py             ← Student must implement fully
│   │
│   ├── ann/                     ← Core neural network module
│   │   ├── __init__.py
│   │   ├── activations.py       ← Student implements all activation functions
│   │   ├── neural_layer.py      ← Student implements a single MLP layer
│   │   ├── neural_network.py    ← Student implements the full MLP class
│   │   ├── objective_functions.py ← Student implements loss functions
│   │   └── optimizers.py        ← Student implements all 6 optimizers
│   │
│   └── utils/
│       ├── __init__.py
│       └── data_loader.py       ← Student implements data loading + preprocessing
│
└── models/
    ├── bestmodel.npy            ← Student generates this (serialized weights)
    └── bestconfig.json          ← Student generates this (best hyperparameters)
```

<details>
<summary><strong>▶ What students had to implement inside each file</strong></summary>

### `src/ann/activations.py`
Students implement every activation function and its derivative from scratch:
- `sigmoid(z)` — $\sigma(z) = \frac{1}{1+e^{-z}}$
- `sigmoid_derivative(z)`
- `tanh(z)` — $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
- `tanh_derivative(z)`
- `relu(z)` — $\max(0, z)$
- `relu_derivative(z)`
- `softmax(z)` — $\frac{e^{z_i}}{\sum_j e^{z_j}}$ (output layer only, never with derivatives directly)

> **What to check:** Are they using NumPy only? Any `torch.sigmoid` or similar is a violation.

---

### `src/ann/neural_layer.py`
Implements a **single layer** of the MLP. Each layer holds:
- `W` — weight matrix of shape `(input_size, output_size)`
- `b` — bias vector of shape `(1, output_size)`
- `gradW` — gradient of loss w.r.t. W (must be set after `backward()`)
- `gradb` — gradient of loss w.r.t. b (must be set after `backward()`)

Key methods:
- `__init__(input_size, output_size, activation, weight_init)` — initializes W and b using either random or Xavier
- `forward(X)` — computes `a = X @ W + b`, then applies activation → returns output
- `backward(grad_output)` — computes `gradW`, `gradb`, and returns `grad_input` for the previous layer

> **What to check:** Does `forward()` store the pre-activation `z` and post-activation `a`? These are needed for backprop. Does `backward()` correctly set `self.gradW` and `self.gradb`?

---

### `src/ann/neural_network.py`
Implements the **full MLP** by stacking multiple `NeuralLayer` objects. Key methods:

- `__init__(cli_args)` — reads args and builds a list `self.layers` of `NeuralLayer` objects
- `forward(X)` — passes X through all layers in sequence, returns softmax output
- `backward(y_true, y_pred)` — computes loss gradient, then calls each layer's `backward()` from output to input (chain rule)
- `update_weights()` — calls the optimizer to update `W` and `b` in each layer
- `train(X_train, y_train, epochs, batch_size)` — mini-batch training loop with W&B logging
- `evaluate(X, y)` — runs forward pass and computes metrics

> **What to check:** `self.layers` must be a list/array accessible from outside (autograder iterates it). The `backward()` must propagate gradients all the way back, not just to the last layer.

---

### `src/ann/objective_functions.py`
Implements loss functions:
- `cross_entropy(y_pred, y_true)` — $-\sum y \log(\hat{y})$, averaged over batch
- `mse(y_pred, y_true)` — $\frac{1}{n}\sum (y - \hat{y})^2$
- Corresponding gradient functions used by `backward()`

> **What to check:** Cross-entropy must be numerically stable (should clip predictions to avoid `log(0)`).

---

### `src/ann/optimizers.py`
Implements all 6 optimizers. Each optimizer maintains its own state (momentum buffers, etc.) and updates weights given gradients:

| Optimizer | State it maintains | Update rule |
|---|---|---|
| **SGD** | None | `w -= lr * grad` |
| **Momentum** | `v` (velocity) | `v = β*v + grad`, `w -= lr*v` |
| **NAG** | `v` (velocity) | Look-ahead gradient then update |
| **RMSProp** | `s` (squared grad avg) | `s = β*s + (1-β)*grad²`, `w -= lr*grad/√(s+ε)` |
| **Adam** | `m` (1st moment), `v` (2nd moment) | Bias-corrected m and v, then `w -= lr*m̂/√v̂` |
| **Nadam** | `m`, `v` | Adam + Nesterov look-ahead |

> **What to check:** Are moment buffers initialized to zero? Is bias correction applied in Adam? Are `m` and `v` stored between batches (not reset each step)?

---

### `src/utils/data_loader.py`
Loads and preprocesses the dataset:
- Loads MNIST or Fashion-MNIST via `keras.datasets`
- Normalizes pixel values to [0, 1] by dividing by 255
- Flattens 28×28 images to 784-dim vectors
- **Splits training data into train + validation** (the split must be random; test set must never be used during training)
- One-hot encodes labels into vectors of length 10

> **What to check:** Is the train/test split truly isolated? Using any test samples during training is an automatic **grade of zero**.

---

### `src/train.py`
The main entry point. Must:
1. Parse all 11 CLI arguments with `argparse`
2. Load data via `data_loader`
3. Build a `NeuralNetwork` with the given config
4. Run the training loop
5. Log loss and accuracy to W&B each epoch
6. Save the best model weights as `models/bestmodel.npy`

---

### `src/inference.py`
Must:
1. Parse CLI arguments (model path, dataset, architecture)
2. Reconstruct the network from the architecture args
3. Load weights from the `.npy` file into the network
4. Run the test set through the network (forward pass only)
5. Print: Accuracy, Precision, Recall, F1-score

---

### `models/bestmodel.npy`
A NumPy `.npy` file containing the trained weights. Saved with `np.save()`. The autograder loads this with `np.load()` and injects it into a freshly constructed network to test performance on held-out data.

### `models/bestconfig.json`
A JSON file with the hyperparameter configuration used to produce `bestmodel.npy`. Example:
```json
{
  "optimizer": "adam",
  "activation": "relu",
  "loss": "crossentropy",
  "numlayers": 3,
  "hiddensize": 128,
  "learningrate": 0.001,
  "weightdecay": 0.0005,
  "batchsize": 32,
  "weightinit": "xavier"
}
```

</details>

<details>
<summary><strong>▶ What the skeleton looked like (what students started with)</strong></summary>

The skeleton files had empty method bodies like this:

```python
# neural_network.py — as given to students
class NeuralNetwork:
    def __init__(self, cli_args):
        pass   # ← student must fill this

    def forward(self, X):
        pass   # ← student must fill this

    def backward(self, y_true, y_pred):
        pass   # ← student must fill this

    def update_weights(self):
        pass   # ← student must fill this
```

```python
# train.py — as given to students
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')
    # TODO: add all required arguments
    return parser.parse_args()

def main():
    args = parse_arguments()
    print("Training complete!")   # ← placeholder only
```

**Students had to replace every `pass` and `TODO` with working code.** If you see a submitted repo that still has `pass` or `print("Training complete!")` as the entire body of a function, that function was not implemented — deduct accordingly.

</details>

---

## What's in This `testing/` Folder?

<details>
<summary><strong>▶ Click to see all autograder files and what they do</strong></summary>

| File | What it does | Marks covered |
|---|---|---|
| `test_forward_pass.py` | Instantiates `NeuralNetwork`, feeds a fixed dummy input, checks output shape, softmax validity, and determinism | 10 |
| `test_gradients.py` | Runs `backward()`, verifies every layer exposes `.gradW`/`.gradb`, compares analytical vs finite-difference numerical gradients | 10 |
| `test_training.py` | Invokes `train.py` via subprocess for all 6 optimizers, both losses, all 3 activations, both init methods, and checks all 11 CLI flags | 15 |
| `test_inference.py` | Checks `bestmodel.npy` + `bestconfig.json` exist/are valid, runs `inference.py`, verifies it prints all 4 metrics | 10 |
| `run_all_tests.py` | **Master runner** — runs all four scripts above and prints a unified grade table | 45 (auto) |
| `how_to_test.md` | This file — complete TA guide | — |

</details>

---

## AUTOGRADER — One-Command Grade

> Run this **after** completing Parts 1–3 (clone + env setup). This covers all 45 automatically gradeable marks.

### Run everything in one shot

```powershell
# From the ROOT of the cloned student repo, after activating the venv:
python testing/run_all_tests.py
```

This prints a live test log followed by a grade table like:

```
══════════════════════════════════════════════════════════
  GRADE REPORT — Implementation (45 auto + 5 manual)
══════════════════════════════════════════════════════════
  Forward Pass Verification              8   /  10
  Gradient Consistency                  10   /  10
  Training Functionality                13   /  15
  Inference & Private Test               8   /  10
  ─────────────────────────────────── ──────  ────
  AUTO-GRADED TOTAL                     39   /  45
  Code Quality (manual)                  ?   /   5
  W&B Report (manual)                    ?   /  50
══════════════════════════════════════════════════════════
```

To run a single section:

```powershell
python testing/test_forward_pass.py
python testing/test_gradients.py
python testing/test_training.py
python testing/test_inference.py
```

> **The W&B Report (50 marks) and Code Quality (5 marks) require manual review — use Parts 4 and 17 of this guide.**

---

## PART 0 — Prerequisites *(Do this once on your machine)*

<details>
<summary><strong>▶ Why this matters</strong></summary>

The student's code uses Python 3.12 type hints and NumPy/Keras APIs that may not exist in older Python versions. Using a different Python version can cause silent failures or import errors that are the TA's fault, not the student's. Always test in a clean 3.12 environment.

</details>

Make sure you have the following installed before starting:

| Tool | Minimum Version | Check Command |
|---|---|---|
| Python | 3.12 | `py --version` |
| Git | Any recent | `git --version` |
| pip | ≥ 23 | `pip --version` |

---

## PART 1 — Clone the Student's Repository

<details>
<summary><strong>▶ Why clone fresh?</strong></summary>

You must test the student's **actual submitted code**, not a local copy. Cloning fresh ensures you have exactly what they pushed to GitHub. The assignment requires a **public** GitHub repo — if the repo is private or the link is broken, deduct under Code Quality.

</details>

```powershell
# Replace <STUDENT_GITHUB_URL> with the student's public GitHub repo link
# Example: https://github.com/EE23B085/da6401_assignment_1

git clone <STUDENT_GITHUB_URL>
cd <CLONED_REPO_FOLDER_NAME>
```

<details>
<summary><strong>▶ Expected folder structure (check against this)</strong></summary>

The assignment specifies a skeleton from `https://github.com/MiRL-IITM/da6401_assignment_1`. Each module has a specific role:

```
README.md
requirements.txt
src/
    train.py              ← main entry point for training
    inference.py          ← loads .npy weights, prints metrics
    ann/
        __init__.py
        activations.py        ← sigmoid, tanh, relu, softmax functions
        neural_layer.py       ← single layer: weights, forward, backward
        neural_network.py     ← full MLP: stacks layers, runs train loop
        objective_functions.py ← cross-entropy and MSE loss
        optimizers.py         ← SGD, Momentum, NAG, RMSProp, Adam, Nadam
    utils/
        __init__.py
        data_loader.py        ← loads MNIST / Fashion-MNIST via keras.datasets
models/
    bestmodel.npy         ← serialized NumPy weights of best run
    bestconfig.json       ← hyperparameter config that produced best model
```

If this structure is missing or broken, flag it under **Code Quality (5 Marks)**.

</details>
>
> ```
> README.md
> requirements.txt
> src/
>     train.py
>     inference.py
>     ann/
>         __init__.py
>         activations.py
>         neural_layer.py
>         neural_network.py
>         objective_functions.py
>         optimizers.py
>     utils/
>         __init__.py
>         data_loader.py
> models/
>     bestmodel.npy
>     bestconfig.json
> ```

---

## PART 2 — Paste the Testing Folder

<details>
<summary><strong>▶ Why paste the testing folder into the student repo?</strong></summary>

The autograder scripts use **relative paths** to find `src/` and `models/` — they expect to be located at `<repo_root>/testing/`. If you run them from a different location, path resolution fails. Always copy this folder to the root of each student's repo before running.

</details>

```powershell
# Run from INSIDE the cloned student repo directory.
# Adjust the source path to wherever you saved this testing folder.

Copy-Item -Recurse "C:\<PATH_TO_THIS_TESTING_FOLDER>" "."
```

Your directory should now look like:

```
<student_repo>/
    testing/          ← you just pasted this
        how_to_test.md
        run_all_tests.py
        test_forward_pass.py
        test_gradients.py
        test_training.py
        test_inference.py
    src/
    models/
    README.md
```

---

## PART 3 — Set Up the Python Environment

<details>
<summary><strong>▶ Why a fresh venv for every student?</strong></summary>

If you reuse your system Python, packages from other projects may interfere. A fresh `venv` guarantees you are testing with exactly what the student's `requirements.txt` specifies — the same environment the autograder uses. This prevents false failures and false passes.

</details>

```powershell
# Create a fresh virtual environment inside the student repo root
py -3.12 -m venv ta_test_env

# Activate it
.\ta_test_env\Scripts\Activate.ps1

# Upgrade pip to avoid resolver issues
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

Verify all packages are importable:

```powershell
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import keras; print('keras:', keras.__version__)"
python -c "import wandb; print('wandb:', wandb.__version__)"
python -c "import sklearn; print('sklearn:', sklearn.__version__)"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
```

<details>
<summary><strong>▶ What each package is used for in this assignment</strong></summary>

| Package | Role |
|---|---|
| `numpy` | **Core math** — ALL neural network operations must use only NumPy (no autograd) |
| `keras` | **Data loading only** — `keras.datasets.mnist` and `keras.datasets.fashion_mnist` |
| `wandb` | **Experiment tracking** — students must log loss/accuracy to W&B for the report |
| `scikit-learn` | **Metrics** — confusion matrix, precision, recall, F1-score |
| `matplotlib` | **Visualization** — plots for the W&B report sections |

> If `numpy` fails to import, nothing will work at all — deduct under Code Quality.

</details>

---

## PART 4 — Project Structure Check *(Code Quality — 5 Marks)*

<details>
<summary><strong>▶ What are we grading for Code Quality?</strong></summary>

Code Quality (5 marks total) covers three things:
1. **Structure** — Does the project match the required skeleton exactly?
2. **Documentation** — Are files commented? Is the README useful and complete?
3. **Coding style** — Is the code readable, modular, and not a giant spaghetti script in one file?

A student who dumped everything into `train.py` without using the `ann/` module structure should lose marks here, even if the code runs. The point of the skeleton is to enforce software engineering practices.

**Mark guide:**
- All files present, good README, commented code → 5/5
- Minor structural issues or thin README → 3–4/5
- Missing files or empty README → 1–2/5
- No README, no comments, wrong structure → 0/5

</details>

```powershell
# Check top-level structure
Get-ChildItem -Recurse -Depth 2 | Select-Object FullName

# Verify required files exist (each line should print True)
Test-Path "src\train.py"          # Must be True
Test-Path "src\inference.py"      # Must be True
Test-Path "src\ann\__init__.py"   # Must be True
Test-Path "src\ann\neural_layer.py"      # Must be True
Test-Path "src\ann\neural_network.py"    # Must be True
Test-Path "src\ann\activations.py"       # Must be True
Test-Path "src\ann\optimizers.py"        # Must be True
Test-Path "src\ann\objective_functions.py" # Must be True
Test-Path "src\utils\data_loader.py"     # Must be True
Test-Path "models\bestmodel.npy"         # Must be True
Test-Path "models\bestconfig.json"       # Must be True
Test-Path "README.md"                    # Must be True
```

<details>
<summary><strong>▶ README.md checklist (open and read manually)</strong></summary>

A good README should have:
- [ ] Instructions to install dependencies (`pip install -r requirements.txt`)
- [ ] A sample `python train.py ...` command with all arguments explained
- [ ] A sample `python inference.py ...` command
- [ ] Link to the public W&B report
- [ ] Link to the GitHub repo

Deduct marks if the README is empty, missing, or has no useful content.

</details>

---

## PART 5 — CLI Argument Check *(Training Functionality — 15 Marks)*

<details>
<summary><strong>▶ Why do CLI arguments matter so much?</strong></summary>

The assignment's automated grading pipeline drives `train.py` entirely via command-line flags. If a flag is missing, the autograder cannot control that hyperparameter — and the student loses marks for every test that depends on it. Think of this as an **API contract**: the student must implement the full interface exactly as specified. This is why 11 arguments are mandatory (not optional).

</details>

> **Covered by `test_training.py` automatically.** Manual commands below are for debugging specific failures.

```powershell
cd src
python train.py --help
```

> **Expected output should show ALL of the following flags:**
>
> | Flag | Description |
> |---|---|
> | `-d` / `--dataset` | mnist or fashionmnist |
> | `-e` / `--epochs` | Number of epochs |
> | `-b` / `--batchsize` | Mini-batch size |
> | `-l` / `--loss` | meansquarederror or crossentropy |
> | `-o` / `--optimizer` | sgd, momentum, nag, rmsprop, adam, nadam |
> | `-lr` / `--learningrate` | Float learning rate |
> | `-wd` / `--weightdecay` | L2 regularization weight decay |
> | `-nhl` / `--numlayers` | Number of hidden layers |
> | `-sz` / `--hiddensize` | Neurons per hidden layer |
> | `-a` / `--activation` | sigmoid, tanh, relu |
> | `-wi` / `--weightinit` | random or xavier |
>
<details>
<summary><strong>▶ What each flag controls</strong></summary>

| Flag | What it controls | Valid values |
|---|---|---|
| `-d` / `--dataset` | Which dataset to load | `mnist`, `fashionmnist` |
| `-e` / `--epochs` | Training iterations over full data | Any positive int |
| `-b` / `--batchsize` | Mini-batch size | Any positive int |
| `-l` / `--loss` | Loss function | `crossentropy`, `meansquarederror` |
| `-o` / `--optimizer` | Weight update rule | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` |
| `-lr` / `--learningrate` | Step size for weight updates | Float, e.g. `0.001` |
| `-wd` / `--weightdecay` | L2 regularization coefficient | Float, e.g. `0` or `0.0005` |
| `-nhl` / `--numlayers` | Number of hidden layers | Int ≤ 6 |
| `-sz` / `--hiddensize` | Neurons per hidden layer | Int ≤ 128 |
| `-a` / `--activation` | Hidden layer activation | `relu`, `sigmoid`, `tanh` |
| `-wi` / `--weightinit` | Weight initialization strategy | `random`, `xavier` |

</details>

Deduct marks for every missing argument.

Similarly for `inference.py`:

```powershell
python inference.py --help
```

> **Expected output should show:**  
> `--model_path`, `--dataset`, `--batchsize`, `--numlayers`, `--hiddensize`, `--activation`

---

## PART 6 — Quick Smoke Test *(Training Functionality — 15 Marks)*

<details>
<summary><strong>▶ What is a smoke test?</strong></summary>

A smoke test checks that the code **runs at all** — it does not test correctness. If `train.py` raises a Python error or crashes immediately, the student gets 0 for training functionality regardless of how good the underlying math might be. This is the baseline sanity check before running the full test battery.

Signs of a healthy run:
- No `Traceback` or `Error` in output
- Some indication of progress (epoch number, loss value, or accuracy printed)
- Finishes within a reasonable time (< 5 minutes for 1 epoch on CPU)

</details>

Run a minimal training run (1 epoch, small batch) to check that `train.py` does not crash:

```powershell
# Still inside src/ directory
python train.py -d mnist -e 1 -b 32 -l crossentropy -o adam -lr 0.001 -wd 0 -nhl 2 -sz 64 -a relu -wi xavier
```

> **Expected:** Script runs without Python errors. Some form of loss/accuracy output per epoch is ideal.  
> If it crashes, note the exact error and deduct under **Training Functionality**.

Run it again with `fashionmnist` to verify dataset switching:

```powershell
python train.py -d fashionmnist -e 1 -b 32 -l crossentropy -o adam -lr 0.001 -wd 0 -nhl 2 -sz 64 -a relu -wi xavier
```

---

## PART 7 — Forward Pass Verification *(10 Marks)*

> **Autoscript:** `python testing/test_forward_pass.py`

<details>
<summary><strong>▶ What is the forward pass and why does it get 10 marks?</strong></summary>

The **forward pass** is the core computation: given an input image (784 pixels), each layer applies `output = activation(W @ input + b)`, and the final layer applies **softmax** to produce a probability distribution over 10 classes.

This test gets 10 marks because if the forward pass is wrong, **nothing else works** — training gives wrong loss, gradients are meaningless, and inference is garbage. The script checks:

1. Output shape is `(batch_size, 10)` — one probability per class per sample.
2. All values ≥ 0 (softmax outputs are always non-negative).
3. Each row sums to ~1.0 (valid probability distribution).
4. Same input always gives same output (no accidental randomness in forward pass).
5. Works for batch sizes 1, 16, and 64.

</details>

This section verifies that the model produces **correct output logits** for a known input.

```powershell
# Still inside src/
python - << 'EOF'
import numpy as np
import sys
sys.path.insert(0, '.')

# Load the student's NeuralNetwork class
from ann.neural_network import NeuralNetwork

# Create a minimal mock args object
class Args:
    dataset = 'mnist'
    numlayers = 2
    hiddensize = 64
    activation = 'relu'
    weightinit = 'xavier'
    loss = 'crossentropy'
    optimizer = 'adam'
    learningrate = 0.001
    weightdecay = 0
    batchsize = 32
    epochs = 1

args = Args()
net = NeuralNetwork(args)

# Create a fixed dummy input: 1 sample, 784 features (flattened 28x28)
np.random.seed(42)
X_fixed = np.random.randn(1, 784)

# Run forward pass
output = net.forward(X_fixed)
print("Output shape:", output.shape)  # Expected: (1, 10)
print("Output sum (should be ~1 for softmax):", output.sum())
print("All outputs non-negative:", np.all(output >= 0))
print("FORWARD PASS: OK")
EOF
```

> **Expected:**
> - `Output shape: (1, 10)` — 10 classes
> - `Output sum: ~1.0` — confirms softmax output
> - `All outputs non-negative: True`
> - No Python errors

---

## PART 8 — Gradient Consistency Check *(10 Marks)*

> **Autoscript:** `python testing/test_gradients.py`

<details>
<summary><strong>▶ What is gradient checking and why does it get 10 marks?</strong></summary>

The **backward pass** (backpropagation) computes gradients — how much each weight should change to reduce the loss. If implemented incorrectly, the network appears to train (loss decreases) but learns nothing useful.

**Gradient checking** is the gold-standard test for backprop correctness. It uses the definition of a derivative:

$$\frac{\partial L}{\partial w_{ij}} \approx \frac{L(w_{ij} + \varepsilon) - L(w_{ij} - \varepsilon)}{2\varepsilon}$$

We compute this **numerically** (by perturbing weights and measuring the loss change) and compare it to what `backward()` computed **analytically**. If they agree to within `1e-7`, the implementation is correct.

**Grading:**
- Max difference `< 1e-7` → **10/10** (full marks)
- Max difference `< 1e-5` → **6/10** (partial credit)
- Max difference `≥ 1e-5` or crash → **0/10**

This also checks that every layer exposes `.gradW` and `.gradb` after `backward()` — the spec explicitly requires this for the autograder to read them.

</details>

This is the most critical check. Analytical gradients must match numerical gradients within tolerance `1e-7`.

```powershell
python - << 'EOF'
import numpy as np
import sys
sys.path.insert(0, '.')

from ann.neural_network import NeuralNetwork

class Args:
    dataset = 'mnist'
    numlayers = 1
    hiddensize = 8
    activation = 'relu'
    weightinit = 'xavier'
    loss = 'crossentropy'
    optimizer = 'sgd'
    learningrate = 0.01
    weightdecay = 0
    batchsize = 4
    epochs = 1

args = Args()
net = NeuralNetwork(args)

np.random.seed(0)
X = np.random.randn(4, 784)
y = np.eye(10)[[0, 1, 2, 3]]  # one-hot labels for 4 samples

# Analytical gradients
y_pred = net.forward(X)
net.backward(y, y_pred)

# Grab gradients from first layer
# The student's layer objects must expose .gradW and .gradb
first_layer = net.layers[0]
analytical_gradW = first_layer.gradW.copy()

# Numerical gradients using finite differences (epsilon = 1e-5)
epsilon = 1e-5
numerical_gradW = np.zeros_like(first_layer.W)
for i in range(min(3, first_layer.W.shape[0])):   # check first 3 rows only for speed
    for j in range(min(3, first_layer.W.shape[1])): # check first 3 cols only
        first_layer.W[i, j] += epsilon
        loss_plus = net.objective(net.forward(X), y)
        first_layer.W[i, j] -= 2 * epsilon
        loss_minus = net.objective(net.forward(X), y)
        first_layer.W[i, j] += epsilon
        numerical_gradW[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

diff = np.max(np.abs(analytical_gradW[:3, :3] - numerical_gradW[:3, :3]))
print(f"Max gradient difference: {diff:.2e}")
if diff < 1e-5:
    print("GRADIENT CHECK: PASSED (diff < 1e-5)")
else:
    print("GRADIENT CHECK: FAILED — difference too large")
EOF
```

> **Grading note:**  
> - Difference `< 1e-7` → Full 10 marks  
> - Difference between `1e-7` and `1e-5` → Partial marks  
> - Difference `> 1e-5` or crash → 0 marks

---

## PART 9 — Optimizer Variety Test *(Training Functionality — 15 Marks)*

> **Autoscript:** `python testing/test_training.py` — covers Parts 9–12 and Part 5 all at once.

<details>
<summary><strong>▶ Why are there 6 optimizers and why test each one?</strong></summary>

Each optimizer is a different update rule for weights. Students must implement all 6 **from scratch using NumPy** (no `torch.optim`):

| Optimizer | What makes it different |
|---|---|
| **SGD** | Simplest: `w -= lr * grad` |
| **Momentum** | Adds a velocity term to smooth updates and escape local minima |
| **NAG** | Nesterov Accelerated Gradient — looks ahead before computing the gradient |
| **RMSProp** | Adapts learning rate per-parameter using moving average of squared gradients |
| **Adam** | Combines Momentum + RMSProp — most commonly used in practice |
| **Nadam** | Adam + Nesterov momentum |

If an optimizer crashes, it's likely a shape mismatch in the update formula or missing initialization of momentum/velocity buffers.

</details>

Run one epoch with each of the 6 required optimizers. None of them should crash.

```powershell
foreach ($opt in @("sgd", "momentum", "nag", "rmsprop", "adam", "nadam")) {
    Write-Host "`n--- Testing optimizer: $opt ---"
    python train.py -d mnist -e 1 -b 64 -l crossentropy -o $opt -lr 0.001 -wd 0 -nhl 2 -sz 64 -a relu -wi xavier
}
```

> Log which optimizers pass or fail.

---

## PART 10 — Loss Function Test *(Training Functionality — 15 Marks)*

<details>
<summary><strong>▶ Why two loss functions?</strong></summary>

**Cross-Entropy** is the standard loss for multi-class classification — it directly penalizes the negative log-likelihood of the correct class and produces strong gradients early in training.

**Mean Squared Error (MSE)** is more natural for regression, but the assignment asks students to implement both and observe the difference (covered in W&B section 2.6). A common bug: MSE with softmax produces very small gradients early on (the gradient of MSE saturates), so learning is much slower.

</details>

```powershell
Write-Host "--- Testing crossentropy ---"
python train.py -d mnist -e 1 -b 64 -l crossentropy -o adam -lr 0.001 -wd 0 -nhl 2 -sz 64 -a relu -wi xavier

Write-Host "--- Testing meansquarederror ---"
python train.py -d mnist -e 1 -b 64 -l meansquarederror -o adam -lr 0.001 -wd 0 -nhl 2 -sz 64 -a relu -wi xavier
```

---

## PART 11 — Activation Function Test *(Training Functionality — 15 Marks)*

<details>
<summary><strong>▶ Why three activation functions?</strong></summary>

Activations introduce non-linearity — without them a deep network collapses to a single linear transformation. Each has different properties:

| Activation | Output range | Common issue |
|---|---|---|
| **ReLU** | [0, ∞) | "Dead neurons" when weights go very negative (gradient = 0 permanently) |
| **Sigmoid** | (0, 1) | Vanishing gradients in deep networks — saturates at both ends |
| **Tanh** | (−1, 1) | Better than sigmoid (zero-centered), but still saturates |

The assignment has entire W&B sections (2.4 and 2.5) dedicated to analysing these differences. This code test just checks that each one runs without crashing.

</details>

```powershell
foreach ($act in @("relu", "sigmoid", "tanh")) {
    Write-Host "`n--- Testing activation: $act ---"
    python train.py -d mnist -e 1 -b 64 -l crossentropy -o adam -lr 0.001 -wd 0 -nhl 2 -sz 64 -a $act -wi xavier
}
```

---

## PART 12 — Weight Initialization Test *(Training Functionality — 15 Marks)*

<details>
<summary><strong>▶ Why does weight initialization matter?</strong></summary>

If all weights start at zero, every neuron in a layer produces the same output and receives the same gradient — they all learn identically and the network never differentiates features. This is called the **symmetry problem**.

**Xavier initialization** (Glorot) samples weights from a distribution scaled by the layer size:
$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}+n_{out}}}, +\sqrt{\frac{6}{n_{in}+n_{out}}}\right)$$
This keeps activation variances stable across layers. The assignment's W&B section 2.9 asks students to demonstrate this with gradient plots.

</details>

```powershell
Write-Host "--- Testing xavier init ---"
python train.py -d mnist -e 1 -b 64 -l crossentropy -o adam -lr 0.001 -wd 0 -nhl 2 -sz 64 -a relu -wi xavier

Write-Host "--- Testing random init ---"
python train.py -d mnist -e 1 -b 64 -l crossentropy -o adam -lr 0.001 -wd 0 -nhl 2 -sz 64 -a relu -wi random
```

---

## PART 13 — Inference Script Test *(Private Test Performance — 10 Marks)*

> **Autoscript:** `python testing/test_inference.py`

<details>
<summary><strong>▶ What is inference.py supposed to do?</strong></summary>

After training, students save their best-performing model as `models/bestmodel.npy` — a serialized NumPy array of all weights and biases. `inference.py` must:

1. Load that `.npy` file and reconstruct the network architecture using the config.
2. Run the test set through the network (forward passes only, no training).
3. Print four metrics:
   - **Accuracy** — fraction of correctly classified samples
   - **Precision** — of all predicted-positive, how many were truly positive (averaged across classes)
   - **Recall** — of all actual-positive, how many did we catch (averaged across classes)
   - **F1-score** — harmonic mean of precision and recall

The `bestconfig.json` exists so that `inference.py` knows the architecture (number of layers, neurons, activation) needed to reconstruct the model from the raw `.npy` weights.

</details>

First, check that the best model files exist:

```powershell
# Go back to repo root
cd ..

Test-Path "models\bestmodel.npy"    # Must be True
Test-Path "models\bestconfig.json"  # Must be True
```

Inspect the config file:

```powershell
Get-Content "models\bestconfig.json"
```

> **Expected:** A JSON file with keys like `optimizer`, `activation`, `numlayers`, `hiddensize`, `learningrate`, etc.

Now run inference:

```powershell
cd src

# Adjust -nhl and -sz to match what is inside bestconfig.json
python inference.py --model_path ..\models\bestmodel.npy -d mnist -b 64 -nhl 3 -sz 128 -a relu
```

> **Expected output must include (check stdout):**
> - `Accuracy: XX.XX%`
> - `Precision: XX.XX`
> - `Recall: XX.XX`
> - `F1-score: XX.XX`
>
> If any metric is missing, deduct under this section.

---

## PART 14 — gradW and gradb Exposure Check *(Gradient Consistency — 10 Marks)*

> **Covered by autoscript:** `python testing/test_gradients.py` (tests T2 and T3).

<details>
<summary><strong>▶ Why must layer objects expose .gradW and .gradb?</strong></summary>

The assignment spec explicitly states:

> *"layer objects must expose `self.gradW` and `self.gradb` after every call for verification"*

This is because the autograder reaches into layer objects after `backward()` to read gradients for the numerical gradient check. A student might compute correct gradients internally but never store them as attributes — this still fails the autograder.

</details>

After a backward pass, every layer object **must** expose `self.gradW` and `self.gradb` as attributes:

```powershell
python - << 'EOF'
import numpy as np
import sys
sys.path.insert(0, '.')

from ann.neural_network import NeuralNetwork

class Args:
    dataset = 'mnist'
    numlayers = 2
    hiddensize = 16
    activation = 'relu'
    weightinit = 'xavier'
    loss = 'crossentropy'
    optimizer = 'adam'
    learningrate = 0.001
    weightdecay = 0
    batchsize = 4
    epochs = 1

net = NeuralNetwork(Args())
X = np.random.randn(4, 784)
y = np.eye(10)[[0,1,2,3]]
y_pred = net.forward(X)
net.backward(y, y_pred)

for i, layer in enumerate(net.layers):
    has_gradW = hasattr(layer, 'gradW') and layer.gradW is not None
    has_gradb = hasattr(layer, 'gradb') and layer.gradb is not None
    print(f"Layer {i}: gradW={'OK' if has_gradW else 'MISSING'}, gradb={'OK' if has_gradb else 'MISSING'}")
EOF
```

> **Expected:** Every layer prints `gradW=OK, gradb=OK`.  
> Any `MISSING` → deduct under Gradient Consistency.

---

## PART 15 — W&B Logging Check *(Training Functionality — 15 Marks)*

<details>
<summary><strong>▶ What should be logged to W&B and why does it matter?</strong></summary>

Weights & Biases (W&B) is a platform for tracking ML experiments. The assignment requires students to log every run so that the W&B report (50 marks) can be produced. Without W&B logging:
- The sweep (Section 2.2) with ≥ 100 runs cannot exist.
- The comparison plots (optimizer showdown, vanishing gradient, etc.) cannot be produced.
- The student cannot write the report.

At minimum, each `train.py` run should log:
- Loss per epoch (training loss)
- Accuracy per epoch (training and validation)
- Hyperparameters visible in the run config panel (so the Parallel Coordinates plot works)

</details>

Ask the student for their W&B project name and entity. Then run:

```powershell
python train.py -d mnist -e 2 -b 64 -l crossentropy -o adam -lr 0.001 -wd 0 -nhl 2 -sz 64 -a relu -wi xavier
```

After the run, visit `https://wandb.ai/<student_entity>/<project_name>` and check:

<details>
<summary><strong>▶ W&B run checklist</strong></summary>

- [ ] A new run appeared in the W&B project after the script finished
- [ ] The run has a loss curve (at least one logged metric per epoch)
- [ ] The run shows hyperparameters in the config tab (`optimizer`, `activation`, etc.)
- [ ] Validation/test accuracy is logged (not just training accuracy)

No W&B logging at all → deduct marks under Training Functionality.

</details>

---

## PART 16 — Private Test Score *(10 Marks)*

<details>
<summary><strong>▶ What is the private test set?</strong></summary>

The assignment states that `bestmodel.npy` will be evaluated against a **held-out private dataset** not seen during training. You run `inference.py` on it and record the F1-score. F1 is used (not just accuracy) because it is more robust to class imbalance — some Fashion-MNIST categories (like shirt vs T-shirt) are visually similar and harder to classify, pulling accuracy down unevenly.

</details>

Run inference on the held-out test data:

```powershell
cd src
# Read bestconfig.json first, then match -nhl, -sz, -a to those values
python inference.py --model_path ..\models\bestmodel.npy -d fashionmnist -b 64 -nhl 3 -sz 128 -a relu
```

Note the **F1-score** and use the rubric below:

| F1-Score (Fashion-MNIST) | Marks |
|---|---|
| ≥ 0.88 | 10/10 |
| 0.82 – 0.87 | 7/10 |
| 0.75 – 0.81 | 5/10 |
| 0.65 – 0.74 | 3/10 |
| < 0.65 | 0/10 |

---

## PART 17 — W&B Report Checklist *(50 Marks)*

<details>
<summary><strong>▶ What is the W&B report?</strong></summary>

The W&B report is a written + visual document inside Weights & Biases — like a lab report. Students must answer 10 specific questions with plots generated from their actual training runs, plus written analysis. This is worth 50 marks (half of the total assignment).

A good report has:
- Actual plots generated from real runs (not screenshots from Google Images or made-up curves)
- Written analysis explaining what the plots show and what it means conceptually
- Correct terminology (the student understands what they are plotting)

**Common issues to watch for:**
- Fake/suspiciously perfect loss curves
- Missing written answers (plots with no explanation)
- Sweep with < 100 runs
- Wrong architecture for optimizer showdown (must be 3 layers × 128 neurons × ReLU)

</details>

Open the student's public W&B report link and verify each section:

| Section | What to look for | Marks |
|---|---|---|
| **2.1 Data Exploration** | W&B Table with 5 real images from each of the 10 classes (50 total). Student identifies visually similar classes (e.g. shirt vs T-shirt in Fashion-MNIST) | 3 |
| **2.2 Hyperparameter Sweep** | ≥ 100 runs in a W&B Sweep, Parallel Coordinates plot present, best config clearly stated with values | 6 |
| **2.3 Optimizer Showdown** | All 6 optimizers on same plot, **same arch: 3 hidden layers × 128 neurons × ReLU**, convergence compared over first 5 epochs minimum | 5 |
| **2.4 Vanishing Gradient** | Gradient L2-norm of **first hidden layer** plotted per epoch for Sigmoid vs ReLU. Sigmoid norms should visibly decay toward zero for deeper nets | 5 |
| **2.5 Dead Neuron** | A run with ReLU + high LR (e.g. 0.1) showing plateau in validation accuracy. Activation histogram showing neurons stuck at zero. Comparison with Tanh run and explanation | 6 |
| **2.6 Loss Comparison** | MSE and Cross-Entropy training curves on same plot, same architecture. Written explanation of why Cross-Entropy with Softmax is better for classification | 4 |
| **2.7 Global Performance** | Overlay plot of train vs test accuracy across **all** runs. Student identifies overfitting runs (large train–test gap) | 4 |
| **2.8 Error Analysis** | Confusion matrix for best model on test set. Bonus: creative visualization of failures (most-confused pairs, etc.) | 5 |
| **2.9 Weight Initialization** | Two runs: Zeros and Xavier. Line plot of gradients for **5 specific neurons** in same layer over first 50 iterations. Zeros run must show overlapping lines. Written explanation of symmetry | 7 |
| **2.10 Fashion-MNIST Transfer** | **Exactly 3** configs tried on Fashion-MNIST, accuracies reported, discussion of whether MNIST best config transferred and why | 5 |

---

## PART 18 — Final Cleanup

When done grading this student:

```powershell
# Deactivate the virtual environment
deactivate

# Delete the env (start fresh for the next student)
Remove-Item -Recurse -Force ta_test_env

# Go up a directory and delete the cloned repo
cd ..
Remove-Item -Recurse -Force <CLONED_REPO_FOLDER_NAME>
```

Then repeat from **PART 1** for the next student.

---

## Quick Grading Summary Sheet

| Criterion | Max Marks | Autoscript | Manual Parts |
|---|---|---|---|
| Forward Pass Verification | 10 | `test_forward_pass.py` | Part 7 |
| Gradient Consistency | 10 | `test_gradients.py` | Parts 8, 14 |
| Training Functionality | 15 | `test_training.py` | Parts 5, 6, 9–12, 15 |
| Private Test Performance | 10 | `test_inference.py` | Parts 13, 16 |
| Code Quality | 5 | *(manual only)* | Part 4 |
| W&B Report | 50 | *(manual only)* | Part 17 |
| **Total** | **100** | `run_all_tests.py` for 45 marks | Parts 4, 17 for remaining 55 |

---

*Guide prepared for DA6401 — Introduction to Deep Learning, Assignment 1, Spring 2026.*
