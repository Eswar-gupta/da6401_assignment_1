"""
test_forward_pass.py
====================
DA6401 Assignment-1 — Autograder: Forward Pass Verification (10 Marks)

Tests:
  T1  Model can be instantiated with CLI-like args             (1 mark)
  T2  forward() returns numpy array of shape (batch, 10)       (3 marks)
  T3  Output is a valid probability distribution (softmax)     (3 marks)
  T4  Output is deterministic for the same seed               (1 mark)
  T5  Works for varying batch sizes (1, 16, 64)               (2 marks)

Usage (run from the CLONED STUDENT REPO ROOT):
    python testing/test_forward_pass.py
"""

import sys
import os
import traceback
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH  = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_PATH)

# ── colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def info(msg): print(f"  {YELLOW}INFO{RESET}  {msg}")

# ── minimal mock args ──────────────────────────────────────────────────────────
class Args:
    dataset      = "mnist"
    numlayers    = 2
    hiddensize   = 64
    activation   = "relu"
    weightinit   = "xavier"
    loss         = "crossentropy"
    optimizer    = "adam"
    learningrate = 0.001
    weightdecay  = 0.0
    batchsize    = 32
    epochs       = 1

# ── score accumulator ─────────────────────────────────────────────────────────
score       = 0
max_score   = 10
passed      = []
failed      = []


def record(name, marks, ok_flag, reason=""):
    global score
    if ok_flag:
        score += marks
        passed.append(name)
        ok(f"[+{marks}] {name}")
    else:
        failed.append(name)
        fail(f"[ 0] {name}  ← {reason}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}=== Forward Pass Verification ==={RESET}")
print(f"Importing NeuralNetwork from {SRC_PATH} ...\n")

# ── T0: import ────────────────────────────────────────────────────────────────
try:
    from ann.neural_network import NeuralNetwork
    ok("NeuralNetwork imported successfully")
except Exception as e:
    fail(f"Could not import NeuralNetwork: {e}")
    print(f"\n{RED}Cannot continue — NeuralNetwork import failed.{RESET}")
    sys.exit(1)

# ── T1: instantiation ─────────────────────────────────────────────────────────
try:
    net = NeuralNetwork(Args())
    record("T1 Model instantiation", 1, True)
except Exception as e:
    record("T1 Model instantiation", 1, False, str(e))
    print(f"\n{RED}Cannot continue — constructor failed.{RESET}")
    sys.exit(1)

# ── T2: output shape ──────────────────────────────────────────────────────────
try:
    np.random.seed(42)
    X = np.random.randn(4, 784)
    out = net.forward(X)

    is_array = isinstance(out, np.ndarray)
    good_shape = is_array and out.shape == (4, 10)
    record("T2a Output is a numpy array", 1, is_array, f"got type {type(out)}")
    record("T2b Output shape is (batch=4, classes=10)", 2,
           good_shape, f"got shape {getattr(out,'shape','N/A')}")
except Exception as e:
    record("T2a Output is a numpy array", 1, False, traceback.format_exc(limit=2))
    record("T2b Output shape is (batch=4, classes=10)", 2, False, "forward() raised exception")

# ── T3: valid probability distribution ────────────────────────────────────────
try:
    np.random.seed(42)
    X = np.random.randn(8, 784)
    out = net.forward(X)

    all_nonneg   = bool(np.all(out >= -1e-6))
    sums_to_one  = bool(np.allclose(out.sum(axis=1), 1.0, atol=1e-4))
    record("T3a All outputs non-negative (softmax)", 1, all_nonneg,
           f"min value = {out.min():.4f}")
    record("T3b Row sums ≈ 1.0 (valid probability dist)", 2, sums_to_one,
           f"max row-sum deviation = {np.abs(out.sum(axis=1)-1).max():.4f}")
except Exception as e:
    record("T3a All outputs non-negative (softmax)", 1, False, str(e))
    record("T3b Row sums ≈ 1.0 (valid probability dist)", 2, False, str(e))

# ── T4: determinism ───────────────────────────────────────────────────────────
try:
    net2 = NeuralNetwork(Args())
    # copy weights from net to net2 to isolate randomness
    np.random.seed(7)
    X_det = np.random.randn(2, 784)
    o1 = net.forward(X_det)
    o2 = net.forward(X_det)
    deterministic = bool(np.allclose(o1, o2))
    record("T4 Deterministic output for same input", 1, deterministic,
           "two calls with same input gave different results")
except Exception as e:
    record("T4 Deterministic output for same input", 1, False, str(e))

# ── T5: varying batch sizes ───────────────────────────────────────────────────
batch_results = []
for bs in [1, 16, 64]:
    try:
        np.random.seed(0)
        X_bs = np.random.randn(bs, 784)
        o = net.forward(X_bs)
        batch_results.append(o.shape == (bs, 10))
    except Exception:
        batch_results.append(False)

record("T5 Works for batch sizes 1, 16, 64", 2,
       all(batch_results),
       f"results per batch: {dict(zip([1,16,64], batch_results))}")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}─── Forward Pass Score: {score} / {max_score} ───{RESET}")
if failed:
    print(f"{RED}Failed tests: {', '.join(failed)}{RESET}")
print()
