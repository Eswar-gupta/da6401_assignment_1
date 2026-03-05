"""
test_gradients.py
=================
DA6401 Assignment-1 — Autograder: Gradient Consistency (10 Marks)

Tests:
  T1  backward() runs without error                               (1 mark)
  T2  Every layer exposes .gradW and .gradb after backward()      (2 marks)
  T3  Gradient shapes match weight/bias shapes                    (2 marks)
  T4  Numerical vs analytical gradient check  (tol 1e-7 → full)  (5 marks)
       ├─ diff < 1e-7  → 5/5
       ├─ diff < 1e-5  → 3/5
       └─ diff ≥ 1e-5  → 0/5

Usage (run from the CLONED STUDENT REPO ROOT):
    python testing/test_gradients.py
"""

import sys
import os
import traceback
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH  = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_PATH)

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
RESET  = "\033[0m";  BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}WARN{RESET}  {msg}")

score, max_score = 0, 10
passed, failed_tests = [], []

def record(name, marks, ok_flag, reason=""):
    global score
    if ok_flag:
        score += marks; passed.append(name)
        ok(f"[+{marks}] {name}")
    else:
        failed_tests.append(name)
        fail(f"[ 0] {name}  ← {reason}")


class Args:
    dataset = "mnist"; numlayers = 1; hiddensize = 8
    activation = "relu"; weightinit = "xavier"; loss = "crossentropy"
    optimizer = "sgd"; learningrate = 0.01; weightdecay = 0.0
    batchsize = 4; epochs = 1


print(f"\n{BOLD}=== Gradient Consistency Check ==={RESET}")
print(f"Importing NeuralNetwork from {SRC_PATH} ...\n")

try:
    from ann.neural_network import NeuralNetwork
    ok("NeuralNetwork imported")
except Exception as e:
    fail(f"Import failed: {e}"); sys.exit(1)

try:
    net = NeuralNetwork(Args())
    ok("Model instantiated")
except Exception as e:
    fail(f"Constructor failed: {e}"); sys.exit(1)

# ── T1: backward() runs without error ─────────────────────────────────────────
np.random.seed(0)
X = np.random.randn(4, 784)
y = np.eye(10)[[0, 1, 2, 3]]

try:
    y_pred = net.forward(X)
    net.backward(y, y_pred)
    record("T1 backward() executes without error", 1, True)
except Exception as e:
    record("T1 backward() executes without error", 1, False, traceback.format_exc(limit=3))
    print(f"\n{RED}Cannot continue — backward() raised an exception.{RESET}")
    sys.exit(1)

# ── T2: gradW and gradb exposed on every layer ────────────────────────────────
try:
    layers = net.layers
    all_have_gradW = all(hasattr(l, "gradW") and l.gradW is not None for l in layers)
    all_have_gradb = all(hasattr(l, "gradb") and l.gradb is not None for l in layers)
    for i, l in enumerate(layers):
        gw = "OK" if (hasattr(l,"gradW") and l.gradW is not None) else "MISSING"
        gb = "OK" if (hasattr(l,"gradb") and l.gradb is not None) else "MISSING"
        print(f"         Layer {i}: gradW={gw}, gradb={gb}")
    record("T2a All layers expose .gradW", 1, all_have_gradW,
           "some layers missing .gradW")
    record("T2b All layers expose .gradb", 1, all_have_gradb,
           "some layers missing .gradb")
except AttributeError:
    record("T2a All layers expose .gradW", 1, False, "net.layers attribute missing")
    record("T2b All layers expose .gradb", 1, False, "net.layers attribute missing")

# ── T3: gradient shape matches weight/bias shape ──────────────────────────────
try:
    layers = net.layers
    shapes_ok_W = all(l.gradW.shape == l.W.shape for l in layers
                      if hasattr(l,"gradW") and l.gradW is not None
                      and hasattr(l,"W") and l.W is not None)
    shapes_ok_b = all(l.gradb.shape == l.b.shape for l in layers
                      if hasattr(l,"gradb") and l.gradb is not None
                      and hasattr(l,"b") and l.b is not None)
    record("T3a gradW.shape == W.shape for all layers", 1, shapes_ok_W,
           "shape mismatch in at least one layer")
    record("T3b gradb.shape == b.shape for all layers", 1, shapes_ok_b,
           "shape mismatch in at least one layer")
except Exception as e:
    record("T3a gradW.shape == W.shape for all layers", 1, False, str(e))
    record("T3b gradb.shape == b.shape for all layers", 1, False, str(e))

# ── T4: numerical vs analytical gradient check ────────────────────────────────
print(f"\n  Running finite-difference gradient check (this may take a few seconds)...")
try:
    first_layer = net.layers[0]
    analytical_gradW = first_layer.gradW.copy()

    epsilon = 1e-5
    check_rows = min(5, first_layer.W.shape[0])
    check_cols = min(5, first_layer.W.shape[1])
    numerical_gradW = np.zeros((check_rows, check_cols))

    # helper: recompute loss cleanly
    def compute_loss():
        p = net.forward(X)
        # cross-entropy
        p_clipped = np.clip(p, 1e-12, 1.0)
        return -np.mean(np.sum(y * np.log(p_clipped), axis=1))

    for i in range(check_rows):
        for j in range(check_cols):
            first_layer.W[i, j] += epsilon
            lp = compute_loss()
            first_layer.W[i, j] -= 2 * epsilon
            lm = compute_loss()
            first_layer.W[i, j] += epsilon   # restore
            numerical_gradW[i, j] = (lp - lm) / (2 * epsilon)

    diff = np.max(np.abs(analytical_gradW[:check_rows, :check_cols] - numerical_gradW))
    print(f"  Max gradient difference: {diff:.2e}")

    if diff < 1e-7:
        record("T4 Gradient check (diff < 1e-7 → full marks)", 5, True)
    elif diff < 1e-5:
        # partial credit
        score += 3; passed.append("T4-partial")
        warn(f"[+3] T4 Gradient check PARTIAL (diff={diff:.2e}, need < 1e-7 for full)")
    else:
        record("T4 Gradient check (diff < 1e-7 → full marks)", 5, False,
               f"diff={diff:.2e} exceeds 1e-5 threshold")
except Exception as e:
    record("T4 Gradient check (diff < 1e-7 → full marks)", 5, False,
           traceback.format_exc(limit=3))


print(f"\n{BOLD}─── Gradient Consistency Score: {score} / {max_score} ───{RESET}")
if failed_tests:
    print(f"{RED}Failed tests: {', '.join(failed_tests)}{RESET}")
print()
