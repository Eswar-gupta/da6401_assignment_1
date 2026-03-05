"""
test_training.py
================
DA6401 Assignment-1 — Autograder: Training Functionality (15 Marks)

Tests:
  T1  train.py --help shows all 11 mandatory CLI args            (3 marks)
  T2  inference.py --help shows all 6 mandatory CLI args         (1 mark)
  T3  Smoke test: train.py runs 1 epoch without crashing         (2 marks)
  T4  All 6 optimizers accepted without crash                    (3 marks)
  T5  Both loss functions accepted without crash                 (2 marks)
  T6  All 3 activation functions accepted without crash          (2 marks)
  T7  Both weight init methods accepted without crash            (1 mark)
  T8  fashionmnist dataset flag accepted without crash           (1 mark)

Usage (run from the CLONED STUDENT REPO ROOT):
    python testing/test_training.py
"""

import sys
import os
import subprocess
import traceback

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH    = os.path.join(REPO_ROOT, "src")
TRAIN_PY    = os.path.join(SRC_PATH, "train.py")
INFER_PY    = os.path.join(SRC_PATH, "inference.py")
PYTHON      = sys.executable

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
RESET  = "\033[0m";  BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}WARN{RESET}  {msg}")

score, max_score = 0, 15
passed, failed_tests = [], []

def record(name, marks, ok_flag, reason=""):
    global score
    if ok_flag:
        score += marks; passed.append(name)
        ok(f"[+{marks}] {name}")
    else:
        failed_tests.append(name)
        fail(f"[ 0] {name}  ← {reason}")


def run_script(args, timeout=120):
    """Run a Python script with args. Returns (returncode, stdout+stderr)."""
    cmd = [PYTHON] + args
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=SRC_PATH
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -2, str(e)


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}=== Training Functionality Check ==={RESET}")

# ── T1: train.py --help has all required args ─────────────────────────────────
REQUIRED_TRAIN_FLAGS = {
    "-d": "--dataset",
    "-e": "--epochs",
    "-b": "--batchsize",
    "-l": "--loss",
    "-o": "--optimizer",
    "-lr": "--learningrate",
    "-wd": "--weightdecay",
    "-nhl": "--numlayers",
    "-sz": "--hiddensize",
    "-a": "--activation",
    "-wi": "--weightinit",
}

print(f"\n  Checking train.py --help ...")
rc, out = run_script([TRAIN_PY, "--help"])
missing = []
for short, long in REQUIRED_TRAIN_FLAGS.items():
    found = (short in out) or (long in out)
    if not found:
        missing.append(f"{short}/{long}")
        warn(f"    Missing flag: {short} / {long}")
    else:
        print(f"         found: {short} / {long}")

marks_t1 = 3 if not missing else (2 if len(missing) <= 2 else (1 if len(missing) <= 5 else 0))
record(f"T1 train.py exposes all 11 mandatory CLI args", 3,
       len(missing) == 0,
       f"missing: {missing}" if missing else "")
if missing and marks_t1 > 0:
    score += marks_t1    # partial
    warn(f"[+{marks_t1}] T1 partial — {len(missing)} flags missing")

# ── T2: inference.py --help has required args ─────────────────────────────────
REQUIRED_INFER_FLAGS = {
    "--model_path": "--model_path",
    "-d": "--dataset",
    "-b": "--batchsize",
    "-nhl": "--numlayers",
    "-sz": "--hiddensize",
    "-a": "--activation",
}

print(f"\n  Checking inference.py --help ...")
rc, out = run_script([INFER_PY, "--help"])
missing_i = []
for short, long in REQUIRED_INFER_FLAGS.items():
    found = (short in out) or (long in out)
    if not found:
        missing_i.append(long)
        warn(f"    Missing flag: {long}")
    else:
        print(f"         found: {long}")

record("T2 inference.py exposes required CLI args", 1,
       len(missing_i) == 0,
       f"missing: {missing_i}" if missing_i else "")

# ── T3: smoke test — 1 epoch training ─────────────────────────────────────────
BASE_CMD = [
    TRAIN_PY,
    "-d", "mnist",
    "-e", "1",
    "-b", "256",
    "-l", "crossentropy",
    "-o", "adam",
    "-lr", "0.001",
    "-wd", "0",
    "-nhl", "2",
    "-sz", "64",
    "-a", "relu",
    "-wi", "xavier",
]

print(f"\n  Running smoke test (1 epoch, mnist, adam, crossentropy) ...")
rc, out = run_script(BASE_CMD, timeout=300)
if rc == 0:
    record("T3 train.py completes 1 epoch without error", 2, True)
else:
    print(f"  STDERR/STDOUT tail:\n{out[-800:]}")
    record("T3 train.py completes 1 epoch without error", 2, False,
           f"exit code {rc}")

# ── T4: all 6 optimizers ──────────────────────────────────────────────────────
OPTIMIZERS = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]
print(f"\n  Testing all 6 optimizers (1 epoch each) ...")
opt_results = {}

for opt in OPTIMIZERS:
    cmd = BASE_CMD.copy()
    cmd[cmd.index("-o") + 1] = opt
    rc, out = run_script(cmd, timeout=300)
    opt_results[opt] = (rc == 0)
    status = f"{GREEN}OK{RESET}" if rc == 0 else f"{RED}FAIL (exit {rc}){RESET}"
    print(f"         {opt:<12} {status}")

passing_opts = sum(opt_results.values())
marks_t4 = 3 if passing_opts == 6 else (2 if passing_opts >= 4 else (1 if passing_opts >= 2 else 0))
if passing_opts == 6:
    record("T4 All 6 optimizers work", 3, True)
else:
    # partial
    score += marks_t4
    warn(f"[+{marks_t4}] T4 partial — {passing_opts}/6 optimizers passed: "
         f"{[k for k,v in opt_results.items() if not v]} failed")

# ── T5: both loss functions ───────────────────────────────────────────────────
LOSSES = ["crossentropy", "meansquarederror"]
print(f"\n  Testing both loss functions ...")
loss_results = {}

for loss in LOSSES:
    cmd = BASE_CMD.copy()
    cmd[cmd.index("-l") + 1] = loss
    rc, out = run_script(cmd, timeout=300)
    loss_results[loss] = (rc == 0)
    status = f"{GREEN}OK{RESET}" if rc == 0 else f"{RED}FAIL{RESET}"
    print(f"         {loss:<20} {status}")

record("T5 Both loss functions work", 2, all(loss_results.values()),
       f"failed: {[k for k,v in loss_results.items() if not v]}")

# ── T6: all 3 activations ─────────────────────────────────────────────────────
ACTIVATIONS = ["relu", "sigmoid", "tanh"]
print(f"\n  Testing all 3 activation functions ...")
act_results = {}

for act in ACTIVATIONS:
    cmd = BASE_CMD.copy()
    cmd[cmd.index("-a") + 1] = act
    rc, out = run_script(cmd, timeout=300)
    act_results[act] = (rc == 0)
    status = f"{GREEN}OK{RESET}" if rc == 0 else f"{RED}FAIL{RESET}"
    print(f"         {act:<12} {status}")

record("T6 All 3 activation functions work", 2, all(act_results.values()),
       f"failed: {[k for k,v in act_results.items() if not v]}")

# ── T7: both weight init methods ──────────────────────────────────────────────
INITS = ["xavier", "random"]
print(f"\n  Testing both weight init methods ...")
init_results = {}

for wi in INITS:
    cmd = BASE_CMD.copy()
    cmd[cmd.index("-wi") + 1] = wi
    rc, out = run_script(cmd, timeout=300)
    init_results[wi] = (rc == 0)
    status = f"{GREEN}OK{RESET}" if rc == 0 else f"{RED}FAIL{RESET}"
    print(f"         {wi:<12} {status}")

record("T7 Both weight init methods work", 1, all(init_results.values()),
       f"failed: {[k for k,v in init_results.items() if not v]}")

# ── T8: fashionmnist dataset ──────────────────────────────────────────────────
print(f"\n  Testing fashionmnist dataset flag ...")
cmd = BASE_CMD.copy()
cmd[cmd.index("-d") + 1] = "fashionmnist"
rc, out = run_script(cmd, timeout=300)
record("T8 fashionmnist dataset accepted", 1, rc == 0,
       f"exit code {rc}" if rc != 0 else "")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}─── Training Functionality Score: {min(score, max_score)} / {max_score} ───{RESET}")
if failed_tests:
    print(f"{RED}Failed tests: {', '.join(failed_tests)}{RESET}")
print()
