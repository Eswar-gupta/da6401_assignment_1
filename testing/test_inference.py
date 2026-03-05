"""
test_inference.py
=================
DA6401 Assignment-1 — Autograder: Inference & Private Test Performance (10 Marks)

Tests:
  T1  models/bestmodel.npy exists                                (1 mark)
  T2  models/bestconfig.json exists and is valid JSON            (1 mark)
  T3  bestconfig.json has all required keys                      (2 marks)
  T4  inference.py runs without error using bestconfig values    (2 marks)
  T5  Output contains Accuracy, Precision, Recall, F1-score      (4 marks)

Usage (run from the CLONED STUDENT REPO ROOT):
    python testing/test_inference.py
"""

import sys
import os
import json
import subprocess
import traceback

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH   = os.path.join(REPO_ROOT, "src")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
INFER_PY   = os.path.join(SRC_PATH, "inference.py")
MODEL_NPY  = os.path.join(MODELS_DIR, "bestmodel.npy")
CONFIG_JSON= os.path.join(MODELS_DIR, "bestconfig.json")
PYTHON     = sys.executable

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
RESET  = "\033[0m";  BOLD = "\033[1m"

def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}INFO{RESET}  {msg}")

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


print(f"\n{BOLD}=== Inference & Model Submission Check ==={RESET}\n")

# ── T1: bestmodel.npy exists ──────────────────────────────────────────────────
exists_npy = os.path.isfile(MODEL_NPY)
record("T1 models/bestmodel.npy exists", 1, exists_npy,
       f"not found at {MODEL_NPY}")

# ── T2: bestconfig.json valid ─────────────────────────────────────────────────
cfg = {}
exists_json = os.path.isfile(CONFIG_JSON)
if exists_json:
    try:
        with open(CONFIG_JSON) as f:
            cfg = json.load(f)
        record("T2 models/bestconfig.json is valid JSON", 1, True)
        warn(f"bestconfig.json contents:\n{json.dumps(cfg, indent=2)}")
    except json.JSONDecodeError as e:
        record("T2 models/bestconfig.json is valid JSON", 1, False,
               f"JSON parse error: {e}")
else:
    record("T2 models/bestconfig.json is valid JSON", 1, False,
           f"not found at {CONFIG_JSON}")

# ── T3: required keys in bestconfig.json ──────────────────────────────────────
REQUIRED_KEYS = {
    "optimizer", "activation", "loss",
    "numlayers", "hiddensize", "learningrate"
}
if cfg:
    missing_keys = REQUIRED_KEYS - set(str(k).lower() for k in cfg.keys())
    record("T3 bestconfig.json has required keys", 2,
           len(missing_keys) == 0,
           f"missing keys: {missing_keys}")
else:
    record("T3 bestconfig.json has required keys", 2, False,
           "config file not loaded")

# ── T4 & T5: run inference.py ─────────────────────────────────────────────────
if not exists_npy:
    record("T4 inference.py runs without error", 2, False,
           "bestmodel.npy missing, cannot run inference")
    record("T5 Output has Accuracy/Precision/Recall/F1", 4, False,
           "inference could not be run")
else:
    # Build inference command from bestconfig values (with safe defaults)
    def cfg_get(key, default):
        for k, v in cfg.items():
            if k.lower() == key.lower():
                return v
        return default

    nhl  = str(cfg_get("numlayers",   3))
    sz   = str(cfg_get("hiddensize",  128))
    act  = str(cfg_get("activation",  "relu"))
    ds   = str(cfg_get("dataset",     "mnist"))
    bs   = str(cfg_get("batchsize",   64))

    # relative path to model from src/
    rel_model = os.path.relpath(MODEL_NPY, SRC_PATH)

    cmd = [
        PYTHON, INFER_PY,
        "--model_path", rel_model,
        "-d", ds,
        "-b", bs,
        "-nhl", nhl,
        "-sz", sz,
        "-a", act,
    ]
    warn(f"Running: {' '.join(cmd[2:])}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=SRC_PATH
        )
        rc  = result.returncode
        out = result.stdout + result.stderr
        record("T4 inference.py runs without error", 2, rc == 0,
               f"exit code {rc}\n{out[-600:]}" if rc != 0 else "")

        # T5: check metric keywords in output
        out_lower = out.lower()
        metrics = {
            "accuracy":  any(k in out_lower for k in ("accuracy", "acc")),
            "precision": "precision" in out_lower,
            "recall":    "recall"    in out_lower,
            "f1":        any(k in out_lower for k in ("f1", "f1-score", "f1_score")),
        }
        print(f"\n  Metrics found in output:")
        for m, found in metrics.items():
            status = f"{GREEN}YES{RESET}" if found else f"{RED}NO {RESET}"
            print(f"         {m:<12} {status}")

        marks_t5 = sum(metrics.values())   # 1 mark per metric
        if marks_t5 == 4:
            record("T5 All 4 metrics printed (Accuracy/Precision/Recall/F1)", 4, True)
        else:
            score += marks_t5
            warn(f"[+{marks_t5}] T5 partial — {marks_t5}/4 metrics found in stdout")

        # Also print the raw stdout for TA to read F1 score
        if out.strip():
            print(f"\n  --- inference.py output ---\n{out.strip()}\n  ---")

    except subprocess.TimeoutExpired:
        record("T4 inference.py runs without error", 2, False, "TIMEOUT (>300s)")
        record("T5 All 4 metrics printed", 4, False, "inference timed out")


print(f"\n{BOLD}─── Inference Score: {score} / {max_score} ───{RESET}")
if failed_tests:
    print(f"{RED}Failed tests: {', '.join(failed_tests)}{RESET}")
print()
