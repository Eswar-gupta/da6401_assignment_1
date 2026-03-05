"""
run_all_tests.py
================
DA6401 Assignment-1 — Master Autograder (50 Implementation Marks)

Runs all four test modules in sequence and prints a unified grade report.

Usage (run from the CLONED STUDENT REPO ROOT):
    python testing/run_all_tests.py

To test a specific section only:
    python testing/test_forward_pass.py
    python testing/test_gradients.py
    python testing/test_training.py
    python testing/test_inference.py
"""

import sys
import os
import subprocess
import re
import time

TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON      = sys.executable

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; RESET = "\033[0m"; BOLD = "\033[1m"

TESTS = [
    {
        "file":    "test_forward_pass.py",
        "label":   "Forward Pass Verification",
        "max":     10,
        "pattern": r"Forward Pass Score:\s*(\d+)\s*/\s*10",
    },
    {
        "file":    "test_gradients.py",
        "label":   "Gradient Consistency",
        "max":     10,
        "pattern": r"Gradient Consistency Score:\s*(\d+)\s*/\s*10",
    },
    {
        "file":    "test_training.py",
        "label":   "Training Functionality",
        "max":     15,
        "pattern": r"Training Functionality Score:\s*(\d+)\s*/\s*15",
    },
    {
        "file":    "test_inference.py",
        "label":   "Inference & Private Test",
        "max":     10,
        "pattern": r"Inference Score:\s*(\d+)\s*/\s*10",
    },
]

# ── banner ─────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
print(f"{BOLD}{CYAN}  DA6401 Assignment-1 — Implementation Autograder{RESET}")
print(f"{BOLD}{CYAN}  45 Marks (Code Quality +5 is manual){RESET}")
print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")

total_score = 0
total_max   = sum(t["max"] for t in TESTS)
results     = []

for t in TESTS:
    script = os.path.join(TESTING_DIR, t["file"])
    print(f"{BOLD}{CYAN}── Running: {t['label']} ──{RESET}")
    print(f"   {script}\n")

    start = time.time()
    try:
        result = subprocess.run(
            [PYTHON, script],
            capture_output=False,     # stream output live to terminal
            text=True,
            timeout=600,
        )
        elapsed = time.time() - start

        # Re-run with captured output to parse score
        result2 = subprocess.run(
            [PYTHON, script],
            capture_output=True,
            text=True,
            timeout=600,
        )
        combined_out = result2.stdout + result2.stderr

        # Parse score from output
        m = re.search(t["pattern"], combined_out)
        if m:
            s = int(m.group(1))
        else:
            # fallback: count PASS lines
            pass_count = combined_out.count("PASS")
            s = min(pass_count, t["max"])

    except subprocess.TimeoutExpired:
        elapsed = 600
        s = 0
        print(f"  {RED}TIMEOUT — test script exceeded 10 minutes{RESET}\n")
    except Exception as e:
        elapsed = 0
        s = 0
        print(f"  {RED}ERROR running script: {e}{RESET}\n")

    total_score += s
    results.append((t["label"], s, t["max"]))
    colour = GREEN if s == t["max"] else (YELLOW if s > 0 else RED)
    print(f"\n  {colour}Score: {s} / {t['max']}{RESET}   (took {elapsed:.1f}s)\n")

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  GRADE REPORT — Implementation (45 auto + 5 manual){RESET}")
print(f"{BOLD}{'='*60}{RESET}")
print(f"  {'Criterion':<35} {'Score':>6}  {'Max':>4}")
print(f"  {'-'*35} {'------':>6}  {'----':>4}")

for label, s, mx in results:
    colour = GREEN if s == mx else (YELLOW if s > 0 else RED)
    print(f"  {label:<35} {colour}{s:>6}{RESET}  {mx:>4}")

print(f"  {'─'*35} {'──────':>6}  {'────':>4}")
pct = (total_score / total_max * 100) if total_max else 0
colour = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)
print(f"  {'AUTO-GRADED TOTAL':<35} {colour}{total_score:>6}{RESET}  {total_max:>4}")
print(f"  {'Code Quality (manual)':<35} {'?':>6}  {5:>4}")
print(f"  {'W&B Report (manual)':<35} {'?':>6}  {50:>4}")
print(f"\n  {BOLD}Implementation auto-score: {colour}{total_score}/{total_max}{RESET}{BOLD} ({pct:.1f}%){RESET}")
print(f"{BOLD}{'='*60}{RESET}\n")

# ── TA notes ──────────────────────────────────────────────────────────────────
print(f"{YELLOW}  TA Notes:{RESET}")
print(f"  • Code Quality (5 marks) → grade manually: README, comments, style")
print(f"  • W&B Report (50 marks)  → grade manually using PART 17 of how_to_test.md")
print(f"  • For private test F1-score rubric, see PART 16 of how_to_test.md")
print()
