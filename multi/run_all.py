"""
run_all.py — one-command reproduction of the multi-issuer paper's computed exhibits.

Runs every script, echoes its output, and checks the published headline numbers against
what the run actually produced. Exits non-zero if any gate fails or any script errors.
Gates live here rather than inside the scripts so that the scripts stay identical to what
the paper describes and the assertions are auditable in one place.

Scripts run with this folder as the working directory, so any figures they emit land
here rather than in the repository root. Every script prints its own total wall-clock. Longest is buyer_jump_tip.py (~70 s); total
run is a little over two minutes. Dependencies: numpy, scipy, matplotlib.
"""
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent


def near(got, want, tol):
    return got is not None and abs(got - want) <= tol


def grab(pattern, text, group=1):
    m = re.search(pattern, text, re.M)
    return float(m.group(group)) if m else None


# --- gates: each takes the script's stdout and returns (ok, message) -------------
def gate_finite_cascade(o):
    err = grab(r"rel\. err\s+([\d.]+)%", o)
    mult = grab(r"^\s*0\.64\s+[\d.]+\s+[\d.]+\s+([\d.]+)", o)
    checks = [
        (near(err, 0.21, 0.15), f"n=0 closed-form validation {err}% (paper 0.2%)"),
        (near(mult, 1.09, 0.06), f"operative multiplier at n=0.64 {mult} (paper ~1.1)"),
    ]
    return checks


def gate_linearisation(o):
    ends = grab(r"^\s*0\.00\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+x\s+([\d.]+)x", o)
    peak = grab(r"^\s*0\.35\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+x\s+([\d.]+)x", o)
    return [
        (near(ends, 1.03, 0.02), f"error at gamma_fs=0 is {ends}x (paper 1.03x)"),
        (near(peak, 1.44, 0.05), f"peak error {peak}x (paper 1.44x, i.e. 20% of half-width)"),
    ]


def gate_buyer_jump(o):
    base = grab(r"^\s*0\.00\s+[\d.]+\s+(?:[\d.]+%\s+){4}([\d.]+)%", o)
    lift = grab(r"^\s*0\.05\s+[\d.]+\s+(?:[\d.]+%\s+){4}([\d.]+)%", o)
    return [
        (near(base, 8.85, 0.5), f"five-year tip at phi=0 is {base}% (paper 8.9%)"),
        (near(lift, 15.46, 0.8), f"five-year tip at phi=0.05 is {lift}% (paper 15.5%)"),
    ]


def gate_tip_identification(o):
    coef = grab(r"coefficient:\s+([\d.]+)pp of face per unit", o)
    return [(near(coef, 9.3, 0.1), f"recovery-leg coefficient {coef}pp per unit p (paper 9.3pp)")]


JOBS = [
    ("finite_cascade.py", "exact finite-name cascade: envelope vs operative multiplier",
     gate_finite_cascade),
    ("linearisation_error.py", "linearisation error in the cluster-basis floor",
     gate_linearisation),
    ("buyer_jump_tip.py", "compound-Poisson buyer commitments and tip probabilities",
     gate_buyer_jump),
    ("tip_identification.py", "is the tip identifiable from traded prices?",
     gate_tip_identification),
    ("arch_bifurcation.py", "architecture bifurcation and first-passage tip law", None),
    ("arch_overall_layer.py", "overall-mix coexistence layer and the buffer walk", None),
    ("arch_simplex.py", "landscape geometry on the family simplex", None),
    ("n_structural_gao.py", "structural branching ratio from the exposure network", None),
    ("fig_nS_structural.py", "structural-prior figure", None),
    ("cohort_sensitivity.py", "2022 cohort sensitivity and the reattribution row", None),
]

t0 = time.time()
failed = []
for rel, desc, gate in JOBS:
    print(f"\n{'=' * 88}\n>> {rel} — {desc}\n{'=' * 88}")
    r = subprocess.run([sys.executable, str(HERE / rel)], capture_output=True, text=True,
                       cwd=HERE)
    sys.stdout.write(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)
    if r.returncode != 0:
        failed.append(f"{rel} (exit {r.returncode})")
        continue
    if gate:
        for ok, msg in gate(r.stdout):
            print(f"   [{'PASS' if ok else 'FAIL'}] {msg}")
            if not ok:
                failed.append(f"{rel}: {msg}")

print(f"\n{'=' * 88}")
if failed:
    print("RESULT: FAIL")
    for f in failed:
        print(f"   - {f}")
    sys.exit(1)
print(f"RESULT: all gates PASS   (total wall-clock {time.time() - t0:.1f}s)")
