"""
run_all.py — one-command reproduction of the single-issuer paper's computed exhibits.

Runs the two gated pipelines and the two supporting calibrations, echoes their output,
and exits non-zero if any internal gate reports FAIL. Every script prints its own
per-stage and total wall-clock. Dependencies: numpy (scipy optional, erf fallback).
"""
import subprocess, sys, time, pathlib

HERE = pathlib.Path(__file__).parent
JOBS = [
    ("clock_curve.py",                    "commodity branch: reference curve, gate G^g = $1.63"),
    ("cohort_pipeline.py",                "credit branch: V0 gate (12/12 to the cent) + fidelity variants"),
    ("scripts/cohort_rerun_closedform.py","executory-recovery cohort re-pricing, closed form"),
    ("scripts/calibrate_gamma.py",        "surprise loading c'_g = 0.148 +/- 0.021, three routes"),
]

t0 = time.time()
failed = []
for rel, desc in JOBS:
    print(f"\n{'='*88}\n>> {rel} — {desc}\n{'='*88}")
    r = subprocess.run([sys.executable, str(HERE / rel)], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)
    if r.returncode != 0 or "FAIL" in r.stdout:
        failed.append(rel)

print(f"\n{'='*88}")
if failed:
    print(f"RESULT: FAIL — {failed}")
    sys.exit(1)
print(f"RESULT: all gates PASS   (total wall-clock {time.time()-t0:.2f}s)")
