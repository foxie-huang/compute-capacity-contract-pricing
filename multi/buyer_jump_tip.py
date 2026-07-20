#!/usr/bin/env python3
r"""
Discrete buyer commitments as compound-Poisson share jumps, and their effect on the
traded-tenor tip probabilities.

WHY.  Assumption ass:arch(ii) makes the share flow deterministic conditional on the
frontier, justified by "aggregate reallocation by many buyers facing the same merits."  In
THIS market reallocation is a handful of gigawatt-scale decisions -- OpenAI's 6 GW to AMD,
10 GW to Broadcom, TPU inference; Anthropic's 5 GW Trainium and 5 GW TPU (sec:circular,
app:clusters).  Those are discrete idiosyncratic share jumps, not an average.  This script
prices that channel instead of assuming it away.

THE MODEL.  Under the two-family reduction eq:two-regime with rho = 0,

    ds = s(1-s) [ 2*gamma*s - gamma + m(t) ] dt  -  Delta_j dN^buy,
    dm = -mdot dt - sigma_m dW,          barrier   s_dag(t) = (gamma - m(t)) / (2*gamma).

The tip is the first passage  s(t) <= s_dag(t).  Note this UNIFIES the paper's two routes:
with no jumps s stays pinned at the vertex s = 1, so the crossing happens exactly when
s_dag = 1, i.e. when m hits -gamma -- which is Lemma lem:tip-law's Inverse-Gaussian first
passage.  The IG law is therefore the zero-jump limit of this simulation, and reproducing
it is the validation below.  Buyer jumps can only bring the crossing forward.

CONDITIONAL AFFINITY SURVIVES.  N^buy has deterministic intensity and marks independent of
the credit state, so conditional on the frontier AND the buyer path the share path is still
deterministic: N^buy joins the state as a second Cox coordinate exactly as N^tip does in
prop:cond-affine(i)-(ii), with deterministic post-jump paths.

CALIBRATION.  arch: m0 = 0.24/yr, gamma = 0.50, mdot = 0.04/yr^2, sigma_m = 0.151, from
IG(mean 18.5y, shape 24) in sec:bifurcation.  Buyer record 2025-26: five gigawatt-scale
commitments over ~1.5y => nu_buy = 3/yr, marks resampled from the observed {6,10,5,5} GW.
Contestable merchant pool = OpenAI's 10 GW NVIDIA + 16 GW non-NVIDIA = 26 GW (sec:circular).

THE KEY PARAMETER is phi, the TRAINING-scoped fraction of a commitment.  The paper's own
read (sec:circular, app:calibration) is that the record is "inference-scoped to date", i.e.
phi ~ 0, so it moves the overall mix rather than the contestable training basin.  phi is
swept here to find what it would take to matter.  Share jump  Delta = phi * GW / 26.

Deterministic: fixed seeds.
"""
import time

import numpy as np
from scipy.stats import invgauss

t_wall = time.time()

GAMMA, M0, MDOT = 0.50, 0.24, 0.04
SIGMA_M = np.sqrt((M0 + GAMMA) ** 2 / 24.0)      # shape 24 => sigma_m
NU_BUY = 3.0                                      # commitments/yr, 2025-26 record
MARKS_GW = np.array([6.0, 10.0, 5.0, 5.0])        # AMD, Broadcom, Trainium, TPU
POOL_GW = 26.0                                    # 10 GW NVIDIA + 16 GW non-NVIDIA
T_MAX, DT = 5.0, 1.0 / 1000.0
NPATH = 100_000
TENORS = [1, 2, 3, 4, 5]


def tip_probs(phi, nu=NU_BUY, npath=NPATH, seed=0):
    """P(tip <= T) at each tenor, plus the mean pre-tip share at T_MAX (the mechanism)."""
    rng = np.random.default_rng(seed)
    nstep = int(round(T_MAX / DT))
    s = np.ones(npath)
    m = np.full(npath, M0)
    tipped_at = np.full(npath, np.inf)
    sq = SIGMA_M * np.sqrt(DT)
    p_jump = nu * DT                              # thinned Poisson, p_jump << 1
    for k in range(nstep):
        t = (k + 1) * DT
        m -= MDOT * DT + sq * rng.standard_normal(npath)
        live = np.isinf(tipped_at)
        # replicator drift back toward the vertex (deterministic given m)
        s[live] += s[live] * (1 - s[live]) * (2 * GAMMA * s[live] - GAMMA + m[live]) * DT
        if phi > 0:
            hit = live & (rng.random(npath) < p_jump)
            if hit.any():
                gw = rng.choice(MARKS_GW, hit.sum())
                s[hit] -= phi * gw / POOL_GW
        s_dag = (GAMMA - m) / (2 * GAMMA)
        crossed = live & (s <= s_dag)
        tipped_at[crossed] = t
    surv = np.isinf(tipped_at)
    return [np.mean(tipped_at <= T) for T in TENORS], s[surv].mean()


def realloc_rate(phi, nu=NU_BUY):
    """Annual contestable-training share reallocation, in share points per year."""
    return nu * phi * MARKS_GW.mean() / POOL_GW * 100


# ------------------------------------------------------------------ validation
# zero-jump arm must reproduce the paper's IG(mean 18.5y, shape 24) first passage
mu_ig, lam_ig = (M0 + GAMMA) / MDOT, (M0 + GAMMA) ** 2 / SIGMA_M ** 2
closed = [invgauss.cdf(T, mu_ig / lam_ig, scale=lam_ig) for T in TENORS]
sim0, _ = tip_probs(0.0, seed=1)
print("validation: zero-jump arm vs the closed-form Inverse-Gaussian of lem:tip-law")
print(f"   IG(mean {mu_ig:.1f}y, shape {lam_ig:.0f});  paper reports "
      f"0.0, 0.2, 1.5, 4.5, 8.9 %\n")
print(f"{'T':>3} {'closed form':>12} {'simulated':>10}")
for T, c, s_ in zip(TENORS, closed, sim0):
    print(f"{T:>3} {c*100:11.2f}% {s_*100:9.2f}%")

# ------------------------------------------------------------------ the sweep
print(f"\nfive-year tip probability with compound-Poisson buyer commitments"
      f"   [nu_buy = {NU_BUY}/yr, marks {MARKS_GW} GW, pool {POOL_GW:.0f} GW, "
      f"{NPATH:,} paths]")
print(f"{'phi':>6} {'pp/yr':>7} " +
      " ".join(f"{'T='+str(T):>8}" for T in TENORS) + f" {'mean s|no tip':>14}")
for j, phi in enumerate([0.0, 0.05, 0.10, 0.25, 0.50, 1.00]):
    pr, sbar = tip_probs(phi, seed=10 + j)
    print(f"{phi:6.2f} {realloc_rate(phi):6.1f}  " +
          " ".join(f"{p*100:7.2f}%" for p in pr) + f" {sbar:13.3f}")

# is the (nu, phi) split sufficient, or only their product?  same reallocation rate,
# different granularity: many small commitments vs few large ones.
print(f"\nis only the PRODUCT nu*phi identified?  matched reallocation rate {realloc_rate(0.10):.1f} pp/yr")
print(f"{'nu':>6} {'phi':>6} {'pp/yr':>7} " + " ".join(f"{'T='+str(T):>8}" for T in TENORS))
for j, (nu, phi) in enumerate([(1.0, 0.30), (3.0, 0.10), (9.0, 0.0333)]):
    pr, _ = tip_probs(phi, nu=nu, seed=40 + j)
    print(f"{nu:6.1f} {phi:6.3f} {realloc_rate(phi, nu):6.1f}  " +
          " ".join(f"{p*100:7.2f}%" for p in pr))

print(f"\nreading: phi is the fraction of a committed gigawatt that lands in the")
print("CONTESTABLE TRAINING basin.  phi = 0 is the paper's read of the record as")
print("inference-scoped, and reproduces the unchanged IG law.  The column to watch is")
print("T = 5, against the paper's 8.9%.")
print("\nMECHANISM: read the last column.  Jumps do not tip the market directly -- a single")
print("commitment never spans the 74-point gap from the vertex to the barrier.  They hold")
print("the share OFF the vertex at a quasi-stationary level where replicator recovery")
print("balances arrival, so the rising barrier has less distance to climb.  The tip channel")
print("is a ratchet, not a shock.")
print(f"\n[wall {time.time()-t_wall:.1f}s]")
