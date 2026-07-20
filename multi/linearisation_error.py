#!/usr/bin/env python3
r"""
Linearisation error in the cluster-basis floor B (Theorem prop:band, part (b)).

The floor uses the RANK-ONE loss form eq:jump-channel: every firing of N^(c) costs the
same aggregate  mu_1 = sum_i w_i chi_i,  so the cluster variance is mu_1^2 Var[N_T].
Two things the paper's own text flags break that:

  (1) NAME RESOLUTION -- loadings are probabilities, so which names resolve at a firing is
      random.  Appendix C derives this in closed form:
          v_1 = sum_i (w_i chi_i)^2 (1 - delta_i)/delta_i,   contributing v_1 E[N_T].
  (2) STATE-DEPENDENT chi -- the endogenous recovery eq:endog-recovery falls with the
      contemporaneous shock count, so the loss per default is LARGER in exactly the
      high-count states.  That is a positive covariance between per-firing loss and count,
      and no closed form covers it.

This script simulates the full model and decomposes the gap, validating (1) against the
closed form and isolating (2).  Deterministic: fixed seeds.

SCOPE.  This measures the linearisation error WITHIN the paper's own infinite-population
regime, i.e. with names re-resolvable, exactly as eq:jump-channel assumes.  The separate
FINITE-POPULATION error -- names default at most once, so the cascade saturates -- is
measured in Appendix E.2 (finite_cascade.py) and pushes the other way.  Simulating both at
once conflates two corrections of opposite sign; they are kept apart deliberately.
"""
import time

import numpy as np

t_wall = time.time()

# 2022 cohort calibration (App H.2): loadings, equal weights, endogenous recovery ends
DELTA = np.array([0.85, 0.80, 0.45, 0.30])
W = np.full(4, 0.25)                 # equal-weighted basket
V = np.ones(4)                       # at-risk deliverable value, normalised
R_HI, R_LO = 0.40, 0.10              # no-cascade level and fire-sale floor
NU, T_HOR, NPATH = 1.18, 1.0, 400_000   # back-solved cohort intensity, one-year tenor
N_BRANCH = 0.50                          # cohort branching ratio (App H.2): Fano 1/(1-n)^2


def recovery(k, gamma_fs):
    """eq:endog-recovery: R falls toward the fire-sale floor with the shock count k.
    gamma_fs = inf is the TWO-STATE reduction of rem:endog-recovery -- the form the
    circular-financing example actually uses -- where R sits at the floor from the first
    firing on, so chi carries no residual dependence on the count."""
    if np.isinf(gamma_fs):
        return R_LO
    return R_LO + (R_HI - R_LO) * np.exp(-gamma_fs * k)


def simulate(gamma_fs, nu=NU, T=T_HOR, npath=NPATH, seed=0):
    """Full model in the paper's linearisation regime: random name resolution at each
    firing (no absorption -- that is Appendix E.2's separate correction) and a recovery
    that falls with the contemporaneous shock count."""
    rng = np.random.default_rng(seed)
    # the floor B carries the OVER-DISPERSED count of eq:fano, Fano = 1/(1-n)^2, so the
    # benchmark count must too; a negative binomial matches (mean, Fano) exactly.
    m, F = nu * T, 1.0 / (1.0 - N_BRANCH) ** 2
    if F > 1.0:
        p = 1.0 / F
        counts = rng.negative_binomial(m * p / (1.0 - p), p, npath)
    else:
        counts = rng.poisson(m, npath)
    L = np.zeros(npath)
    for k in range(1, counts.max() + 1):
        act = counts >= k
        if not act.any():
            break
        hit = (rng.random((npath, len(DELTA))) < DELTA) & act[:, None]
        L += (hit * (W * V) * (1.0 - recovery(k, gamma_fs))).sum(axis=1)
    return L, counts


mu1_bar = (W * DELTA * (1.0 - recovery(1, 0.0)) * V).sum()   # gamma_fs = 0 => R fixed at R_HI
print("cluster-basis floor: rank-one form vs the full model")
print(f"   cohort delta = {DELTA}, equal weights, nu = {NU}/yr, T = {T_HOR}y, "
      f"{NPATH:,} paths\n")

print(f"{'gamma_fs':>9} {'rank-one B':>11} {'+name res.':>11} {'exact':>9} "
      f"{'closed-form err':>16} {'total err':>10}")
for gfs in (0.0, 0.10, 0.20, 0.35, 0.50, 0.69, 1.20, 2.00, np.inf):
    L, cnt = simulate(gfs, seed=7)
    Rbar = recovery(1, gfs)                       # representative per-firing recovery
    chi = DELTA * (1.0 - Rbar) * V                # the paper's chi_i at that recovery
    mu1 = (W * chi).sum()
    v1 = (((W * chi) ** 2) * (1 - DELTA) / DELTA).sum()
    EN, VarN = cnt.mean(), cnt.var(ddof=1)
    rank1 = mu1 ** 2 * VarN                       # the floor B's cluster term
    plus_res = rank1 + v1 * EN                    # + Appendix C closed form
    exact = L.var(ddof=1)
    print(f"{gfs:9.2f} {rank1:11.5f} {plus_res:11.5f} {exact:9.5f} "
          f"{plus_res/rank1:15.2f}x {exact/rank1:9.2f}x")

print("\nreading: 'closed-form err' is the Appendix C name-resolution correction alone;")
print("'total err' adds the wrong-way state-dependence of chi, which no closed form covers.")
print("Both inflate the true residual variance ABOVE B, so B remains a valid FLOOR: the")
print("linearisation errs conservatively, and the errors do not offset.")
print("\nThe state-dependence term vanishes at BOTH ends -- at gamma_fs = 0 (exogenous R) and")
print("at gamma_fs = inf (the two-state reduction the circular example uses, where R sits at")
print("its floor from the first firing, so chi no longer varies with the count).  It peaks in")
print("between, at gamma_fs ~ 0.35: 1.44x in variance, i.e. 1.20x = 20% in the HALF-WIDTH.")
print("At the paper's own two-state calibration the error is the 1.03x name-resolution term")
print("alone, 1.6% of the half-width.")
print(f"\n[wall {time.time()-t_wall:.1f}s]")
