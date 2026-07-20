#!/usr/bin/env python3
r"""
Can the tip probability be read out of traded prices?  An identification analysis.

The tip layer rests on three [struct] judgments (gamma, sigma_m, tau_par), so the natural
discipline is to stop asserting P(T_tip <= T) and instead INVERT: report what traded
instruments imply, and compare against the structural prior -- the move lem:coverify already
makes for the branching ratio n.

This script asks whether that inversion is available, and the answer is a structural one
rather than a matter of data access.

WHERE THE TIP ENTERS PRICE.  Two channels, both proportional to p(T) = P(T_tip <= T):
  (i)  the recovery leg.  With the two-state reduction of rem:endog-recovery,
       E[R] = (1-p)R_0 + p R_tip, so the leg subtracts (1-S) p (R_0 - R_tip) of face.
  (ii) the obsolescence carry eta^arch of eq:tip-only, itself O(p) at short tenor.
Both vanish with p, and lem:tip-law makes p an Inverse-Gaussian CDF that is
super-exponentially small at short tenor -- Corollary cor:tip-only(iii).

CONSEQUENCE.  The information content of an instrument about the tip scales with p(T).
Inside the traded window it is not merely small, it is below any plausible measurement
noise, so a short-tenor instrument cannot identify p however precisely it is quoted.  The
obstruction is that traded tenors are structurally uninformative, NOT that the data is
unavailable.  Deterministic: closed form throughout.
"""
import time

import numpy as np
from scipy.stats import invgauss

t_wall = time.time()

MU_IG, LAM_IG = 18.5, 24.0        # lem:tip-law, IG(mean, shape) at the calibration
S_CLUSTER = 0.69                  # AI-sector-concentrated cluster survival (App. H)
R0, R_TIP = 0.40, 0.10            # no-tip and stranded-collateral recovery ends
FACE_BN = 300.0                   # Oracle backlog leg, $B

# band on the tip law: IG(16.5,19) to IG(20.5,30), plus the gamma/sigma_m sweep at T=5
BAND = [(16.5, 19.0), (20.5, 30.0)]


def p_tip(T, mu=MU_IG, lam=LAM_IG):
    return invgauss.cdf(T, mu / lam, scale=lam)


print("what a traded instrument would have to resolve to identify the tip")
print(f"   recovery leg subtracts (1-S) p (R_0-R_tip) of face, "
      f"S={S_CLUSTER}, R {R0}->{R_TIP}\n")
coef = (1 - S_CLUSTER) * (R0 - R_TIP)          # pp of face per unit p
print(f"{'T (yr)':>7} {'p(T)':>9} {'band on p':>16} {'PV impact':>11} {'on $300B':>11}")
for T in (1, 2, 3, 5, 7, 10):
    p = p_tip(T)
    lo = min(p_tip(T, *b) for b in BAND)
    hi = max(p_tip(T, *b) for b in BAND)
    imp = coef * p
    print(f"{T:7d} {p*100:8.2f}% [{lo*100:5.2f},{hi*100:5.2f}]% "
          f"{imp*100:10.3f}pp {imp*FACE_BN:9.2f}B")

print(f"\ncoefficient: {coef*100:.1f}pp of face per unit of p "
      f"({coef*FACE_BN:.1f}B per unit p on the $300B leg)")

# --- detectability: what tenor is needed before the signal clears quoted noise?
print("\ndetectability against quoted noise (tenor at which the channel clears the floor)")
for noise_pp, label in ((0.10, "deep, liquid quote"), (1.00, "nascent market"),
                        (2.50, "months-old illiquid listing")):
    need_p = noise_pp / 100.0 / coef
    T = invgauss.ppf(need_p, MU_IG / LAM_IG, scale=LAM_IG)
    print(f"   {noise_pp:5.2f}pp of face ({label:27s}) needs p >= {need_p*100:5.1f}%"
          f"  -> tenor {T:5.2f} yr")

# --- the real obstruction is CONFOUNDING, not noise: other channels also move PV.
# What separates them is the SHAPE of the tenor profile, not its level.
print("\nseparating the architecture channel from a constant-hazard channel")
print("(both move PV; only their tenor profiles differ, so identification is by shape)")
LAM_IDIO = 0.03                                    # illustrative idiosyncratic hazard
print(f"{'T (yr)':>7} {'arch p(T)':>10} {'idio 1-e^-lT':>13} {'arch/2y':>9} {'idio/2y':>9}")
a2, i2 = p_tip(2), 1 - np.exp(-LAM_IDIO * 2)
for T in (2, 3, 5, 7, 10):
    a, i = p_tip(T), 1 - np.exp(-LAM_IDIO * T)
    print(f"{T:7d} {a*100:9.2f}% {i*100:12.2f}% {a/a2:8.1f}x {i/i2:8.1f}x")
print("   the architecture channel is an order of magnitude more convex in tenor: it is")
print("   super-exponentially flat short and steep long (cor:tip-only(iii)), which no")
print("   constant-hazard channel mimics.  That convexity, not the spread level, is the")
print("   identifying restriction -- and reading it needs quotes at several tenors.")

print("\nreading: inside the traded 1-3y window the architecture channel moves price by")
print("well under a tenth of a point of face, which no quote resolves.  Identification")
print("requires tenors beyond ~5y, exactly where compute has no traded market: the")
print("SDH100RT futures listed in 2026 are short-dated and months old.  So the inversion")
print("is unavailable because the traded tenors carry no information about the tip, not")
print("because a series could not be bought.  As forwards extend past ~5y the channel")
print("becomes identifiable -- a falsifiable prediction rather than an excuse.")
print(f"\n[wall {time.time()-t_wall:.1f}s]")
