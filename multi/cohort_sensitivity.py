#!/usr/bin/env python3
r"""
Companion computation for tab:cluster-sensitivity (§9, 2022 mining-pivot cohort) and the
tab:reattribution row (App. D.1) of the accompanying paper.

Model (eq:8.1, MO-thinned at constant intensities, T = 1yr):
  - idiosyncratic hazard mu_i = 0.10/yr per name;
  - one funding-cascade shock, K ~ Poisson(nu); given K = k, name i defaults w.p.
    1 - e^{-mu_i} (1-delta_i)^k  (independent across names given k);
  - per-name value V_i = 1{survive} + R 1{default}, R = 0.40 (mean recovery, held fixed);
  - basket Vbar = mean of the four V_i; deltas from tab:exposure.

Columns:
  joint survival  = E_k[ prod_i e^{-mu}(1-delta_i)^k ]         (= eq:8.1)
  mean value      = E[Vbar]
  cluster share   = Var(E[Vbar|K]) / Var(Vbar)   -- the share of basket-value variance
                    carried by the common firing count K (exact conditional split).
                    It SATURATES near 74%: heavier firing also raises the within-firing
                    name-resolution variance E[Var(Vbar|K)].

Reattribution row (D.1): f = 2/3 of each cluster loading moves to the continuous macro
factor; marginals preserved via mu_i' = mu_i + f*nu*delta_i, discrete intensity nu/3.
Reattributed share = Var(E[Vbar|K]; nu/3, mu') / Var(Vbar; baseline)  -- the moved
co-movement re-enters through the (hedgeable) continuous channel, so the held total is
the honest denominator.  Mean value is exactly preserved (0.678).
"""
import time
import numpy as np
from scipy.stats import poisson

t0 = time.time()
R = 0.40
DELTA = np.array([0.85, 0.80, 0.45, 0.30])   # tab:exposure: CN, CoreSci, Hut8, AppliedD
MU0 = np.full(4, 0.10)

def parts(nu, mu_vec=MU0, kmax=120):
    ks = np.arange(kmax)
    pk = poisson.pmf(ks, nu) if nu > 0 else np.eye(1, kmax)[0]
    surv = np.exp(-mu_vec)[None, :] * (1 - DELTA)[None, :] ** ks[:, None]
    joint = float((pk * np.prod(surv, axis=1)).sum())
    EVbar_k = (R + (1 - R) * surv).mean(axis=1)
    EV = float((pk * EVbar_k).sum())
    varE = float((pk * EVbar_k ** 2).sum()) - EV ** 2          # Var(E[Vbar|K])
    Evar = float((pk * ((1 - R) ** 2 * surv * (1 - surv)).sum(axis=1) / 16).sum())
    return joint, EV, varE, Evar

print(f"{'nu':>6} {'survival':>9} {'value':>7} {'share':>7}")
for nu in (0.0, 0.04, 0.10, 0.30, 1.18):
    s, v, varE, Evar = parts(nu)
    share = varE / (varE + Evar) if nu > 0 else 0.0
    print(f"{nu:>6} {s:>9.3f} {v:>7.3f} {share*100:>6.1f}%")

# D.1 reattribution: f = 2/3 to macro, marginal-preserving
_, _, N1, D1 = parts(1.18)
tot = N1 + D1
muP = 0.10 + (2/3) * 1.18 * DELTA
_, EVr, N2, _ = parts(1.18 / 3, muP)
print(f"\nreattribution (f=2/3): mean value {EVr:.3f} (preserved); "
      f"discrete share {N1/tot*100:.0f}% -> {N2/tot*100:.0f}%")
print(f"band coefficients: sqrt(share@1.18) = {np.sqrt(N1/tot):.3f}; "
      f"x h_perp=0.5 -> {0.5*np.sqrt(N1/tot):.2f}")
print(f"[wall {time.time()-t0:.2f}s]")
