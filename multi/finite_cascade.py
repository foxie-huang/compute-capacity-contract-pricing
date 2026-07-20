#!/usr/bin/env python3
r"""
Finite-name correction to the funding-cascade band multiplier (Appendix E).

Proposition prop:hawkes gives the branching moments of the self-exciting cascade
eq:fc-selfexcite in the INFINITE-population limit and states the Fano factor as a
T -> infinity limit:

    E[C] = 1/(1-n),  Var[C] = n/(1-n)^3,  Var[N_T]/E[N_T] -> 1/(1-n)^2,   [eq:fano]

so the band half-width, which scales with sd of the cluster loss, carries the
multiplier 1/(1-n).  The paper applies that multiplier at |k| = 3-6 names and a
five-year horizon.  Two approximations are therefore live, and this script measures
BOTH, separately:

  (i)  FINITE HORIZON    -- eq:fano is a T -> infinity limit; at T = 5y the count is far
                            from its stationary law even with an unbounded population.
  (ii) FINITE POPULATION -- names default at most once, so the total loss is NOT
                            (per-firing loss) x (count): the cascade saturates at |k|.

EXACT MODEL (faithful to eq:fc-selfexcite): firings arrive at nu(t) = nu_0 + gamma_fc J(t);
J decays at kappa_fc and jumps by the number of NEW defaults; at a firing each SURVIVING
name i defaults independently w.p. delta_i; a defaulted name is absorbed.

BAND MULTIPLIER.  The half-width scales with sd of the total cluster loss
L = sum_i (w_i chi_i) 1{i defaults by T}, so the operative multiplier is
M(n) = sd(L | n) / sd(L | n = 0) -- directly comparable to the paper's 1/(1-n).

Ogata thinning (intensity decays between events, so the current value bounds it).
Deterministic: fixed seeds throughout.
"""
import math
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t_wall = time.time()

KAPPA = 2.0          # kappa^fc /yr          (tab:arch-cal, [STRUCT])
T_HOR = 5.0          # priced five-year horizon
NU0 = 0.10           # exogenous baseline /yr (mid-sweep, tab:cluster-sensitivity [ILLUS])
NPATH = 200_000

DELTA_MQ = np.array([0.90, 0.55, 0.35])   # marquee loadings, tab:circular-exposure
LOSS_MQ = np.array([1.00, 0.55, 0.30])    # w_i*chi_i, relative at-risk deliverable value


def simulate(n, delta, loss, nu0=NU0, kappa=KAPPA, T=T_HOR, npath=NPATH, seed=0):
    """Exact finite-name cascade. Returns (total loss, firing count) per path."""
    rng = np.random.default_rng(seed)
    gamma, N = n * kappa, len(delta)
    L = np.empty(npath)
    cnt = np.empty(npath)
    for p in range(npath):
        t, J, k = 0.0, 0.0, 0
        alive = np.ones(N, dtype=bool)
        while True:
            lam_max = nu0 + gamma * J             # decays until the next event
            w = rng.exponential(1.0 / lam_max)
            t_new = t + w
            if t_new >= T:
                break
            J_new = J * math.exp(-kappa * w)
            if rng.random() <= (nu0 + gamma * J_new) / lam_max:
                k += 1
                if alive.any():
                    hit = alive & (rng.random(N) < delta)     # survivors only
                    J_new += hit.sum()
                    alive &= ~hit
            t, J = t_new, J_new
        L[p] = loss[~alive].sum()
        cnt[p] = k
    return L, cnt


# ----------------------------------------------------------------- validation
p_analytic = 1.0 - np.exp(-NU0 * DELTA_MQ * T_HOR)      # n=0: firings are Poisson(nu_0)
mean_analytic = (LOSS_MQ * p_analytic).sum()
L0, _ = simulate(0.0, DELTA_MQ, LOSS_MQ, seed=1)
print("validation at n=0 (Poisson firings; closed form vs simulation)")
print(f"   E[L] closed form {mean_analytic:.4f} | simulated {L0.mean():.4f} "
      f"| rel. err {abs(L0.mean()-mean_analytic)/mean_analytic*100:.2f}%\n")

# --------------------------------------------- operative multiplier + attribution
GRID = [0.0, 0.30, 0.50, 0.64, 0.71, 0.76, 0.87, 0.92, 0.95]
TOTAL = LOSS_MQ.sum()
sd0_fin = L0.std(ddof=1)
_, c0 = simulate(0.0, DELTA_MQ, LOSS_MQ, seed=1)
sd0_inf = (TOTAL * c0).std(ddof=1)

print(f"band multiplier: envelope vs exact, with the error attributed"
      f"   [|k|=3, nu_0={NU0}/yr, kappa={KAPPA}/yr, T={T_HOR}y, {NPATH:,} paths]")
print(f"{'n':>6} {'envelope':>9} {'inf-pop@T':>10} {'exact':>7} "
      f"{'horizon':>8} {'saturation':>11} {'total':>7}")
exact_mq = []
for k, n in enumerate(GRID):
    L, c = simulate(n, DELTA_MQ, LOSS_MQ, seed=10 + k)
    env = 1.0 / (1.0 - n)
    m_inf = (TOTAL * c).std(ddof=1) / sd0_inf
    m_fin = L.std(ddof=1) / sd0_fin
    exact_mq.append(m_fin)
    print(f"{n:6.2f} {env:9.2f} {m_inf:10.2f} {m_fin:7.2f} "
          f"{env/m_inf:7.2f}x {m_inf/m_fin:10.2f}x {env/m_fin:6.2f}x")

print(f"\nsd(L) is bounded by (sum_i w_i chi_i)/2 = {TOTAL/2:.2f}, so the multiplier is "
      f"capped at {TOTAL/2/sd0_fin:.2f}\nregardless of n: the 1/(1-n) divergence is "
      f"unreachable in any finite cluster.")

# ----------------------------------------------------------------- robustness
print("\nrobustness of the exact multiplier (envelope is 2.78 at n=0.64, 7.69 at n=0.87)")
print(f"{'|k|':>4} {'nu_0':>6} {'n=0.64':>8} {'n=0.87':>8}")
ROB = {}
for K, dd in ((3, DELTA_MQ), (6, np.full(6, 0.6)), (10, np.full(10, 0.5))):
    ll = LOSS_MQ if K == 3 else np.ones(K)
    for nu0 in (0.05, 0.10, 0.30, 1.00):
        s0 = simulate(0.0, dd, ll, nu0=nu0, npath=60_000, seed=1)[0].std(ddof=1)
        row = []
        for n in (0.64, 0.87):
            s = simulate(n, dd, ll, nu0=nu0, npath=60_000, seed=2)[0].std(ddof=1)
            row.append(s / s0)
        ROB[(K, nu0)] = row
        print(f"{K:>4} {nu0:>6.2f} {row[0]:>8.2f} {row[1]:>8.2f}")

# ----------------------------------------------------------------- the walk
print(f"\nalong the deterministic walk (n rises at unchanged exogenous nu_0)")
print(f"{'n':>6} {'envelope':>9} {'exact':>7}  (both indexed to n_0=0.64)")
Lb, _ = simulate(0.64, DELTA_MQ, LOSS_MQ, seed=41)
sdb = Lb.std(ddof=1)
for n in (0.64, 0.68, 0.71, 0.76):
    L, _ = simulate(n, DELTA_MQ, LOSS_MQ, seed=42)
    print(f"{n:6.2f} {(1/(1-n))/(1/(1-0.64)):9.2f} {L.std(ddof=1)/sdb:7.2f}")

# ----------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(6.3, 3.5))
nn = np.linspace(0, 0.95, 300)
ax.plot(nn, 1.0 / (1.0 - nn), color="tab:red", lw=2.0,
        label=r"envelope $1/(1-n)$:  $|k|\to\infty$, $T\to\infty$")
ax.plot(GRID, exact_mq, color="tab:blue", marker="o", ms=4, lw=1.7,
        label=r"exact finite-name cascade, $|k|=3$ (marquee)")
ax.plot((0.64, 0.87), (ROB[(10, 0.10)][0], ROB[(10, 0.10)][1]), color="tab:green",
        marker="s", ms=4, lw=1.5, ls="--", label=r"exact, $|k|=10$")
ax.axvline(0.64, color="k", ls=":", lw=1.0)
ax.text(0.655, 12.2, "operating $n_0=0.64$", fontsize=7.5, rotation=90, va="top")
ax.axvspan(0.85, 0.95, color="0.86", alpha=0.6, zorder=0)
ax.text(0.90, 0.6, "stranded corner", fontsize=7, ha="center", color="0.3")
ax.set_xlabel(r"branching ratio $n=\gamma^{\rm fc}/\kappa^{\rm fc}$")
ax.set_ylabel("good-deal band multiplier")
ax.set_xlim(0, 0.95)
ax.set_ylim(0, 13)
ax.legend(fontsize=7.5, loc="upper left")
plt.tight_layout()
plt.savefig("fig_finite_cascade.pdf")
plt.close()
print(f"\nfigure: fig_finite_cascade.pdf   [wall {time.time()-t_wall:.1f}s]")
