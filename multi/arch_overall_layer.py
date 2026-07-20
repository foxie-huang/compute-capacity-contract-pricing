#!/usr/bin/env python3
r"""
OVERALL/MIX architecture layer for the accompanying paper -- the coexistence variable of the
two-variable synthesis (2026-07-06).

The overall NVIDIA share of AI-accelerator VALUE (training + inference, all deployment
modes) is genuinely eroding as inference migrates to ASICs; unlike the contestable
merchant-training share (arch_bifurcation.py, vertex/bistable), this variable sits in the
COEXISTENCE branch (rho^o > 2*gamma) and relaxes toward an interior equilibrium. It does
NOT drive substitution of the training deliverable; it enters the pricing where it
economically operates:
  (i)   demand-mix drift on G (marginal renter of NVIDIA-hours is inference-weighted)
        -- carried SYMBOLICALLY in the paper (no invented number);
  (ii)  resale/collateral depth behind the endogenous recovery R;
  (iii) the buffer-erosion walk n(Sbar(t)) arming the funding cascade continuously.

FLOW vs STOCK: sbar(t) = overall flow (value) share, fitted below; Sbar(t) = the
vintage-weighted installed-base share that collateral and buffers actually ride,
    dSbar/dt = delta_dep * (sbar - Sbar),   delta_dep ~ 0.3/yr (fleet turnover 3-4yr).

FIT (metric-consistent, replaces the retired 95%->82% splice): coexistence relaxation
    dsbar/dt = -k * sbar (1-sbar) (sbar - sstar)
fitted to the overall AI-accelerator REVENUE-share consensus series
    sbar(2024) = 0.87, sbar(2025) = 0.81, sbar(2026) = 0.75  [analyst estimates,
    Mercury/TrendForce/Morgan Stanley via aggregator -- PROVISIONAL].
Two transit conditions pin (k, sstar). CAVEATS: series is estimate-grade; value share
overstates physical share (NVIDIA $/GW is 1.5-2.7x ASIC); pre-2024 the series is
NON-monotonic (share rose into 2024), so an autonomous relaxation is valid post-2024
only. Unit-share robustness: TrendForce has GPUs at ~70% of 2026 AI-server SHIPMENTS
with ASIC units growing ~3x faster (JPM: unit crossover ~2027) -- units LEAD value, so
the fitted k is if anything conservative on direction.
"""
import time, math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t_wall0 = time.time()

# ---------------- anchors and structural constants ----------------
ANCH = {2024: 0.87, 2025: 0.81, 2026: 0.75}   # overall value share [DATA, provisional]
SBAR0_STOCK = 0.85                             # installed-base value share today [STRUCT]
DELTA_DEP   = 0.30                             # fleet turnover /yr [STRUCT]
N0, Q       = 0.64, 1.0                        # branching at today's stock; elasticity [DATA/ILLUS]
KAPPA_CAP   = 0.06                             # post-tip migration cap, share/yr [STRUCT]

# ---------------- fit (k, sstar) from the two 1-yr transits ----------------
def transit_time_unit_k(s_from, s_to, sstar, n=40001):
    """Time to flow from s_from DOWN to s_to under ds/dt=-1*s(1-s)(s-sstar) (k=1)."""
    ss = np.linspace(s_to, s_from, n)
    integrand = 1.0/(ss*(1-ss)*(ss-sstar))
    return float(np.trapezoid(integrand, ss))

def k_required(s_from, s_to, sstar):
    return transit_time_unit_k(s_from, s_to, sstar)   # k = t(k=1)/1yr

def gap(sstar):
    return k_required(0.87, 0.81, sstar) - k_required(0.81, 0.75, sstar)

lo, hi = 0.02, 0.7499                                   # wide bracket: +0.005 anchor shifts push sstar to ~0.10
for _ in range(80):                                     # bisection
    mid = 0.5*(lo+hi)
    if gap(lo)*gap(mid) <= 0: hi = mid
    else: lo = mid
SSTAR_O = 0.5*(lo+hi)
K_O     = k_required(0.87, 0.81, SSTAR_O)
RHO_2G  = None   # rho_o/2gamma not separately identified; k_o = rho_o - 2gamma_o > 0 is the regime statement

print("="*88)
print("OVERALL/MIX LAYER (coexistence): fit + stock walk   [provisional revenue-share anchors]")
print("="*88)
print(f"fit: sstar_o = {SSTAR_O:.3f}   k_o = {K_O:.3f}/yr   (k_o = rho_o - 2*gamma_o > 0 => coexistence)")
print(f"     check transits 0.87->0.81: {k_required(0.87,0.81,SSTAR_O)/K_O:.3f} yr, "
      f"0.81->0.75: {k_required(0.81,0.75,SSTAR_O)/K_O:.3f} yr (both should be 1.000)")

# ---------------- forward paths: flow sbar(t), stock Sbar(t), t=0 at 2026 ----------------
T_END, DT = 14.0, 0.005
tg = np.arange(0, T_END+DT, DT)
sbar = np.empty_like(tg); Sbar = np.empty_like(tg)
sbar[0], Sbar[0] = ANCH[2026], SBAR0_STOCK
for i in range(1, len(tg)):
    s, S = sbar[i-1], Sbar[i-1]
    for _ in (0,):                                      # RK4
        def f(sv): return -K_O*sv*(1-sv)*(sv-SSTAR_O)
        k1 = f(s); k2 = f(s+0.5*DT*k1); k3 = f(s+0.5*DT*k2); k4 = f(s+DT*k3)
        sbar[i] = s + DT*(k1+2*k2+2*k3+k4)/6
        g1 = DELTA_DEP*(s-S); g2 = DELTA_DEP*(0.5*(s+sbar[i])-(S+0.5*DT*g1))
        g3 = DELTA_DEP*(0.5*(s+sbar[i])-(S+0.5*DT*g2)); g4 = DELTA_DEP*(sbar[i]-(S+DT*g3))
        Sbar[i] = S + DT*(g1+2*g2+2*g3+g4)/6

def n_of(S):  return 1 - (1-N0)*(np.clip(S,1e-3,1)/SBAR0_STOCK)**Q
def M_of(S):  return 1.0/(1.0-n_of(S))

n_inf = float(n_of(SSTAR_O)); M_inf = float(M_of(SSTAR_O))
marks = [0,1,3,5,7,10]
print("-"*88)
print("walk (central path):")
print("  t      sbar    Sbar     n(Sbar)   band mult 1/(1-n)")
for T in marks:
    i = int(round(T/DT))
    print(f"  {T:>2}y   {sbar[i]:.3f}   {Sbar[i]:.3f}    {float(n_of(Sbar[i])):.3f}     {float(M_of(Sbar[i])):.2f}")
print(f"  inf   {SSTAR_O:.3f}   {SSTAR_O:.3f}    {n_inf:.3f}     {M_inf:.2f}")
print(f"  (deterministic walk: n {N0:.2f} -> {float(n_of(Sbar[int(5/DT)])):.2f} @5y -> {n_inf:.2f} @inf;"
      f"  multiplier 2.78 -> {float(M_of(Sbar[int(5/DT)])):.2f} -> {M_inf:.2f} -- no tail event required)")
print(f"  stranded corners for reference: Sbar=0.30 -> n={float(n_of(0.30)):.2f}, M={float(M_of(0.30)):.1f};"
      f"  Sbar=0.20 -> n={float(n_of(0.20)):.2f}, M={float(M_of(0.20)):.1f}")

# post-tip overlay (illustrative): contestable tip at year 5 accelerates the overall flow
# decline at KAPPA_CAP toward a stranded level 0.30 (severity capped -- a ramp, not a step)
t_tip = 5.0
sb2 = sbar.copy(); Sb2 = Sbar.copy()
i0 = int(round(t_tip/DT))
for i in range(i0+1, len(tg)):
    s_prev = sb2[i-1]
    sb2[i] = max(s_prev - KAPPA_CAP*DT, 0.30)
    Sb2[i] = Sb2[i-1] + DT*DELTA_DEP*(sb2[i-1]-Sb2[i-1])

# ---------------- figure 1: the walk ----------------
fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.7))
ax[0].plot(tg, 100*sbar, color='tab:blue', lw=2.2, label=r'overall flow share $\bar s(t)$ (fitted relaxation)')
ax[0].plot(tg, 100*Sbar, color='tab:orange', lw=2.2, label=r'installed-base share $\bar S(t)$: $\dot{\bar S}=\delta_{\rm dep}(\bar s-\bar S)$')
ax[0].axhline(100*SSTAR_O, color='gray', ls=':', lw=1)
ax[0].text(13.6, 100*SSTAR_O+0.6, f'$\\bar s^{{\\,*}}={SSTAR_O:.2f}$', ha='right', fontsize=8, color='gray')
for yr, sv in ((-2, 0.87), (-1, 0.81), (0, 0.75)):
    ax[0].plot(yr, 100*sv, 'D', color='k', ms=5)
ax[0].plot([], [], 'D', color='k', ms=5, label='anchors 2024--26 (revenue share, est.)')
ax[0].set_xlim(-2.4, 14); ax[0].set_ylim(52, 92)
ax[0].set_xlabel('years from 2026'); ax[0].set_ylabel('overall NVIDIA share (%)')
ax[0].legend(fontsize=7.5, loc='upper right')
ax[0].set_title('(a) overall/mix relaxation; stock lags flow', fontsize=10)

ax[1].plot(tg, M_of(Sbar), color='tab:blue', lw=2.4, label='central walk $1/(1-n(\\bar S(t)))$')
ax[1].plot(tg, M_of(Sb2), color='k', lw=1.6, ls='--',
           label=f'+ contestable tip at {t_tip:g}y ($\\kappa_{{\\rm cap}}$-limited)')
ax[1].axhline(1/(1-N0), color='gray', ls=':', lw=1)
ax[1].text(13.6, 1/(1-N0)+0.05, 'today: 2.8', ha='right', fontsize=8, color='gray')
ax[1].axhline(M_inf, color='gray', ls=':', lw=1)
ax[1].text(13.6, M_inf+0.05, f'walk limit: {M_inf:.1f}', ha='right', fontsize=8, color='gray')
ax[1].set_xlim(0, 14); ax[1].set_ylim(2.5, 6.2)
ax[1].set_xlabel('horizon (years)'); ax[1].set_ylabel('good-deal band multiplier $1/(1-n)$')
ax[1].legend(fontsize=7.5, loc='upper left')
ax[1].set_title('(b) the band multiplier walks, then jumps only at a tip', fontsize=10)
plt.tight_layout(); plt.savefig("fig_arch_walk.pdf"); plt.close()

# ---------------- figure 2: co-tipping phase plane, rebased on Sbar ----------------
fig, ax = plt.subplots(figsize=(6.6, 4.4))
Sg, Ng = np.meshgrid(np.linspace(0.05,1.0,240), np.linspace(0.30,0.97,240))
cs = ax.contourf(Sg, Ng, np.clip(1.0/(1.0-Ng),1,12), levels=np.linspace(1,12,23), cmap="YlOrRd", alpha=0.9)
plt.colorbar(cs, ax=ax, label="good-deal band multiplier $1/(1-n)$")
sgrid = np.linspace(0.05,1.0,200)
ax.plot(sgrid, n_of(sgrid),'k-',lw=2.2,label=r'coupling $n(\bar S)=1-(1-n_0)(\bar S/\bar S_0)^q$')
ax.plot(SBAR0_STOCK, N0, 'D', color='tab:blue', ms=10, zorder=5,
        label=f'now ($\\bar S_0$={SBAR0_STOCK}, $n_0$={N0})')
walk_idx = [int(round(T/DT)) for T in (1,3,5,7,10)]
ax.plot(Sbar, n_of(Sbar), color='tab:blue', lw=2.0, zorder=4)
for T, i in zip((1,3,5,7,10), walk_idx):
    ax.plot(Sbar[i], n_of(Sbar[i]), 'o', color='tab:blue', ms=4, zorder=5)
    ax.text(Sbar[i], n_of(Sbar[i])-0.035, f'{T}y', fontsize=6.5, ha='center', color='tab:blue')
ax.plot(Sb2[i0:], n_of(Sb2[i0:]), color='tab:red', lw=1.6, ls='--', zorder=4,
        label='post-tip: $\\kappa_{\\rm cap}$-limited ramp')
ax.annotate('', xy=(0.33, n_of(0.33)), xytext=(0.45, n_of(0.45)),
            arrowprops=dict(arrowstyle='-|>', color='tab:red', lw=1.6))
ax.axvline(SSTAR_O, color='gray', ls=':', lw=1.2)
ax.text(SSTAR_O-0.01, 0.33, 'walk limit $\\bar s^{\\,*}$', color='gray', fontsize=7.5, ha='right', rotation=90)
ax.text(0.09,0.44,'ARMED / near-critical\n(stranded corner)',fontsize=8)
ax.text(0.80,0.925,'the walk: deterministic,\nalong the relaxation',fontsize=8,ha='center')
ax.set_xlabel(r'overall installed-base share $\bar S$')
ax.set_ylabel(r'cascade branching ratio $n=\gamma^{\rm fc}/\kappa^{\rm fc}$')
ax.set_title('Co-tipping, rebased: the buffer walk arms continuously; a tip accelerates it',fontsize=10)
ax.legend(fontsize=7.5, loc='lower left'); ax.set_xlim(0.05,1.0); ax.set_ylim(0.30,0.97)
plt.tight_layout(); plt.savefig("fig_arch_cotipping.pdf"); plt.close()

out = dict(sstar_o=SSTAR_O, k_o=K_O, anchors=ANCH, Sbar0=SBAR0_STOCK, delta_dep=DELTA_DEP,
           n0=N0, q=Q, kappa_cap=KAPPA_CAP, n_inf=n_inf, M_inf=M_inf,
           walk={f"{T}y": dict(sbar=float(sbar[int(round(T/DT))]), Sbar=float(Sbar[int(round(T/DT))]),
                               n=float(n_of(Sbar[int(round(T/DT))])), M=float(M_of(Sbar[int(round(T/DT))])))
                 for T in marks})
json.dump(out, open("arch_overall_results.json","w"), indent=1)
print("-"*88)
print(f"figures: fig_arch_walk.pdf, fig_arch_cotipping.pdf; arch_overall_results.json   "
      f"[wall {time.time()-t_wall0:.2f}s]")

# ---------------- figure 3: the co-tip in time (fig_arch_cotip_path.pdf) ----------------
#   Single credit-side panel: the walk arms n(Sbar(t)) deterministically and an
#   illustrative contestable tip at t=5y (the same timing as the overlays above)
#   accelerates the arming at KAPPA_CAP. At an ILLUSTRATIVE mid-sweep baseline arrival
#   nu_fc = 0.10/yr (a row of the cohort sweep), the expected cluster-event rate
#   nu/(1-n(t)) climbs along the central walk, and one simulated path of the
#   self-exciting arrival (eq:fc-selfexcite; each event lifts the arrival by
#   gamma_fc(t) = n(t)*kappa_fc, decaying at kappa_fc) shows the over-dispersion
#   signature. The theta_fc intercept a tip fires is symbolic -- NOT simulated.
#   (The former n(t)-walk companion panel was dropped 2026-07-16: it re-plotted the
#   fig:arch-walk multiplier panel under n = 1 - 1/M; the walk information survives
#   in the dashed expected-rate curve below.)
t_wall1 = time.time()
KAPPA_FC = 2.0     # cascade decay /yr [STRUCT; = kappa_fc of arch_bifurcation.py]
NU_BASE  = 0.10    # illustrative baseline cluster arrival /yr [ILLUS; mid-sweep row]
n_walk = n_of(Sbar)
n_tip  = n_of(Sb2)
rate_walk = NU_BASE/(1.0 - n_walk)
rate_tip  = NU_BASE/(1.0 - n_tip)

rng = np.random.default_rng(5)   # seed chosen for a representative realization
                                 # (an isolated pre-tip event + a post-tip flurry);
                                 # the statistics are seed-independent
events, excit = [], 0.0
lam_path = np.empty_like(tg)
decay = math.exp(-KAPPA_FC*DT)
for i, t in enumerate(tg):
    excit *= decay
    lam = NU_BASE + excit
    lam_path[i] = lam
    if rng.random() < lam*DT:
        events.append(float(t))
        excit += n_tip[i]*KAPPA_FC          # gamma_fc(t) = n(t) * kappa_fc

fig, ax = plt.subplots(figsize=(9.4, 2.9))
ax.plot(tg, rate_walk, color='tab:blue', lw=1.6, ls='--',
        label='expected event rate $\\nu/(1-n(t))$ along the central walk')
ax.plot(tg, lam_path, color='tab:red', lw=1.3,
        label='one realization: self-exciting intensity (with tip)')
ax.plot(events, [0.055]*len(events), '|', color='k', ms=14, mew=1.6)
ax.text(0.25, 0.10, 'events', fontsize=7.5, color='k')
if len(events) >= 2:
    tf = events[-1]
    ax.annotate('self-excited flurry: each event lifts the\narrival by $\\gamma^{\\rm fc}(t)=n(t)\\,\\kappa^{\\rm fc}$',
                xy=(tf, NU_BASE + n_of(0.6)*KAPPA_FC*0.55), xytext=(2.1, 1.38), fontsize=7.5,
                arrowprops=dict(arrowstyle='-|>', color='tab:red', lw=1.2), color='tab:red')
ax.axvline(t_tip, color='k', ls=':', lw=1.0)
ax.text(t_tip+0.25, 0.55, 'contestable tip (§3), illustrative',
        fontsize=7.5, ha='left', va='bottom')
ax.set_ylabel('cluster-event arrival (/yr)')
ax.set_xlabel('years from 2026')
ax.set_xlim(0, 14)
ax.legend(fontsize=7.5, loc='upper left')
ax.set_title('the credit side of the co-tip: over-dispersion is the visible signature '
             '(illustrative $\\nu^{\\rm fc}=0.1$/yr)', fontsize=10)
plt.tight_layout(); plt.savefig("fig_arch_cotip_path.pdf"); plt.close()

print(f"co-tip path: {len(events)} simulated cluster events at t = "
      + ", ".join(f"{t:.1f}" for t in events) + f"  (seed 5, nu={NU_BASE}/yr)")
print(f"figure: fig_arch_cotip_path.pdf   [wall {time.time()-t_wall1:.2f}s]")
