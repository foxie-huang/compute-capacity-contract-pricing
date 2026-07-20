#!/usr/bin/env python3
r"""
Model-native bifurcation / first-passage analysis for the architecture section of
the accompanying paper -- TWO-VARIABLE SYNTHESIS version (2026-07-06).

State variable here: the CONTESTABLE merchant-training share s^c (captive volumes --
Google/Gemini on TPU, Anthropic/Claude on Trainium+TPU -- are off the simplex; they are
not lost in competition and not winnable back). On the contestable market every frontier
training build (Stargate, Colossus, Fairwater, Prometheus/Hyperion) is NVIDIA: the
operating point sits AT THE VERTEX s^c ~ 1, an attractor of the bistable branch
(rho^c < 2*gamma), which is why the position is stable -- a stable INTERIOR point would
be the coexistence signature, impossible in this branch.

1. BISTABILITY (two-family reduction of the paper's replicator eq:share):
       ds = s(1-s)[ gamma*(2s-1) + m ] dt = -U'(s) dt,
   quartic U, wells at s=0/1, barrier s* = 1/2 - m/(2 gamma).
2. SADDLE-NODE at m = -gamma (incumbent basin destroyed).
3. FIRST-PASSAGE TIP (noise on the FRONTIER, not the share): m is BM-with-drift
   (eq:merit-drift), tip = first passage of m to -gamma, Inverse-Gaussian
       T ~ IG(mean = (m0+gamma)/mdot, shape = (m0+gamma)^2/sigma_m^2).

RE-ANCHORING (supersedes the spliced share-trajectory calibration): m0 is NO LONGER inferred from an interior share point
(the old s*=0.42 -> m0=0.08 read the 80-85% OVERALL/inference-inclusive share as the
contestable state). At the vertex, m0 = vertex moat depth, expressed in closure time:
       m0 = mdot * tau_par,
tau_par = years of moat beyond HARDWARE parity: MI450X/Helios (H2 2026) is the first
credible merchant hardware-parity attempt, while the software/fabric moat (CUDA + the
NCCL-vs-RCCL collectives gap + NVLink/NVSwitch scale-up domain + 12-15mo physical
pre-commitment of GB200-class builds) closes only on the common clock (portability:
ROCm/RCCL, Triton, OpenXLA). tau_par in [4,8] yr, central 6 [STRUCT band].

The co-tipping figure moved to arch_overall_layer.py (the buffer walk n(Sbar) rides the
OVERALL stock share, not the contestable share).

PARAMETER PROVENANCE: DATA / STRUCT / ILLUS. To refit, edit only CAL.
"""
import time, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t_wall0 = time.time()

CAL = {
    "s0_c":   dict(val=0.97, units="share",     status="DATA",
                   source="contestable merchant frontier-training deployments universally NVIDIA "
                          "(Stargate GB200 ~450k/1.2GW; xAI Colossus ~555k/2GW; MSFT Fairwater; "
                          "Meta Prometheus/Hyperion) -- vertex ~1, plotted at 0.97",
                   refine="merchant training-cluster architecture census"),
    "tau_par":dict(val=6.0,  units="yr",        status="STRUCT", band=(4.0, 8.0),
                   source="moat-closure horizon BEYOND hardware parity: MI450X/Helios H2-2026 "
                          "closes the hardware leg; CUDA + NCCL-vs-RCCL + NVLink fabric + "
                          "physical pre-commitment persist on the common clock",
                   refine="ROCm/RCCL maturity milestones; Meta TPU-training benchmark (2026)"),
    "gamma":  dict(val=0.50, units="1/yr",      status="STRUCT",
                   source="lock-in strength; basin relaxation ~1/gamma ~2yr",
                   refine="joint fit of (gamma,m) once a contestable-share panel exists"),
    "mdot":   dict(val=0.04, units="1/yr^2",    status="STRUCT",
                   source="merit-closure rate beta^mig*(mu_a - phidot_a), the TWO common-clock "
                          "legs: perf/W catch-up mu_a + deterministic porting-cost decline "
                          "-phidot_a (ROCm/RCCL, Triton, OpenXLA) -- same maturation drivers "
                          "as the overall-layer relaxation (arch_overall_layer.py)",
                   refine="frontier drift mu_a + porting-cost slope phidot_a, x beta^mig"),
    "sigma_m":dict(val=0.15, units="1/yr^1.5",  status="STRUCT",
                   source="merit volatility beta^mig*sigma_a; perf/W generational dispersion "
                          "(~1.3-2x/gen) scaled by beta^mig",
                   refine="frontier vol sigma_a (relative perf/W series) x beta^mig"),
    "n0":     dict(val=0.64, units="-",         status="DATA",
                   source="co-verified structural x market branching ratio (eq:coverify; "
                          "Gao-network n_S + Oracle-spread n_M)",
                   refine="Route-C Hawkes on credit-spread jumps as defaults accumulate"),
    "kappa_fc":dict(val=2.0, units="1/yr",      status="STRUCT",
                   source="cascade decay from refinancing cadence (half-life ~4mo)",
                   refine="contractual payment/maturity/refi schedule of the cluster"),
}
def v(k): return CAL[k]["val"]

# ---- derived primitives (vertex anchoring) ----
g, s0 = v("gamma"), v("s0_c")
mdot, sig_m = v("mdot"), v("sigma_m")
tau_c = v("tau_par"); tau_lo, tau_hi = CAL["tau_par"]["band"]
m0      = mdot * tau_c                    # vertex moat depth, central
m0_lo   = mdot * tau_lo
m0_hi   = mdot * tau_hi
n0, kap = v("n0"), v("kappa_fc")
gam_fc  = n0 * kap

def U(s, m):  return (g/2)*s**4 - ((3*g-m)/3)*s**3 - ((m-g)/2)*s**2
def sbar_of(m): return 0.5 - m/(2*g)      # barrier s*(m)
sstar = sbar_of(m0)

# ---- Inverse-Gaussian first passage of m(t) to -gamma ----
def Phi(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def ig_cdf(T, mu, lam):
    if T <= 0: return 0.0
    a =  math.sqrt(lam/T)*(T/mu - 1)
    b = -math.sqrt(lam/T)*(T/mu + 1)
    return Phi(a) + math.exp(2*lam/mu)*Phi(b)

def ig_params(m0x):
    D0 = m0x + g
    return D0/mdot, D0**2/sig_m**2        # (mean, shape)

muIG,  lamIG  = ig_params(m0)
muLO,  lamLO  = ig_params(m0_lo)          # shallower moat -> nearer tip (upper P)
muHI,  lamHI  = ig_params(m0_hi)          # deeper moat -> farther tip (lower P)

tenors = [1, 2, 3, 4, 5, 7, 10]
P_c  = {T: ig_cdf(T, muIG, lamIG) for T in tenors}
P_lo = {T: ig_cdf(T, muLO, lamLO) for T in tenors}
P_hi = {T: ig_cdf(T, muHI, lamHI) for T in tenors}

print("="*88)
print("CONTESTABLE-MARKET LOCK-IN, VERTEX-ANCHORED (edit CAL to refit; band = tau_par in [4,8]yr)")
print("="*88)
for k, d in CAL.items():
    print(f"  {k:9s}= {str(d['val']):<6}{d['units']:<10} [{d['status']:6s}] {d['source'][:58]}")
print("-"*88)
print(f"derived: m0 = mdot*tau_par = {m0:.2f}/yr  (band [{m0_lo:.2f},{m0_hi:.2f}]);  m0/gamma = {m0/g:.2f}")
print(f"         barrier s*(m0) = {sstar:.2f};  vertex operating point s^c ~ 1 (plotted {s0})")
print(f"         saddle-node distance D0 = m0+gamma = {m0+g:.2f}  [{m0_lo+g:.2f},{m0_hi+g:.2f}]")
print(f"TIP LAW  T ~ IG(mean {muIG:.1f} yr, shape {lamIG:.1f})   band: IG({muLO:.1f},{lamLO:.1f}) .. IG({muHI:.1f},{lamHI:.1f})")
print(f"         (supersedes the spliced-trajectory IG(14.5, 15))")
print("         P(tip by T), central:  " + "  ".join(f"{T}y={P_c[T]*100:4.1f}%" for T in tenors))
print("         P(tip by T), lo-moat:  " + "  ".join(f"{T}y={P_lo[T]*100:4.1f}%" for T in tenors))
print("         P(tip by T), hi-moat:  " + "  ".join(f"{T}y={P_hi[T]*100:4.1f}%" for T in tenors))
print(f"cascade: n0={n0}  kappa^fc={kap}/yr  gamma^fc={gam_fc:.2f}/yr  band0=1/(1-n0)={1/(1-n0):.2f}")
print(f"         (the buffer walk n(Sbar) rides the OVERALL stock share -> arch_overall_layer.py)")

# ====================================================================================
# FIGURE: bistability + saddle-node + first-passage tip (vertex-anchored, with band)
# ====================================================================================
fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.6))
ss = np.linspace(-0.04, 1.04, 400)
ax[0].plot(ss, U(ss, m0), 'k', lw=2)
ax[0].plot([0,1],[U(0,m0),U(1,m0)],'o',color='tab:green',ms=7)
ax[0].plot(sstar,U(sstar,m0),'o',color='tab:red',ms=7)
ax[0].plot(s0,U(s0,m0),'D',color='tab:blue',ms=7)
ax[0].text(0.0,U(0,m0)+0.006,'locked\nout',fontsize=7,ha='center')
ax[0].text(sstar,U(sstar,m0)+0.006,'barrier $s^*$',color='tab:red',fontsize=8,ha='center')
ax[0].text(s0-0.05,U(s0,m0)+0.0035,'vertex (now)',color='tab:blue',fontsize=8,ha='right')
ax[0].text(1.0,U(1,m0)+0.006,'lock-in\n(NVIDIA)',fontsize=7,ha='center')
ax[0].set_xlabel('contestable training share $s^{c}$'); ax[0].set_ylabel('potential $U(s)$')
ax[0].set_title('(a) contestable double-well, vertex operating point',fontsize=10)

mr = np.linspace(-1.4, 1.4, 400)
ax[1].plot(mr, np.where(np.abs(mr)<1, 0.5-mr/2, np.nan),'--',color='tab:red',lw=1.6,label='unstable $s^*$')
ax[1].plot(mr, np.ones_like(mr),'-',color='tab:green',lw=1.6,label='stable')
ax[1].plot(mr, np.zeros_like(mr),'-',color='tab:green',lw=1.6)
for mc in (-1,1): ax[1].axvline(mc,color='gray',ls=':',lw=1)
ax[1].plot([m0_lo/g, m0_hi/g],[1.0,1.0],'-',color='tab:blue',lw=4,alpha=0.35)
ax[1].plot(m0/g, 1.0,'D',color='tab:blue',ms=8)
ax[1].annotate('moat closes on the\ncommon clock ($\\dot m$)', xy=(-0.95,0.97),
               xytext=(m0/g-0.05,0.72), fontsize=7, ha='right', va='top',
               arrowprops=dict(arrowstyle='-|>',color='tab:blue',lw=1.6))
ax[1].text(0,1.09,'bistable lock-in',ha='center',fontsize=8)
ax[1].text(-1.0,-0.13,'incumbent\ntips',ha='center',fontsize=7,color='tab:red')
ax[1].set_xlabel('merit/moat ratio $m/\\gamma$'); ax[1].set_ylabel('fixed points $s$')
ax[1].set_title('(b) saddle-node ($m\\!\\to\\!-\\gamma$); moat band',fontsize=10)
ax[1].legend(fontsize=7,loc='lower left'); ax[1].set_ylim(-0.2,1.24)

Tg = np.linspace(0.2, 25, 300)
ax[2].fill_between(Tg, [ig_cdf(T,muHI,lamHI) for T in Tg], [ig_cdf(T,muLO,lamLO) for T in Tg],
                   color='tab:blue', alpha=0.20, label=r'$\tau_{\rm par}\in[4,8]$yr band')
ax[2].plot(Tg, [ig_cdf(T,muIG,lamIG) for T in Tg], 'tab:blue', lw=2, label='central')
ax[2].plot(Tg, [ig_cdf(T,14.5,15.0) for T in Tg], color='tab:red', lw=1.4, ls='-.',
           label='superseded IG(14.5, 15)')
ax[2].axvspan(1,3, color='tab:green', alpha=0.12)
ax[2].text(2,0.86,'traded\n1--3yr',ha='center',fontsize=7,color='tab:green')
for T in (5,10):
    ax[2].plot(T, P_c[T],'o',color='tab:blue',ms=5)
    ax[2].text(T+0.3, P_c[T]+0.02, f'{P_c[T]*100:.0f}%', fontsize=7, color='tab:blue')
ax[2].set_xlabel('tenor $T$ (yr)'); ax[2].set_ylabel(r'$P(\mathrm{tip}\ \mathrm{by}\ T)$')
ax[2].set_title('(c) first-passage tip, re-anchored (band)',fontsize=10)
ax[2].legend(fontsize=7, loc='upper left'); ax[2].set_ylim(0,1)
plt.tight_layout(); plt.savefig("fig_arch_bifurcation.pdf"); plt.close()

print("-"*88)
print(f"figure: fig_arch_bifurcation.pdf   [wall {time.time()-t_wall0:.2f}s]")

# ====================================================================================
# FIGURE 2: anatomy of a tip -- one realization in time (fig_arch_tip_path.pdf)
#   (a) merit gap m(t) = m0 - mdot*t + sigma_m*W(t): the frontier does the diffusing;
#       a tip is the first passage of m to -gamma (the saddle-node), law
#       IG((m0+g)/mdot, (m0+g)^2/sigma_m^2) -- panel (c) of the figure above.
#   (b) the contestable share responds DETERMINISTICALLY given the merit path
#       (Assumption ass:arch): it holds at the vertex while the incumbent well exists;
#       once the well is destroyed, migration proceeds at the kappa_cap pool-limited
#       ramp (~6pp/yr) -- the SAME post-tip convention as the overlay in
#       arch_overall_layer.py -- a multi-year ramp, not a step.
# ====================================================================================
t_wall1 = time.time()
KAPPA_CAP = 0.06   # share/yr, post-tip migration cap [STRUCT; = arch_overall_layer.py]
rng   = np.random.default_rng(20260715)
DTP, T_SHOW = 0.01, 32.0
tgrid = np.arange(0.0, T_SHOW + DTP, DTP)
NPATH = 400
dW = rng.standard_normal((NPATH, len(tgrid)-1)) * math.sqrt(DTP)
Mp = np.empty((NPATH, len(tgrid))); Mp[:, 0] = m0
for i in range(1, len(tgrid)):
    Mp[:, i] = Mp[:, i-1] - mdot*DTP + sig_m*dW[:, i-1]

hit     = Mp <= -g
tipped  = hit.any(axis=1)
tip_i   = np.where(tipped, np.argmax(hit, axis=1), len(tgrid)+1)
tp      = np.where(tipped, tgrid[np.clip(tip_i, 0, len(tgrid)-1)], np.inf)
tp_fin  = tp[np.isfinite(tp)]
i_med   = int(np.argmin(np.abs(tp - muIG)))                    # near the central IG mean
early_target = float(np.quantile(tp_fin, 0.10))
i_early = int(np.argmin(np.abs(tp - early_target)))            # a left-tail passage
i_hold  = next(j for j in range(NPATH) if not tipped[j])       # a no-passage path
grays   = [j for j in range(NPATH) if j not in (i_med, i_early, i_hold)][:5]

def share_path(mrow, itip):
    """Replicator hold at the vertex until the passage; kappa_cap ramp after
    (the post-tip convention of arch_overall_layer.py's overlay)."""
    s = np.empty(len(tgrid)); s[0] = s0
    for i in range(1, len(tgrid)):
        if i > itip:
            s[i] = max(s[i-1] - KAPPA_CAP*DTP, 0.0)
        else:
            drift = s[i-1]*(1-s[i-1])*(g*(2*s[i-1]-1.0) + mrow[i-1])
            s[i] = min(max(s[i-1] + drift*DTP, 0.0), 1.0)
    return s

sB = share_path(Mp[i_med],  tip_i[i_med])
sE = share_path(Mp[i_early], tip_i[i_early])
sG = share_path(Mp[i_hold], tip_i[i_hold])
tB, tE = float(tp[i_med]), float(tp[i_early])

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9.6, 5.6),
                       gridspec_kw=dict(height_ratios=[1.12, 1.0]))
for j in grays:
    ax[0].plot(tgrid, Mp[j], color='0.78', lw=0.7, zorder=1)
ax[0].plot(tgrid, Mp[i_hold], color='0.55', lw=0.9, zorder=2)
ax[0].plot(tgrid, Mp[i_early], color='tab:orange', lw=1.7, zorder=3,
           label=f'left-tail passage ($T^*\\!\\approx\\!{tE:.0f}$y)')
ax[0].plot(tgrid, Mp[i_med], color='tab:blue', lw=2.1, zorder=4,
           label=f'central passage ($T^*\\!\\approx\\!{tB:.0f}$y $\\approx$ IG mean {muIG:.1f})')
ax[0].axhline(-g, color='tab:red', lw=1.5)
ax[0].axhline(0, color='gray', ls=':', lw=0.9)
ax[0].fill_between(tgrid, -0.98, -g, color='tab:red', alpha=0.05, zorder=0)
ax[0].text(0.4, -g-0.05, 'saddle-node $m=-\\gamma$: incumbent well destroyed',
           fontsize=7.5, color='tab:red', va='top')
ax[0].text(31.6, 0.015, 'merit parity', fontsize=7, color='gray', ha='right', va='bottom')
ax[0].plot(0, m0, 'D', color='tab:blue', ms=6, zorder=5)
ax[0].text(0.5, 0.64, '$m_0=0.24$: the vertex moat, closing at $\\dot m=0.04$/yr',
           fontsize=7.5, color='tab:blue', va='top')
for tt, cc in ((tB, 'tab:blue'), (tE, 'tab:orange')):
    ax[0].axvline(tt, color=cc, ls=':', lw=1.0, alpha=0.8)
    ax[1].axvline(tt, color=cc, ls=':', lw=1.0, alpha=0.8)
ax[0].set_ylabel('merit gap $m(t)$ (1/yr)')
ax[0].set_ylim(-0.98, 0.66)
ax[0].legend(fontsize=7.5, loc='upper right')
ax[0].set_title('(a) the frontier does the diffusing: merit-gap sample paths', fontsize=10)

ax[1].plot(tgrid, sG, color='0.55', lw=1.1, label='no passage: the vertex holds')
ax[1].plot(tgrid, sE, color='tab:orange', lw=1.7)
ax[1].plot(tgrid, sB, color='tab:blue', lw=2.1)
ax[1].set_ylim(-0.04, 1.09)
ax[1].set_ylabel('contestable share $s^{c}(t)$')
ax[1].set_xlabel('years')
ax[1].text(16.0, 1.025, 'vertex hold: shares do not jitter (deterministic given the merit path)',
           fontsize=7.5, color='0.25', ha='center')
ax[1].annotate('$\\kappa_{\\rm cap}$ pool-limited ramp, $\\leq$ 6 pp/yr:\na multi-year ramp, not a step',
               xy=(tE+5.6, s0-KAPPA_CAP*5.6), xytext=(5.4, 0.30), fontsize=7.5,
               arrowprops=dict(arrowstyle='-|>', color='tab:orange', lw=1.2), color='tab:orange')
ax[1].legend(fontsize=7.5, loc='lower left')
ax[1].set_title('(b) the share does the holding: hold at the vertex, then the capped ramp', fontsize=10)
plt.tight_layout(); plt.savefig("fig_arch_tip_path.pdf"); plt.close()

print(f"tip-path: {tipped.mean()*100:.0f}% of {NPATH} paths tip by {T_SHOW:.0f}y "
      f"(median of tipped {np.median(tp_fin):.1f}y); highlighted T* = {tB:.1f}y central, {tE:.1f}y left-tail")
print(f"figure: fig_arch_tip_path.pdf   [wall {time.time()-t_wall1:.2f}s]")
