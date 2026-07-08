"""
Cohort re-run under the executory §365 recovery refinement, CLOSED FORM (no Monte Carlo).
Matches the paper's own method: the recovery leg is a truncated-lognormal moment + a normal-CDF
digital, integrated against the default-time density by quadrature (the paper's "risky-annuity
quadrature"). Exact to quadrature precision -> no sampling noise (unlike the earlier MC).

Inputs are the paper's real calibration (λ from TRACE/spreads, R^bk from seniority, η from used-GPU
resale, σ_P from realized rentals, G^g from the commodity branch). Nothing here is simulated.

Recovery(t) = w(t)·R_a·Π^g(t) + (1-w(t))·R^bk·V,  w(t)=Φ((k-Y^P(t))/s),  k=ln K (=own forward).
Y^P(t) ~ N(m(t), v(t)),  m(t)=ln P0 - η t,  v(t)=σ_P² t  (spot at default, unconditional law).
Two exact moments used:
    E[Φ((k-X)/s)]      = Φ((k-m)/√(v+s²))
    E[e^X Φ((k-X)/s)]  = e^{m+v/2} Φ((k-m-v)/√(v+s²))
Default-time density f(t)=λ e^{-λ t} on [0,T]; ∫_0^T f = 1-S, so recovery weights match the flat leg.
"""
import numpy as np, time
from numpy.polynomial.legendre import leggauss
try:
    from scipy.special import ndtr as Phi
except Exception:
    from math import erf
    Phi = np.vectorize(lambda x: 0.5*(1+erf(x/np.sqrt(2))))
t0 = time.time()

# ---- fixed grade-set inputs (real calibration) ----
P0, G, V, Lop, sigP, T = 2.58, 1.63, 2.58, 0.04, 0.35, 2.0
eta = 0.5*np.log(P0/G)          # ~0.230/yr, so median spot(T)=G^g
Ra, s_sm = 0.95, 0.30

# Gauss-Legendre nodes on [0,T] for the default-time integral
xg, wg = leggauss(256)
tq = 0.5*T*(xg+1.0);  wq = 0.5*T*wg          # nodes, weights on [0,T]
m  = np.log(P0) - eta*tq                       # E[Y^P(t)]
v  = sigP**2 * tq                              # Var[Y^P(t)]

def rec_legs(lam, R, K, s):
    """closed-form assume/reject recovery legs, integrated over default time."""
    k   = np.log(K)
    sd  = np.sqrt(v + s*s)
    Ew  = Phi((k - m)/sd)                       # E[w(t)]  (assume prob at t)
    EPw = np.exp(m + v/2.0) * Phi((k - m - v)/sd)  # E[Π^g(t) w(t)]
    dens = lam*np.exp(-lam*tq)                  # default-time density
    assume = np.sum(wq * (Ra*EPw)        * dens)
    reject = np.sum(wq * (R*V*(1.0-Ew))  * dens)
    S = np.exp(-lam*T)
    wbar = np.sum(wq * Ew * dens) / (1.0 - S)
    return assume, reject, wbar

def price(lam, R, Gamma):
    S = np.exp(-lam*T); SGe = S*G*np.exp(-Gamma)
    Fflat = SGe + R*V*(1-S) - Lop
    K = Fflat
    for _ in range(6):
        a, r, wbar = rec_legs(lam, R, K, s_sm)
        FA = SGe + (a+r) - Lop
        if abs(FA-K) < 1e-6: K = FA; break
        K = FA
    return dict(S=S, SGe=SGe, Fflat=Fflat, assume=a, reject=r, Lrec=a+r, FA=FA, wbar=wbar)

cohort = [   # (name, λ^term per-yr, R^bk, Γ) — S=e^{-λ·T}=e^{-2λ^term}, reproduces tab:issuer-outputs
    ("AWS/GCP/Azure",   0.010, 0.35,  0.0000),
    ("Oracle",          0.030, 0.35,  0.0000),
    ("Hut 8",           0.003, 0.55,  0.0000),
    ("Galaxy (Helios)", 0.080, 0.60,  0.0000),
    ("Core Scientific", 0.100, 0.55, -0.0050),
    ("Applied Digital", 0.070, 0.55, -0.0035),
    ("TeraWulf",        0.080, 0.55, -0.0040),
    ("Cipher",          0.070, 0.55, -0.0035),
    ("CoreWeave",       0.070, 0.35, -0.0093),
    ("Nebius",          0.060, 0.35, -0.0059),
    ("Lambda",          0.120, 0.55, -0.0119),
    ("IREN",            0.100, 0.35, -0.0099),
]

print(f"{'Issuer':17s} {'λ/yr':>5s} {'R':>4s} {'Γ':>8s} | {'wbar':>4s} "
      f"{'Fflat':>6s} {'assume':>6s} {'reject':>6s} {'F_A':>6s} {'ΔF':>6s}")
print("-"*94)
Ff, FA = [], []
for name, lam, R, Gam in cohort:
    d = price(lam, R, Gam)
    Ff.append(d['Fflat']); FA.append(d['FA'])
    print(f"{name:17s} {lam:5.3f} {R:4.2f} {Gam:8.4f} | {d['wbar']:4.2f} "
          f"{d['Fflat']:6.3f} {d['assume']:6.3f} {d['reject']:6.3f} {d['FA']:6.3f} {d['FA']-d['Fflat']:+6.3f}")
print("-"*94)
print(f"BAND flat [{min(Ff):.3f},{max(Ff):.3f}] span {max(Ff)-min(Ff):.3f} ({100*(max(Ff)-min(Ff))/np.mean(Ff):.1f}%)")
print(f"BAND A    [{min(FA):.3f},{max(FA):.3f}] span {max(FA)-min(FA):.3f} ({100*(max(FA)-min(FA))/np.mean(FA):.1f}%)")

print("\nR^bk tornado (CoreWeave base λ=0.07/yr, Γ=-0.0093):")
lam, Gam = 0.07, -0.0093
for R in (0.15, 0.35, 1.00):
    S=np.exp(-lam*T); Ff_=S*G*np.exp(-Gam)+R*V*(1-S)-Lop
    d=price(lam,R,Gam)
    print(f"  R={R:4.2f}: F_flat={Ff_:.3f}  F_A={d['FA']:.3f}  (wbar={d['wbar']:.2f})")
def w(kind):
    S=np.exp(-lam*T); SGe=S*G*np.exp(-Gam)
    lo = (SGe+0.15*V*(1-S)-Lop) if kind=='flat' else price(lam,0.15,Gam)['FA']
    hi = (SGe+1.00*V*(1-S)-Lop) if kind=='flat' else price(lam,1.00,Gam)['FA']
    return hi-lo, (SGe+0.35*V*(1-S)-Lop) if kind=='flat' else price(lam,0.35,Gam)['FA']
wf,bf=w('flat'); wa,ba=w('A')
print(f"  tornado width flat={wf:.3f} ({100*wf/bf:.0f}%)  A={wa:.3f} ({100*wa/ba:.0f}%)  shrink {100*(1-wa/wf):.0f}%")

print("\ns-robustness (CoreWeave):")
for s in (0.15, 0.30, 0.001):
    s_sm = s
    print(f"  s={s:5.3f}: F_A={price(0.07,0.35,-0.0093)['FA']:.3f}")
s_sm=0.30
print(f"\nwall-clock: {time.time()-t0:.3f}s  (closed form, 256-node quadrature, 0 simulated paths)")
