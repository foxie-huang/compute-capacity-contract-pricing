"""
cohort_pipeline.py — production-fidelity confirmation of the paper's per-issuer cohort tables
(tab:issuer-outputs and tab:issuer-outputs-A) under the executory-recovery and surprise-erosion
refinements. Extends scripts/cohort_rerun_closedform.py with fidelity variants: the goal is that
every number printed in the paper either reproduces here or the paper gets corrected.

Stages
  V0  nested-limit GATE: flat recovery must reproduce tab:issuer-outputs to the cent.
  V1  published A numbers: closed form, flat-λ default density, median-matched drift (E ln Π hits ln G).
  V2  β·D-drifted default density (per-name REALIZED β from tab:omega; ramp calibrated to the paper's
      own ceiling statement: at β=0.17 the 24-mo cum hazard rises 0.140→0.144).
  V3  drift-convention variant: MEAN-matched (E[Π(T)] = G^g, the martingale-consistent reading) —
      the step-6-style decision quantified for the author.
  V4  coupled spot law: antithetic MC with the intensity spot-loaded to reproduce each name's Γ_total
      (B.2's channel) — closes the "unconditional law" simplification.
  T   R^(bk) tornado flat vs A;  S  s-robustness;  B2  analytic S-shift bound + Γ_struct/ϱ_resid table.

Inputs are the paper's real calibration (TRACE/EDGAR λ,R; SEC-filing ω incl. IREN 0.15 post-resolution;
rental-series σ_P; commodity-branch G^g). Wall-clock printed per stage and total.
"""
import numpy as np, time
from numpy.polynomial.legendre import leggauss
try:
    from scipy.special import ndtr as Phi
except Exception:
    from math import erf
    Phi = np.vectorize(lambda x: 0.5*(1.0+erf(x/np.sqrt(2.0))))

T0 = time.time()
P0, G, V, Lop, sigP, T = 2.58, 1.63, 2.58, 0.04, 0.35, 2.0
Ra, s_base = 0.95, 0.30
eta_med  = np.log(P0/G)/T                 # median-matched: m(T)=ln G          (published convention)
eta_mean = eta_med + 0.5*sigP**2          # mean-matched:   E[Pi(T)]=G         (variant V3)

# name, λ^term/yr, R^bk, Γ_table, β^life (tab:omega, post-IREN), F^g published, F^g_A published(±0 => ≈0)
COHORT = [
    ("AWS/GCP/Azure",   0.010, 0.35,  0.0000, 0.002, 1.58, None),
    ("Oracle",          0.030, 0.35,  0.0000, 0.007, 1.55, None),
    ("Hut 8",           0.003, 0.55,  0.0000, 0.003, 1.59, None),   # 1.58->1.59: gate-caught cell fix 2026-07-06
    ("Galaxy (Helios)", 0.080, 0.60,  0.0000, 0.000, 1.58, None),
    ("Core Scientific", 0.100, 0.55, -0.0050, 0.000, 1.56, None),
    ("Applied Digital", 0.070, 0.55, -0.0035, 0.007, 1.57, None),
    ("TeraWulf",        0.080, 0.55, -0.0040, 0.000, 1.56, None),
    ("Cipher",          0.070, 0.55, -0.0035, 0.000, 1.57, None),
    ("CoreWeave",       0.070, 0.35, -0.0093, 0.043, 1.51, 1.52),
    ("Nebius",          0.060, 0.35, -0.0059, 0.024, 1.52, 1.53),
    ("Lambda",          0.120, 0.55, -0.0119, 0.068, 1.56, None),
    ("IREN",            0.100, 0.35, -0.0099, 0.026, 1.47, 1.49),
]

xg, wg = leggauss(256)
tq = 0.5*T*(xg+1.0); wq = 0.5*T*wg

def closed_form(lam0, R, Gamma, eta, s=s_base, beta=0.0, ramp=0.0):
    """A-recovery forward. beta,ramp>0 => λ(t)=lam0+beta*ramp*t (β·D-drifted density, V2)."""
    m = np.log(P0) - eta*tq
    v = sigP**2 * tq
    lam_t = lam0 + beta*ramp*tq                       # instantaneous hazard path
    Lam   = lam0*tq + 0.5*beta*ramp*tq**2             # cumulative
    S     = np.exp(-(lam0*T + 0.5*beta*ramp*T**2))
    SGe   = S*G*np.exp(-Gamma)
    dens  = lam_t*np.exp(-Lam)
    K = SGe + R*V*(1-S) - Lop
    for _ in range(6):
        k  = np.log(K); sd = np.sqrt(v + s*s)
        Ew  = Phi((k-m)/sd)
        EPw = np.exp(m+v/2.0)*Phi((k-m-v)/sd)
        assume = np.sum(wq*(Ra*EPw)*dens)
        reject = np.sum(wq*(R*V*(1.0-Ew))*dens)
        FA = SGe + assume + reject - Lop
        if abs(FA-K) < 1e-7: K = FA; break
        K = FA
    wbar = np.sum(wq*Ew*dens)/max(1e-12, 1.0-S)
    Fflat = SGe + R*V*(1-S) - Lop
    return dict(S=S, Fflat=Fflat, FA=FA, assume=assume, reject=reject, wbar=wbar)

# ---------------- V0 + V1: gate and published numbers ----------------
t0 = time.time()
print("="*100)
print("V0 GATE (flat recovery vs tab:issuer-outputs)   |   V1 published A numbers (flat-λ, median drift)")
print(f"{'name':17s} {'F_flat':>7s} {'pub':>5s} {'gate':>5s} | {'F_A':>6s} {'pubA':>5s} {'ΔF':>7s} {'wbar':>5s}")
v1 = {}
gate_ok = True
for nm, lam, R, Gam, beta, Fpub, FApub in COHORT:
    d = closed_form(lam, R, Gam, eta_med)
    v1[nm] = d
    g = abs(d['Fflat'] - Fpub) < 0.005
    gate_ok &= g
    pa = f"{FApub:.2f}" if FApub else "  ≈0"
    print(f"{nm:17s} {d['Fflat']:7.3f} {Fpub:5.2f} {'PASS' if g else 'FAIL':>5s} | "
          f"{d['FA']:6.3f} {pa:>5s} {d['FA']-d['Fflat']:+7.3f} {d['wbar']:5.2f}")
Ff=[v1[n[0]]['Fflat'] for n in COHORT]; FA=[v1[n[0]]['FA'] for n in COHORT]
print(f"GATE: {'PASS — table reproduced to the cent' if gate_ok else '*** FAIL ***'}")
print(f"band flat {100*(max(Ff)-min(Ff))/np.mean(Ff):.1f}%  ->  A {100*(max(FA)-min(FA))/np.mean(FA):.1f}%"
      f"   [{time.time()-t0:.2f}s]")

# ---------------- V2: β·D-drifted default density ----------------
t0 = time.time()
ramp_ceiling = (0.144-0.140)/ (0.17*0.5*T**2)   # ∫β·a·t dt = β·a·T²/2 = 0.004 at β=0.17 → a
print("\nV2 β·D-DRIFTED DENSITY (per-name realized β; ramp a=%.4f/yr² from the paper's ceiling stmt)" % ramp_ceiling)
print(f"{'name':17s} {'ΔS':>8s} {'F_A(V2)':>8s} {'vs V1':>7s}")
worstS = 0.0
for nm, lam, R, Gam, beta, Fpub, FApub in COHORT:
    d2 = closed_form(lam, R, Gam, eta_med, beta=beta, ramp=ramp_ceiling)
    dS = d2['S'] - v1[nm]['S']; worstS = max(worstS, abs(dS))
    print(f"{nm:17s} {dS:+8.4f} {d2['FA']:8.3f} {d2['FA']-v1[nm]['FA']:+7.3f}")
print(f"max |ΔS| = {worstS:.4f}  (claim: ≤0.004 at realized β)   [{time.time()-t0:.2f}s]")

# ---------------- V3: mean-matched drift convention ----------------
t0 = time.time()
print("\nV3 DRIFT CONVENTION (mean-matched E[Π(T)]=G^g vs published median-matched)")
print(f"{'name':17s} {'F_A(mean)':>9s} {'vs V1':>7s} {'wbar':>5s}")
for nm, lam, R, Gam, beta, Fpub, FApub in COHORT:
    if beta < 0.02 and Gam == 0: continue          # report the credit-relevant names
    d3 = closed_form(lam, R, Gam, eta_mean)
    print(f"{nm:17s} {d3['FA']:9.3f} {d3['FA']-v1[nm]['FA']:+7.3f} {d3['wbar']:5.2f}")

# ---------------- V4: coupled spot law (antithetic MC) ----------------
t0 = time.time()
Nh, M = 200_000, 104                                # 200k antithetic pairs = 400k paths
rng = np.random.default_rng(20260706)
dt  = T/M; tgrid = np.linspace(0, T, M+1, dtype=np.float32)
Z   = rng.standard_normal((Nh, M)).astype(np.float32)
Z   = np.vstack([Z, -Z])                            # antithetic
ybar = (np.log(P0) - eta_med*tgrid).astype(np.float32)
Y = np.empty((2*Nh, M+1), dtype=np.float32); Y[:,0] = np.log(P0)
Y[:,1:] = np.float32(np.log(P0)) + np.cumsum(-eta_med*dt + sigP*np.sqrt(dt)*Z, axis=1, dtype=np.float32)
E = rng.exponential(1.0, size=2*Nh).astype(np.float32)
gap = ybar[None,1:] - Y[:,1:]
print(f"\nV4 COUPLED LAW (MC {2*Nh:,} antithetic paths; θ per name reproduces Γ_total)")
print(f"{'name':17s} {'F_A(MC)':>8s} {'SE':>6s} {'vs V1':>7s} {'wbar':>5s}")
for nm, lam, R, Gam, beta, Fpub, FApub in COHORT:
    if Gam == 0 and (FApub is None): continue       # coupling only matters where Γ≠0 (+ the movers)
    theta = 0.0 if Gam == 0 else -2.0*Gam/(sigP**2*T**2)
    lamt = np.maximum(0.0, lam + theta*gap)
    Lam  = np.cumsum(lamt*dt, axis=1)
    dflt = Lam[:,-1] >= E
    idx  = np.argmax(Lam >= E[:,None], axis=1)
    Yd   = Y[np.arange(2*Nh), idx+1][dflt]
    S    = np.exp(-2.0*lam)
    K    = v1[nm]['FA']                             # strike at the closed-form fixed point
    w    = Phi((np.log(K)-Yd)/s_base)
    rec  = w*Ra*np.exp(Yd) + (1-w)*R*V
    Lrec = (1-S)*rec.mean()
    se   = (1-S)*rec.std()/np.sqrt(len(rec))
    FAmc = S*G*np.exp(-Gam) + Lrec - Lop
    print(f"{nm:17s} {FAmc:8.3f} {se:6.4f} {FAmc-v1[nm]['FA']:+7.3f} {w.mean():5.2f}")
print(f"[{time.time()-t0:.2f}s]")

# ---------------- tornado + s-robustness ----------------
t0 = time.time()
lam, Gam = 0.07, -0.0093
S=np.exp(-2*lam); SGe=S*G*np.exp(-Gam)
wf = (SGe+1.0*V*(1-S)-Lop) - (SGe+0.15*V*(1-S)-Lop)
wa = closed_form(lam,1.0,Gam,eta_med)['FA'] - closed_form(lam,0.15,Gam,eta_med)['FA']
bf = SGe+0.35*V*(1-S)-Lop; ba = closed_form(lam,0.35,Gam,eta_med)['FA']
print(f"\nTORNADO R∈[0.15,1.00] (CoreWeave base): flat {wf:.3f} ({100*wf/bf:.0f}% of F) -> "
      f"A {wa:.3f} ({100*wa/ba:.0f}%)  shrink {100*(1-wa/wf):.0f}%")
print("s-ROBUSTNESS (CoreWeave):", " ".join(
    f"s={s:.3f}:{closed_form(0.07,0.35,-0.0093,eta_med,s=max(s,1e-4))['FA']:.3f}" for s in (0.001,0.15,0.30)))

# ---------------- B.2: analytic S-shift + re-attribution table ----------------
print("\nB.2  γ-term S-shift bound: ΔS ≈ ½γ²σ_P²T³/3 =",
      " ".join(f"{nm}:{0.5*(0.15*om)**2*sigP**2*T**3/3:.6f}"
               for nm,om in (("CoreWeave",0.25),("Nebius",0.14),("Lambda",0.40),("IREN",0.15))),
      " (all ≪ 0.004 ✓)")
print("B.2  re-attribution (Γ_total fixed): Γ_struct=-γσ_P²τ²/2, ϱ_resid=(Γ-Γ_struct)/(σ_Λσ_P√τ), σ_Λ=Γ/(-0.4σ_P√τ)")
for nm, om, Gam in (("CoreWeave",0.25,-0.0093),("Nebius",0.14,-0.0059),("Lambda",0.40,-0.0119),("IREN",0.15,-0.0099)):
    g = 0.15*om; Gs = -g*sigP**2*T**2/2
    sigL = Gam/(-0.40*sigP*np.sqrt(T)); resid = (Gam-Gs)/(sigL*sigP*np.sqrt(T))
    print(f"   {nm:10s} γ={g:.3f}  Γ_struct={Gs:+.4f}  ϱ_resid={resid:+.2f}")

# ---------------- H: recovery-moneyness delta in the hedge ratio ----------------
# Under eq:recovery-moneyness the recovery leg has a DIRECT spot delta (digital shifts weight to
# the reject leg as spot rises; the in-kind value rises with it). Central-difference bump of the
# log-spot at FIXED strike K; compare to the delivery-term delta S*G*e^{-Gamma} (per unit ln P).
def rec_leg_only(lam, R, K, mshift):
    m = np.log(P0) + mshift - eta_med*tq
    v = sigP**2 * tq
    dens = lam*np.exp(-lam*tq)
    k = np.log(K); sd = np.sqrt(v + s_base**2)
    Ew  = Phi((k-m)/sd)
    EPw = np.exp(m+v/2.0)*Phi((k-m-v)/sd)
    return float(np.sum(wq*(Ra*EPw + R*V*(1.0-Ew))*dens))
print("\nH  RECOVERY-MONEYNESS DELTA (dL_rec/dlnP at fixed K, vs delivery delta S*G*e^-Gamma)")
print(f"{'name':17s} {'dLrec/dlnP':>10s} {'delivery':>9s} {'xi corr.':>8s}")
eps = 0.01
for nm, lam, R, Gam, beta, Fpub, FApub in COHORT:
    if nm not in ("CoreWeave","IREN","Nebius","Lambda","Core Scientific","Galaxy (Helios)"): continue
    d0 = closed_form(lam, R, Gam, eta_med); K = d0['FA']
    dL = (rec_leg_only(lam,R,K,+eps) - rec_leg_only(lam,R,K,-eps))/(2*eps)
    deliv = d0['S']*G*np.exp(-Gam)
    print(f"{nm:17s} {dL:+10.3f} {deliv:9.3f} {100*dL/deliv:+7.1f}%")

print(f"\nTOTAL wall-clock: {time.time()-T0:.2f}s")
