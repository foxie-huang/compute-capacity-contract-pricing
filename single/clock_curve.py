"""
clock_curve.py — pre-launch fair-value REFERENCE curve for the announced compute strips.
Companion to cohort_pipeline.py: that script gates the CREDIT branch;
this one gates the COMMODITY branch and produces the exhibit numbers.

Object: the clock forward — the framework's MMM reference for a cash-settled index futures
(credit-free limit: S=1, Gamma=0, no recovery/op legs), lambda_RP = 0 by construction.
    f_model(tau)/P = E[ exp(-int_0^tau eta(zeta(s)) ds) ]
with the lifecycle chain of the paper: eta(M,C,O) = (0.23, 0.20, 0.125)/yr (tab:eta-anchors),
transition times tau1 ~ N(2.5, 0.5^2) yr from M entry, tau2 ~ N(2.0, 0.5^2) yr from C entry
(tab:transitions; Gaussian probit-hazard jitter), continuous descent (H_C = eta_M*T1bar = 0.575,
H_O = 0.975 — eq:D-def text). Deterministic 2-D quadrature over (tau1, tau2); no simulation.

GATE: f_model(24mo from M entry) * P0 must reproduce the paper's G^g = $1.63 (P0 = $2.58).

Outputs: entry-anchored curve (t0 = Q4-2024 convention), the age-conditional mid-2026 curve
(H100 ~1.75yr into M, conditioned still-M), lambda_RP scenario bands, the sigma_f(tau) Samuelson
kit (kappa = 0.7/yr supply mean reversion, App. A; transient share backed out of the B.2
effective persistence 0.85 at year scale), hedge rescale (P/f)(sigma_P/sigma_f), and the
day-one hook lambda_rp(tau, f_mkt) = ln(f_mkt/f_model)/tau. Settlement convention: point-in-time
index at expiry (ASSUMPTION until the CME/ICE specs are public; a monthly-average settlement
adds an Asian adjustment ~ +eta*Delta/2 in log terms over the averaging window Delta).
"""
import numpy as np, time
T0 = time.time()

P0, G_PAPER = 2.58, 1.63
ETA = {"M": 0.23, "C": 0.20, "O": 0.125}
T1BAR, S1, T2BAR, S2 = 2.5, 0.5, 2.0, 0.5
KAPPA, PERSIST_1Y = 0.7, 0.85          # supply mean-reversion (App. A); B.2 effective persistence

# ---- transition-time grids (truncated normals, renormalized) ----
def tgrid(mu, sd, n=600):
    x = np.linspace(max(1e-4, mu-5*sd), mu+5*sd, n)
    w = np.exp(-0.5*((x-mu)/sd)**2); w /= w.sum()
    return x, w
X1, W1 = tgrid(T1BAR, S1)
X2, W2 = tgrid(T2BAR, S2)
T1 = X1[:, None]; T2 = X2[None, :]; W = W1[:, None]*W2[None, :]   # joint grid

def int_eta(tau, age=0.0):
    """E[exp(-int eta ds)] over [age, age+tau], conditional on state M at `age`."""
    if age > 0:                                   # condition tau1 > age, renormalize
        keep = X1 > age
        w1 = np.where(keep, W1, 0.0); w1 /= w1.sum()
        w = w1[:, None]*W2[None, :]
    else:
        w = W
    tM = np.clip(T1 - age, 0.0, tau)              # time spent in M within the window
    tC = np.clip(T1 + T2 - age, 0.0, tau) - tM    # time in C
    tO = tau - tM - tC                             # remainder in O
    I  = ETA["M"]*tM + ETA["C"]*tC + ETA["O"]*tO
    return float(np.sum(w*np.exp(-I)))

def int_eta_C(tau):
    """E[exp(-int eta ds)] over [0,tau] conditional on C entry AT the valuation date
    (fresh C sojourn ~ N(2.0, 0.5^2) per the semi-Markov clock restart)."""
    tC = np.clip(X2, 0.0, tau)
    tO = tau - tC
    return float(np.sum(W2*np.exp(-(ETA["C"]*tC + ETA["O"]*tO))))

def sigma_f_ratio(tau):
    """sigma_f(tau)/sigma_P: transient (mean-reverting) share decays exp(-kappa*tau)."""
    perm = (PERSIST_1Y - np.exp(-KAPPA)) / (1.0 - np.exp(-KAPPA))   # persistence identity at 1yr
    return np.sqrt(perm + (1-perm)*np.exp(-2*KAPPA*tau)), perm

def lambda_rp(tau, f_mkt, age=0.0, P=P0):
    """Day-one hook: premium over the clock from a TRADED settlement (do not feed marks)."""
    return np.log(f_mkt/(P*int_eta(tau, age)))/tau

# ---- GATE ----
f24 = int_eta(2.0, age=0.0)
print(f"GATE  f_model(24mo|M entry) = P0 x {f24:.4f} = ${P0*f24:.3f}  vs paper G^g = ${G_PAPER:.2f}"
      f"  -> {'PASS' if abs(P0*f24-G_PAPER)<0.005 else '*** FAIL ***'}")
print(f"      (flat-0.23 comparison: ${P0*np.exp(-0.46):.3f} — the mixture's early-C flattening adds "
      f"${P0*(f24-np.exp(-0.46)):+.3f})")

# ---- curves ----
AGE26 = 1.75    # H100 ~1.75yr past B200 launch (Q4-2024) as of mid-2026
print(f"\nREFERENCE CURVE f_model(tau)/P — mid-2026 STATE-CONDITIONAL BRACKET")
print( "  still-M (roadmap prior: C entry ~Q2-2027)  vs  just-C (event reading: Blackwell mainstream,")
print( "  'H100 = legacy' commentary — the M->C trigger arguably fired). Exhibit carries BOTH.")
print(f"{'tau(mo)':>7s} {'entry f/P':>10s} | {'mid26|M':>8s} {'ann%':>6s} | {'mid26|C':>8s} {'ann%':>6s} "
      f"| {'sig_f/sig_P':>11s} {'rescale|M':>10s}")
for m in (1, 2, 3, 6, 9, 12, 18, 24):
    tau = m/12
    fe, fM, fC = int_eta(tau, 0.0), int_eta(tau, AGE26), int_eta_C(tau)
    sr, perm = sigma_f_ratio(tau)
    print(f"{m:7d} {fe:10.4f} | {fM:8.4f} {-np.log(fM)/tau*100:5.1f}% | {fC:8.4f} {-np.log(fC)/tau*100:5.1f}% "
          f"| {sr:11.3f} {(1/fM)*(1/sr):10.3f}")
print(f"      (permanent variance share backed out of persistence(1y)=0.85, kappa=0.7: perm={perm:.2f})")
print(f"      state gap at 12/24mo: {100*(int_eta_C(1.0)-int_eta(1.0,AGE26)):.1f} / "
      f"{100*(int_eta_C(2.0)-int_eta(2.0,AGE26)):.1f} pts of f/P — comparable to the ±2% lambda_RP band")
# realized sanity: clock from the paper's Q4-2024 anchor to mid-2026
for lbl, val in (("still-M path", P0*int_eta(AGE26, 0.0)),
                 ("M 0.75yr then C", P0*np.exp(-(0.23*0.75 + 0.20*(AGE26-0.75))))):
    print(f"      clock-implied mid-2026 spot, {lbl}: ${val:.2f}  (observed marketplace ~$1.5-2.3)")

# ---- lambda_RP scenario bands at key tenors (mid-2026 curve) ----
print("\nSCENARIO BANDS mid-2026 f/P x exp(lambda_RP*tau)   [lambda_RP per yr]")
print(f"{'tau(mo)':>7s} " + " ".join(f"{f'{l:+.0%}':>8s}" for l in (-0.05,-0.02,0.0,0.02,0.05)))
for m in (3, 6, 12, 24):
    tau = m/12; f26 = int_eta(tau, AGE26)
    print(f"{m:7d} " + " ".join(f"{f26*np.exp(l*tau):8.4f}" for l in (-0.05,-0.02,0.0,0.02,0.05)))

# ---- day-one hook demo (SYNTHETIC input, clearly labeled; not a market number) ----
demo = int_eta(0.5, AGE26)*np.exp(0.03*0.5)
print(f"\nday-one hook (SYNTHETIC demo, not market data): a 6-mo settlement at f/P={demo:.4f} "
      f"would imply lambda_RP = {lambda_rp(0.5, P0*demo, AGE26):+.3f}/yr")

print(f"\nwall-clock: {time.time()-T0:.3f}s  (600x600 quadrature, 0 simulated paths)")
