"""
Calibrate B.2's surprise-erosion loading  γ_g = c'_g · ω_g  (grade = H100).
γ loads the realized-vs-clock gap (ȳ - Y^P) in the bankruptcy intensity.

Three INDEPENDENT routes to the grade scale c'_g; the calibration is their agreement.
 (1) Spread-consistency / natural experiment: the c'_g that makes the structural surprise channel
     REPRODUCE each owner's fitted Γ = Cov(Λ,Y^P).  Γ_struct(γ) = -γ σ_P² T²/2  ⇒  γ = -2Γ/(σ_P²T²).
 (2) Asset-side structural: same asset→intensity transmission as the EXPECTED channel (which pins
     c_g), per raw log-unit, scaled by the PERMANENT share of rental shocks (transient shocks are
     ridden out).  c'_g ≈ (c_g / H_O) · perm.
 (3) 2024-episode cross-check: implied owner spread-widening vs. what was observed.
"""
import numpy as np, time
t0 = time.time()

# fixed
sigP, T, cg, HO = 0.35, 2.0, 0.17, 0.975    # c_g=0.17 (asset-side), H_O=0.975 = the paper's eq:D-def calibration
denom = sigP**2 * T**2 / 2.0                  # = Var-weighting so Γ_struct = -γ·denom

# owners: (name, Γ_table, ω_g from tab:omega, R^bk)
owners = [("CoreWeave", -0.0093, 0.25, 0.35),
          ("Nebius",    -0.0059, 0.14, 0.35),
          ("Lambda",    -0.0119, 0.40, 0.55),
          ("IREN",      -0.0099, 0.005,0.35)]   # IREN ω tiny per tab:omega -> outlier test

print("(1) SPREAD-CONSISTENCY: c'_g that makes γ reproduce each fitted Γ")
print(f"{'owner':10s} {'Γ':>8s} {'ω':>6s} {'γ_need':>7s} {'c′':>7s}")
cs = []
for nm, G, w, R in owners:
    g = -2*G/(sigP**2*T**2)            # γ that yields Γ
    cp = g/w
    tag = "  <-- OUTLIER (ω≈0 cannot make a high-β Γ)" if cp > 1 else ""
    if cp <= 1: cs.append(cp)
    print(f"{nm:10s} {G:8.4f} {w:6.3f} {g:7.4f} {cp:7.3f}{tag}")
cbar = np.mean(cs); csd = np.std(cs)
print(f"  mean c'_g (consistent owners) = {cbar:.3f} ± {csd:.3f}")

print("\n(2) ASSET-SIDE STRUCTURAL: c'_g = (c_g/H_O)·persistence, ceiling at persistence=1")
per_unit = cg/HO
for perm in (0.7, 0.85, 1.0):
    print(f"  persistence={perm:.2f}: (c_g/H_O)·p = {per_unit:.3f}·{perm:.2f} = {per_unit*perm:.3f}")
print(f"  -> ceiling {per_unit:.3f}; route (1) mean = {cbar/per_unit:.2f} x ceiling "
      f"(fast collateral-mark/refi response vs ~1yr rental mean reversion)")

print("\n(3) 2024 CROSS-CHECK: implied spread-widening from the realized-vs-clock gap")
# 2024: H100 ~$2.85 early -> ~$2.0-2.6 committed / ~$1-2 auction (web-verified). Clock drop ~0.23/yr.
for gap, lbl in ((0.18, "committed"), (0.35, "auction/spot")):
    for nm, G, w, R in owners[:1]+[owners[2]]:      # CoreWeave, Lambda
        g = cbar*w
        dlam = g*gap
        dspread = dlam*(1-R)
        print(f"  {lbl:12s} gap={gap:.2f}: {nm:9s} γ={g:.4f} -> Δλ={dlam:.4f} -> Δspread≈{1e4*dspread:.0f}bp")

print("\n(4) RE-ATTRIBUTION of ϱ (structural γ now carries the sign; fitted ϱ demoted)")
print(f"{'owner':10s} {'Γ_tot':>8s} {'Γ_struct(γ)':>12s} {'ϱ_resid':>8s}  (was ϱ=-0.40 high-β / -0.20 low-β)")
for nm, G, w, R in owners:
    if w < 0.01:  # skip IREN outlier
        print(f"{nm:10s} {'--':>8s} {'--':>12s} {'--':>8s}  (ω≈0: Γ cannot be structural; see flag)")
        continue
    g = cbar*w
    Gstruct = -g*denom
    rho0 = -0.40                                        # all three are high-β GPU owners
    sigL = G/(rho0*sigP*np.sqrt(T))                     # implied σ_Λ (positive)
    resid = (G - Gstruct)/(sigL*sigP*np.sqrt(T))        # ϱ_resid after removing the γ part
    print(f"{nm:10s} {G:8.4f} {Gstruct:12.4f} {resid:8.2f}   (σ_Λ={sigL:.3f})")

print(f"\nwall-clock: {time.time()-t0:.3f}s")
