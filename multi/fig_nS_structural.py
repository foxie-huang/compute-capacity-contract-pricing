#!/usr/bin/env python3
r"""
Filing-anchored calibration figure for the funding-cascade branching ratio n_S,
replacing the synthetic MCMC illustration. ALL inputs are the documented exposure
network of n_structural_gao.py (Gao 2026 filings) --- NO synthetic data.

Single panel: the structural n_S = mean fatal-knock-on count of eq:n-structural,
evaluated across the documented input box (fire-sale LGD in [0.40,0.85], buffers
+/-30%, and NVIDIA read as an external floor vs a circular member). It is a discrete,
bimodal object on the two eq:ns-gao membership conventions: floor endpoint 0.40,
member endpoint 0.67 at Oracle's book-equity buffer (0.83 only under the
covenant-stress buffer), central ~0.53 (carried conservative 0.67; covenant-stress
central 0.62).

(The former companion panel drawing the band multiplier 1/(1-n_S) was dropped
2026-07-16: the elementary map's numbers -- [0.40,0.67] -> [1.7,3.0], 6.0 under
covenant stress -- are stated in the figure caption and the Section 8 prose, and the
same map is the fig:arch-cotip colorbar.)

Reproducible; mirrors n_structural_gao.py.
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from n_structural_gao import NODES, FIVE, LGD0, n_S, grid_nS, B

t_wall = time.time()

# ---- structural n_S across the documented input box (real, filing-anchored) ----
LGD = np.linspace(0.40, 0.85, 16)
BS  = np.linspace(0.70, 1.30, 16)
grid = np.array(grid_nS(LGD, BS, (NODES, FIVE)))

n_floor  = n_S(FIVE,  LGD0, 1.0)[0]      # 0.40 : NVIDIA as external floor  (eq:ns-gao)
n_member = n_S(NODES, LGD0, 1.0)[0]      # 0.67 : NVIDIA member, book-equity Oracle buffer (PRIMARY)
n_cent   = 0.5 * (n_floor + n_member)    # 0.53
n0, nS_carried = 0.64, 0.67
# buffer-dependent upper endpoint: a covenant-stress Oracle buffer makes NVIDIA->Oracle fatal (grid above already built with book-equity B)
B["Oracle"] = 10.0
n_member_stress = n_S(NODES, LGD0, 1.0)[0]   # 0.83
B["Oracle"] = 20.0                           # restore book-equity primary

vals, cnt = np.unique(np.round(grid, 3), return_counts=True)
freq = cnt / cnt.sum()

fig, ax = plt.subplots(figsize=(5.0, 3.3))

# ---- discrete structural distribution ----
modes = {round(n_floor, 3), round(n_member, 3)}
colors = ["tab:orange" if round(v, 3) in modes else "0.7" for v in vals]
ax.bar(vals, freq, width=0.028, color=colors, edgecolor="k", linewidth=0.4, zorder=3)
ax.axvspan(n_floor, n_member, color="tab:orange", alpha=0.08, zorder=0)
ax.axvline(n_cent, color="k", lw=1.5, zorder=4, label=fr"central $n_S={n_cent:.2f}$")
ax.axvline(n0, color="tab:green", ls="--", lw=1.2, zorder=4, label=fr"operating $n_0={n0:.2f}$")
ax.axvline(nS_carried, color="tab:blue", ls=":", lw=1.4, zorder=4, label=fr"carried $n_S={nS_carried:.2f}$")
ax.annotate("NVIDIA\nfloor", xy=(n_floor, freq[list(np.round(vals,3)).index(round(n_floor,3))]),
            xytext=(0.30, 0.24), fontsize=7, ha="center", color="tab:orange")
ax.annotate("NVIDIA\nmember", xy=(n_member, 0.24), xytext=(0.67, 0.26), fontsize=7,
            ha="center", color="tab:orange")
ax.axvline(n_member_stress, color="tab:orange", ls="--", lw=1.0, alpha=0.7, zorder=4)
ax.annotate("covenant-stress\nOracle buffer", xy=(n_member_stress, 0.20), xytext=(0.86, 0.175),
            fontsize=6.2, ha="center", color="tab:orange")
ax.annotate("market-implied\nread: lower", xy=(0.34, 0.10), xytext=(0.50, 0.135),
            fontsize=6.8, ha="left", va="center", color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.0))
ax.set_xlim(0.12, 1.05)
ax.set_xlabel(r"structural branching ratio $n_S$ (filing knock-on count)")
ax.set_ylabel("fraction of input box")
ax.legend(fontsize=7, loc="upper center")
ax.set_title("structural calibration from filings", fontsize=10)

plt.tight_layout()
plt.savefig("fig_nS_structural.pdf")
plt.savefig("fig_nS_structural.png", dpi=130)
plt.close()

# ---- summary ----
print(f"n_floor={n_floor:.2f}  n_member={n_member:.2f}  central={n_cent:.2f}")
print("discrete n_S distribution over the input box:")
for v, f in zip(vals, freq):
    tag = "  <-- eq:ns-gao mode" if round(v, 3) in modes else ""
    print(f"   n_S={v:.3f}: {f*100:5.1f}%{tag}")
print(f"band multiplier at [{n_floor:.2f},{n_member:.2f}] = "
      f"[{1/(1-n_floor):.2f}, {1/(1-n_member):.2f}], central {1/(1-n_cent):.2f}")
print(f"wrote fig_nS_structural.pdf / .png   [wall {time.time()-t_wall:.2f}s]")
