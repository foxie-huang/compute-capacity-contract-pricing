#!/usr/bin/env python3
r"""
First-pass STRUCTURAL branching ratio n_S for the 2025-26 Stargate circular-financing
cluster, computed from a FILING-SOURCED exposure network (Gao 2026 "AI's Chokepoints"
DebtRank inputs + issuer filings), instead of the assumed prior n_S = 0.70.

Model object (the paper's eq:n-structural):
    n_S = E_i[ #{ j != i : LGD * E_{j<-i} > B_j } ]
i.e. draw a uniformly-random first defaulter i in the cluster, count how many OTHER
members j take a loss LGD*E_{j<-i} large enough to exhaust their buffer B_j, average over i.
n_S = the mean out-degree of the "fatal knock-on" graph = the Hawkes mean-offspring.

HONEST STATUS: structural FIRST-PASS, not econometric. Exposures/buffers are filing-anchored
point estimates with documented judgment (revenue-at-risk / committed capital vs equity-or-
covenant buffer; LGD from GPU fire-sale severity). Gao's DebtRank distress-SHARE is NOT n_S
(different functional); we use his disclosed exposure INPUTS, not his output. Some of his edge
weights are estimated -> treat the band, not the third digit.
"""
import itertools

# ---- cluster nodes: the OpenAI-centric core financing counterparties (Gao Fig. 5 core) ----
NODES = ["OpenAI", "Oracle", "CoreWeave", "NVIDIA", "SoftBank", "Microsoft"]

# ---- exposure matrix  E[j][i] = loss to j if i defaults, in $B (filing-sourced) ----
# only material entries; blanks are ~0.  Sources in comments.
E = {j: {i: 0.0 for i in NODES} for j in NODES}
E["Oracle"]["OpenAI"]    = 300.0  # ~$300B OpenAI RPO -- WSJ-REPORTED, not Oracle-filed (Oracle discloses only an unnamed $30B/yr contract; the ~57% share holds vs the $523B backlog, ~47% vs the $638B FY26 total)
E["CoreWeave"]["OpenAI"] = 22.4   # OpenAI up-to-$22.4B contracted = ~22.5% of CoreWeave's $99.4B Q1-2026 backlog (derived ratio; per-customer backlog not filed)
E["NVIDIA"]["OpenAI"]    = 40.0   # NVIDIA at-risk of the $100B LOI (disbursed incrementally / near-term GPU sales)
E["SoftBank"]["OpenAI"]  = 40.0   # SoftBank funded/reserve to OpenAI (of $64.6B committed; Gao)
E["Microsoft"]["OpenAI"] = 11.6   # Microsoft funded, equity-method (Gao 6.1)
E["OpenAI"]["NVIDIA"]    = 100.0  # OpenAI loses GPU supply + lead investor if NVIDIA fails (near-impossible branch)
E["Oracle"]["NVIDIA"]    = 20.0   # Oracle loses GPU supply
E["CoreWeave"]["NVIDIA"] = 20.0   # CoreWeave loses GPU supply + offtake backstop
E["OpenAI"]["SoftBank"]  = 40.0   # OpenAI loses the SoftBank reserve
E["OpenAI"]["Microsoft"] = 13.0   # OpenAI loses Azure funding/compute
E["NVIDIA"]["Oracle"]    = 10.0   # NVIDIA loses Oracle GPU sales (absorbed)
E["NVIDIA"]["CoreWeave"] = 10.0   # NVIDIA offtake guarantee + ~11% equity stake (absorbed)

# ---- loss-absorbing buffers  B[j], $B (filing-sourced) ----
B = {
    "NVIDIA":    60.0,  # net-cash (FY26 $62.6B cash+STI vs ~$49.5B total liabilities) + ~$60B/yr FCF -> effectively a floor
    "Oracle":    20.0,  # book equity ~$20-43B (FY25/Q1-FY26); a ~$12B counterparty loss is survivable, so NVIDIA->Oracle is NOT fatal at this buffer -> member n_S=0.67, central 0.53 (PRIMARY, book-equity view). A covenant-stress buffer ~$10B (thin marginal capacity vs $135B debt + neg FCF) instead makes NVIDIA->Oracle fatal -> member 0.83, central 0.62. n_S is BUFFER-DEPENDENT over [0.53, 0.62].
    "CoreWeave": 10.0,  # book equity $4.76B (Q1-2026), thin DSCR headroom. NB Gao's "$17.3B GPU-backed @ ~62% LTV" is CONTESTED: the $17.3B is mostly UNSECURED notes/converts (GPU-secured layer ~$7.5-8.5B), and 62% ties to no filing (covenant LTVs are 65/75%; 62% likely a Microsoft-revenue-share mix-up)
    "OpenAI":    30.0,  # ~$25B year-end-2025 cash (of ~$50B assets) + SoftBank tranches; 2025 cash burn ~$17B rising to ~$45B by 2028 (the "$46B/yr" is a 2028 projection, not a current run-rate)
    "SoftBank":  80.0,  # large NAV, leveraged
    "Microsoft": 100.0, # very large
}

LGD0 = 0.60  # baseline recovery 0.40 (paper's R); GPU fire-sale severity

def n_S(defaulters, LGD, bscale):
    counts = []
    for i in defaulters:
        c = sum(1 for j in NODES if j != i and LGD * E[j][i] > B[j] * bscale)
        counts.append(c)
    return sum(counts) / len(counts), {i: [j for j in NODES if j != i and LGD*E[j][i] > B[j]*bscale]
                                       for i in defaulters}

FIVE = [n for n in NODES if n != "NVIDIA"]   # net-cash NVIDIA excluded as a plausible first defaulter

def grid_nS(lgd_vals, bscale_vals, node_sets):
    """n_S across an input sensitivity box (fire-sale LGD, buffer scale, cluster membership).
    Importable for the calibration figure; the direct-run block below is unchanged."""
    out = []
    for LGD in lgd_vals:
        for bs in bscale_vals:
            for dset in node_sets:
                out.append(n_S(dset, LGD, bs)[0])
    return out


if __name__ == "__main__":
    import time
    _t_wall = time.time()
    # ---- central estimates (two cluster-membership conventions) ----
    c_all, edges_all = n_S(NODES, LGD0, 1.0)
    c_five, _        = n_S(FIVE,  LGD0, 1.0)
    print(f"central n_S  (uniform over all 6, incl. NVIDIA-as-defaulter) = {c_all:.2f}")
    print(f"central n_S  (uniform over 5 capital-constrained, ex-NVIDIA) = {c_five:.2f}")
    print(f"realistic central (small NVIDIA default weight ~ mean)       = {(c_all+c_five)/2:.2f}")

    # ---- fatal knock-on edges under the baseline (should match Gao's top DebtRank edges) ----
    print("\nfatal knock-on edges (baseline LGD=0.6):")
    for i, js in edges_all.items():
        if js: print(f"   {i} default -> fatal to {js}")

    # ---- sensitivity band over LGD, buffer scale, and NVIDIA inclusion ----
    grid = grid_nS((0.40, 0.60, 0.85), (0.7, 1.0, 1.3), (NODES, FIVE))
    print(f"\nsensitivity band over (LGD in .4-.85, buffers +/-30%, +/- NVIDIA): "
          f"[{min(grid):.2f}, {max(grid):.2f}], median {sorted(grid)[len(grid)//2]:.2f}")
    print(f"carried conservative prior 0.67 ; co-verified operating n_0 = 0.64  -> both inside the band: "
          f"{min(grid) <= 0.64 <= max(grid) and min(grid) <= 0.67 <= max(grid)}")
    print(f"[wall {time.time()-_t_wall:.2f}s]")
