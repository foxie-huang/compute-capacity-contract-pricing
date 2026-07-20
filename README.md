# Compute Capacity Contracts — companion code

Replication code for:

> **Cao, Z. and Huang, S. (2026). "A Defaultable-Commodity Framework for Compute
> Capacity Contracts."** Working paper, available at
> [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6961087); arXiv link to
> be added on posting.

The papers price bilateral compute-capacity contracts as defaultable claims on a
non-storable commodity forward, with one hardware-obsolescence lifecycle driving both
the delivered value and the issuer's default intensity. Everything is closed form under
affine intensity; this repository re-derives every computed exhibit from a small set of
public input scalars.

## Paper series

This is a monorepo: one folder per paper.

| Folder | Paper | Preprint |
|---|---|---|
| [`single/`](single/) | Cao, Z. and Huang, S. (2026). "A Defaultable-Commodity Framework for Compute Capacity Contracts." | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6961087) |
| [`multi/`](multi/) | Cao, Z. and Huang, S. (2026). "Pricing Portfolios of Compute Capacity Contracts: Cross-Issuer Dependence and Segment Classification." | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7005619) |

The multi-issuer paper builds on the single-issuer one, but the code does not: each folder
carries its own inputs and there are no cross-folder imports.

## Quickstart

```bash
pip install numpy          # scipy optional (used if present, erf fallback otherwise)
python single/run_all.py   # runs all four scripts; exits non-zero if any gate FAILs
python multi/run_all.py    # multi-issuer scripts and gates (~2 min, simulation-heavy)
```

`single/` runs in a few seconds; `multi/` takes a little under two minutes because two of
its exhibits are Monte Carlo. Each script prints its own wall-clock.

## What reproduces what

| Script | Paper exhibit(s) |
|---|---|
| `single/clock_curve.py` | Reference curve for the announced futures strips (Table `tab:refcurve`): entry-anchored and mid-2026 state-conditional curves, λ_RP scenario bands, Samuelson σ_f(τ) kit, hedge rescale. **Gate:** reproduces the commodity branch G^g = $1.63 at 24 months. Includes the day-one hook `lambda_rp(tau, f_mkt)` for when a cleared settlement first prints. |
| `single/cohort_pipeline.py` | Per-issuer forwards across the twelve-name cohort (Table `tab:issuer-outputs`) and the executory-recovery refinement (Table `tab:issuer-outputs-A`). **Gate V0:** flat-recovery limit must reproduce the published table to the cent (12/12). Fidelity variants V1–V4 (drifted default density, drift conventions, coupled spot law), the R-sensitivity tornado (flat vs. §365 recovery), and the B.2 Γ-structural / residual-correlation re-attribution. |
| `single/scripts/cohort_rerun_closedform.py` | Standalone closed-form derivation of the §365 assume/reject recovery leg (truncated-lognormal moment + normal-CDF digital) behind `tab:issuer-outputs-A`. |
| `multi/finite_cascade.py` | Exact finite-name funding cascade: the operative band multiplier against the infinite-population envelope, with the shortfall attributed to finite horizon versus saturation. **Gate:** closed-form validation at n=0 to 0.2%, operative multiplier ≈1.1 at n₀=0.64. |
| `multi/linearisation_error.py` | Error in the rank-one cluster-basis floor from random name resolution and state-dependent loss. **Gate:** 1.03× at both ends of the fire-sale sensitivity, 1.44× at its peak. |
| `multi/buyer_jump_tip.py` | Discrete gigawatt-scale buyer commitments as compound-Poisson share jumps, and their effect on tip probabilities. **Gate:** 8.9% five-year tip with no jumps, 15.5% at a 5% training-scoped fraction. |
| `multi/tip_identification.py` | Whether the tip probability is identifiable from traded prices: channel size by tenor, detectability against quoted noise, and the term-structure convexity that separates it. **Gate:** recovery-leg coefficient 9.3pp of face per unit tip probability. |
| `multi/arch_bifurcation.py`, `multi/arch_overall_layer.py`, `multi/arch_simplex.py`, `multi/n_structural_gao.py`, `multi/fig_nS_structural.py`, `multi/cohort_sensitivity.py` | Architecture-layer figures (bifurcation and first-passage law, buffer walk, simplex geometry, structural branching ratio) and the 2022 cohort sensitivity. |
| `single/scripts/calibrate_gamma.py` | The surprise-erosion scale c′_g = 0.148 ± 0.021 (three independent routes: Γ-inversion across seasoned GPU owners, asset-side ceiling, 2022-episode cross-check). |

## Data

There is **no proprietary dataset**. Every input is a public point value quoted, with its
source, in the papers' own tables, and embedded as constants in the scripts. The same
scalars are provided in machine-readable form for audit in
[`single/inputs/issuer_axes.csv`](single/inputs/issuer_axes.csv) (credit quality,
seniority, GPU-residual exposure per issuer — FINRA TRACE / SEC EDGAR / rating actions)
and [`single/inputs/commodity_anchors.csv`](single/inputs/commodity_anchors.csv)
(spot anchor, lifecycle drifts and transitions, volatilities — vendor datasheets, the
Epoch AI performance-per-watt series, marketplace medians as cited). Rental-index
*series* (Silicon Data, SemiAnalysis) are not redistributed here; the papers and CSVs
quote cited point readings only, and users needing the series should obtain them from
the source.

## Layout

```
single/                    the single-issuer paper (this release)
  clock_curve.py           commodity branch: pre-launch reference curve for the strips
  cohort_pipeline.py       credit branch: gated cohort pipeline (V0–V4, tornado, B.2)
  scripts/                 supporting derivations (closed-form §365 leg, γ calibration)
  inputs/                  input scalars as CSV, with sources
  run_all.py               one-command runner; asserts all gates
multi/                     multi-issuer companion paper (to be added when it is posted)
```

The multi-issuer companion paper depends on the single-issuer paper, never the reverse;
its scripts will live in `multi/` with their own copies of the shared input scalars.

## License

MIT (see [LICENSE](LICENSE)). Input facts are public data quoted with citation; see the
papers' data-availability notes.
