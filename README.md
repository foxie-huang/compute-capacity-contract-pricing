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

## Quickstart

```bash
pip install numpy          # scipy optional (used if present, erf fallback otherwise)
python single/run_all.py   # runs all four scripts; exits non-zero if any gate FAILs
```

Total runtime is a few seconds; each script prints per-stage wall-clock.

## What reproduces what

| Script | Paper exhibit(s) |
|---|---|
| `single/clock_curve.py` | Reference curve for the announced futures strips (Table `tab:refcurve`): entry-anchored and mid-2026 state-conditional curves, λ_RP scenario bands, Samuelson σ_f(τ) kit, hedge rescale. **Gate:** reproduces the commodity branch G^g = $1.63 at 24 months. Includes the day-one hook `lambda_rp(tau, f_mkt)` for when a cleared settlement first prints. |
| `single/cohort_pipeline.py` | Per-issuer forwards across the twelve-name cohort (Table `tab:issuer-outputs`) and the executory-recovery refinement (Table `tab:issuer-outputs-A`). **Gate V0:** flat-recovery limit must reproduce the published table to the cent (12/12). Fidelity variants V1–V4 (drifted default density, drift conventions, coupled spot law), the R-sensitivity tornado (flat vs. §365 recovery), and the B.2 Γ-structural / residual-correlation re-attribution. |
| `single/scripts/cohort_rerun_closedform.py` | Standalone closed-form derivation of the §365 assume/reject recovery leg (truncated-lognormal moment + normal-CDF digital) behind `tab:issuer-outputs-A`. |
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
