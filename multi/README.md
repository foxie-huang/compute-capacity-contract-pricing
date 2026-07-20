# Multi-issuer compute-capacity contract pricing — replication code

Replication code for the multi-issuer working paper: *Pricing Portfolios of Compute
Capacity Contracts — Cross-Issuer Dependence and Segment Classification*.

The paper prices a portfolio of compute-capacity contracts whose dominant risk does not
diversify — a shared trigger that defaults a tied group of issuers at once — and bounds
that risk with a cluster-basis good-deal bound.

## Quickstart

```
python run_all.py
```

Runs every script, echoes its output, and checks the paper's published headline numbers
against what the run produces. Exits non-zero if any gate fails. Total wall-clock is a
little under two minutes; `buyer_jump_tip.py` accounts for most of it.

Dependencies: `numpy`, `scipy`, `matplotlib`.

## Script → exhibit map

| script | reproduces |
|---|---|
| `arch_bifurcation.py`    | architecture bifurcation and the first-passage tip law |
| `arch_overall_layer.py`  | overall-mix coexistence layer and the buffer walk |
| `arch_simplex.py`        | lock-in geometry on the family simplex |
| `n_structural_gao.py`    | structural branching ratio from the exposure network |
| `fig_nS_structural.py`   | structural-prior figure |
| `finite_cascade.py`      | exact finite-name cascade; envelope vs operative band multiplier |
| `linearisation_error.py` | linearisation error in the cluster-basis floor |
| `buyer_jump_tip.py`      | compound-Poisson buyer commitments and tip probabilities |
| `tip_identification.py`  | whether the tip probability is identifiable from traded prices |
| `cohort_sensitivity.py`  | 2022 cohort sensitivity and the continuous-channel reattribution |

## Gated numbers

`run_all.py` asserts the following against each run:

| quantity | paper | tolerance |
|---|---|---|
| finite-name cascade, closed-form validation at n=0 | 0.2% | ±0.15pp |
| operative band multiplier at n₀ = 0.64 | ≈1.1 | ±0.06 |
| linearisation error at γ_fs = 0 | 1.03× | ±0.02 |
| linearisation error at its peak | 1.44× | ±0.05 |
| five-year tip probability, no buyer jumps | 8.9% | ±0.5pp |
| five-year tip probability, 5% training-scoped | 15.5% | ±0.8pp |
| recovery-leg coefficient per unit tip probability | 9.3pp of face | ±0.1pp |

Simulations use fixed seeds, so these are reproducible rather than merely close.

## Inputs

`inputs/architecture_calibration.csv` and `inputs/cluster_calibration.csv` carry every
scalar the paper calibrates, one row per value with its unit, whether it is an input or
derived, and its source. Provenance grades in the source column match the paper's own
tags: `[data]` anchored to a cited public figure, `[struct]` a structural normalisation or
stated judgment, `[illus]` illustrative and carried for sign rather than magnitude. To
re-fit on new data, edit these rows.

## Data note

Every input is either a scalar stated in the paper or a figure from a cited public source
(company filings, exchange and index publications, regulatory texts). No licensed or
proprietary series is required to run this code. Two quantities the paper discusses are
deliberately *not* reproduced here because the underlying data is not publicly
available — an accelerator-rental price history and issuer CDS series — and the paper
reports what each would resolve rather than substituting a proxy.

## Relation to the other folders

`single/` holds the single-issuer paper's code. This folder depends on that paper
conceptually but not computationally: it carries its own copies of any shared scalars, and
there are no cross-folder imports in either direction.
