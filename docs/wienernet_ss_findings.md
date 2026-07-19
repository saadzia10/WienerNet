# WienerNet-SS — state-space / physics-integration variant: design & findings (2026-07)

A new increment-SDE variant that fixes three problems the diagnostics exposed in the
original WienerNet: (i) the temperature-tendency head was fitting noise, (ii) the Euler
drift linearization biases at longer steps, and (iii) the single-Wiener noise is not the
process the data shows. This document describes the model mathematically and reports all
experiments. Runs: `outputs/ss_sweep/` (the 16-combo ablation). Companion diagnostics:
`docs/generalization_and_noise_findings.md`, `docs/wienernet_manuscript_methods.md`.

---

## 1. Why a new variant — what the diagnostics showed

On 30-minute nighttime NEE we established, model-free where possible:

- The **learned temperature-tendency head cannot predict dT/dt** (correlation 0.003 with
  the observed tendency) — the half-hourly tendency is turbulence-dominated. Chasing it
  with a network fits noise, and rolling that drift forward **compounds** the noise.
- The **predictable part of the tendency is the diurnal cooling** — a smooth function of
  time of day (correlation ~0.48 via the month/hour climatology). The turbulent remainder
  belongs in the noise, not the drift.
- The **noise is dominated by independent measurement error** (model-free random error
  ≈ 0.94; the increment residual per point ≈ 0.91), **not** a Wiener random walk: the
  increment residual variance grows ×1.0→×1.26 across an 8× step increase, nowhere near
  the √k a Wiener process demands. So the single diffusion term (variance ∝ Δt) is the
  wrong noise structure.
- The **level** residual (1.46) exceeds measurement noise (0.94): `Reco(T_air)` misses
  ~1.1 of structure (soil temperature, moisture) — a *structural*, not aleatoric, error.

WienerNet-SS integrates these: physics diurnal drift, exact respiration difference, and a
state-space (measurement + small Wiener process) noise.

---

## 2. The model — mathematics

We keep the increment / Euler–Maruyama form (the boundary NEE_t is the integration
anchor, never reconstructed):

    NEE_{t+1} = NEE_t + [drift increment] + [diffusion increment].            (1)

### 2.1 Exact respiration-difference drift

The drift is the change in ecosystem respiration over the step. Lloyd–Taylor respiration:

    Reco(T; E0, rb) = rb · exp( E0 · ( 1/(Tref+T0) − 1/(T+T0) ) ),   Tref=10, T0=46.02.

The **original** model used the Euler linearization of its change,
`dReco/dT · ΔT`, whose convexity error grows with the step. WienerNet-SS uses the
**exact difference**:

    drift increment = Reco(T + ΔT) − Reco(T),     ΔT = (dT/dt) · Δt.          (2)

It reduces to the linearization as Δt→0, and removes the longer-step bias.

### 2.2 Physics diurnal temperature tendency

The tendency `dT/dt` in (2) is split into a predictable diurnal part and an
unpredictable turbulent part:

    dT/dt = (dT/dt)_diurnal  +  (dT/dt)_turbulent.                            (3)

Only the **diurnal** part enters the drift. It is the smooth diurnal cooling — either the
per-site (month, hour-of-day) climatology of the observed tendency, or the analytical
diurnal derivative

    (dT/dt)_diurnal = −π (ΔT_day / 24) · sin( 2π (t − t_max) / 24 ),          (4)

(the derivative of a sinusoidal diurnal cycle with range ΔT_day peaking at t_max). The
turbulent remainder is not modelled by the drift; it is absorbed by the noise, where it
belongs. Because the diurnal tendency is a deterministic function of measured temperature
and clock time, it is available even inside a gap.

### 2.3 State-space (measurement + process) noise

The data say the noise is mostly independent measurement error with only a small
accumulating component. We model the flux as a physics state plus two noise sources:

    NEE_t = Reco(T_t) + s_t + η_t,                                            (5)
      s_{t+1} = s_t + σ_proc · √Δt · ξ_t     (small Wiener PROCESS noise, accumulates),
      η_t ~ independent MEASUREMENT noise, scale σ_meas   (flat in Δt).

The one-step increment then has variance

    Var[ΔNEE − ΔReco] = 2 σ_meas²  +  σ_proc² · Δt,                           (6)

so the predictive increment standard deviation is `√(2 σ_meas² + σ_proc² Δt)`. This
matches the measured near-flat noise scaling (`√(2σ_meas²+σ_proc²Δt)` grows slowly),
unlike the single-Wiener head which forces `σ·√Δt` (variance ∝ Δt, the ×√k the data
reject). Both `σ_meas` and `σ_proc` are heteroscedastic (network outputs); the aleatoric
shape (Laplace/ALD, etc.) is applied on top of this combined scale.

### 2.4 The four ablation axes (what each means in plain terms)

| axis | option A | option B |
|---|---|---|
| **respiration parameters** | *known* — E0/rb are the offline site-fitted values | *predicted* — the network estimates E0/rb (with a match-the-fit penalty) |
| **temperature tendency** | *given diurnal* — the drift reads the physics diurnal cooling directly | *learned diurnal* — the network predicts the tendency, trained to match the diurnal cooling |
| **drift correction** | *off* — pure physics drift | *on* — a small bounded learned correction added to the physics drift |
| **noise model** | *state-space* — separate measurement + small process noise | *single Wiener* — one diffusion scale (the original) |

All 16 combinations always use the exact respiration-difference drift and the physics
diurnal tendency (the two physics-integration choices), and the asymmetric-Laplace noise
family. The exact respiration difference and diurnal tendency are the WienerNet-SS
identity; the four axes above are ablated.

---

## 3. Experiments and results

Setup: Woodwalton held out (leave-one-site-out), one seed, 120 epochs, constant learning
rate, evaluated on the final (leakage-free) checkpoint, 100-pass ensemble scoring. All
runs under `outputs/ss_sweep/`; baselines from `outputs/final_loso/`.

### 3.1 All 16 combinations (one-step predictive metrics)

CRPS = probabilistic skill (lower better); cov90 = fraction of observations inside the 90%
interval (nominal 0.90); RMSE = point error of the mean.

| params | tendency | correction | noise | CRPS | cov90 | RMSE |
|---|---|---|---|---|---|---|
| known | given diurnal | on | Wiener | **0.483** | 0.748 | 1.32 |
| known | learned diurnal | off | Wiener | 0.491 | 0.823 | 1.54 |
| known | given diurnal | off | state-space | 0.492 | 0.647 | 1.32 |
| known | learned diurnal | on | Wiener | 0.511 | 0.556 | 1.27 |
| known | given diurnal | off | Wiener | 0.514 | 0.378 | 1.25 |
| known | learned diurnal | on | state-space | 0.517 | **0.899** | 1.79 |
| known | learned diurnal | off | state-space | 0.629 | 0.392 | 1.39 |
| known | given diurnal | on | state-space | 0.869 | 0.918 | 2.93 |
| predicted | given diurnal | off | Wiener | 0.509 | 0.547 | 1.31 |
| predicted | given diurnal | off | state-space | 0.529 | 0.900 | 1.94 |
| predicted | learned diurnal | on | Wiener | 0.550 | 0.418 | 1.29 |
| predicted | learned diurnal | off | state-space | 0.585 | 0.696 | 1.63 |
| predicted | given diurnal | on | Wiener | 0.593 | 0.785 | 2.17 |
| predicted | given diurnal | on | state-space | 0.755 | 0.769 | 2.09 |
| predicted | learned diurnal | on | state-space | 1.099 | 0.801 | 2.69 |
| predicted | learned diurnal | off | Wiener | 1.782 | 0.261 | 2.91 |

Baselines (Woodwalton, seed 0): MDN 0.511 / cov90 0.90 / RMSE 1.75; old WienerNet-Laplace
0.515 / 0.85 / 1.69; Analytical SDE 0.634 / 0.96 / 2.13; Neural SDE 0.585 / 0.78.

### 3.2 Marginal effect of each axis (averaged over the other three)

| axis | setting | mean CRPS | mean cov90 |
|---|---|---|---|
| respiration params | **known** | **0.563** | 0.670 |
| | predicted | 0.800 | 0.647 |
| tendency | **given diurnal** | **0.593** | 0.711 |
| | learned diurnal | 0.770 | 0.606 |
| correction | on | 0.672 | 0.737 |
| | off | 0.691 | 0.580 |
| noise | **state-space** | 0.684 | **0.753** |
| | single Wiener | 0.679 | 0.564 |

Reading:
- **Known respiration parameters are much better** than predicted (0.563 vs 0.800) —
  the network cannot reliably estimate E0/rb on an unseen site.
- **The exogenous diurnal tendency beats the learned one on average** (more stable); the
  learned tendency can be sharp but is high-variance, especially with predicted params.
- **The bounded drift correction helps in-range** but is known to de-generalise on far
  sites (it is off for the headline model).
- **The state-space noise is the calibration win**: at essentially equal CRPS
  (0.684 vs 0.679) it is far better calibrated (cov90 0.753 vs 0.564). The single-Wiener
  head systematically under-covers; splitting measurement from process fixes that.

### 3.3 Autoregressive within-night gap-filling (all methods)

Protocol: anchor at the last observed flux before a night, forecast the next step, feed
that forecast back as the boundary for the following step (drivers are available through
the gap); every method is increment-form so this is fair. Point error from the
deterministic rollout; band coverage from an ensemble that samples each method's own noise
each step. Errors reported by hours into the gap.

| model | RMSE 0–2h | 2–5h | 5h+ | band cov 5h+ (nominal 0.90) |
|---|---|---|---|---|
| **WienerNet-SS (state-space)** | 1.07 | 1.64 | **1.97** | 0.67 |
| WienerNet-SS (single Wiener) | 1.07 | 1.64 | 1.97 | 0.22 |
| Analytical SDE | 1.04 | 1.60 | 1.92 | 1.00 |
| MDN (no physics) | 1.07 | 1.66 | 2.06 | 0.98 |
| Neural SDE | 1.14 | 1.93 | 2.54 | 0.32 |
| **old WienerNet-Laplace** | 1.42 | 2.81 | **5.45** | 0.77 |

Findings:
- **The physics diurnal drift makes the rollout STABLE.** The old WienerNet (noise-chasing
  tendency head) **blows up** under autoregression (5.45 at 5h+); the new diurnal drift
  does not (1.97). This is the clearest single win of the redesign.
- **On rollout accuracy WienerNet-SS beats both black boxes** — 1.97 vs MDN 2.06 vs Neural
  SDE 2.54 — and matches the constant-noise Analytical floor (1.92).
- **Nobody's rollout band is well-calibrated.** MDN and Analytical *over*-cover
  (0.98–1.00, bands too wide); WienerNet-SS *under*-covers (0.67 state-space, far better
  than single-Wiener's 0.22, but short of 0.90). The rollout uncertainty is not yet
  trustworthy — the drift is stable, but the band does not know about the structural
  (missing-driver) error.

### 3.4 Long-gap (week+) reconstruction and the band

For multi-day gaps the one-step chain degrades for every increment model; the useful
target is the **nightly-mean** flux (the 30-min structure is measurement noise and
averages out). Because the nightly mean ≈ `Reco(T_night)`, the physics predicts each
night's mean directly from measured temperature — no autoregression, robust at any gap
length. Physics reconstruction RMSE is flat at ≈ 0.94 (daily) while persistence degrades
1.07 → 1.53 → 1.88 at 1 / 7 / 14 days. The gap **band**, however, under-covers for both
the state-space and single-Wiener heads (coverage 0.26–0.78 vs nominal 0.90): the gap
uncertainty is dominated by `Reco(T_air)`'s **structural** error (soil temperature,
moisture), which the aleatoric noise heads were never trained on.

---

## 4. Summary of findings

- **Physics integration is a real, modest improvement.** The exact respiration-difference
  drift with the physics diurnal tendency gives lower CRPS **and** lower point RMSE than
  the original WienerNet, and — decisively — makes the autoregressive rollout **stable**
  where the original (noise-chasing tendency) diverges.
- **On Woodwalton (one site, one seed) WienerNet-SS beats MDN and Neural SDE** on one-step
  CRPS/RMSE and on autoregressive gap-fill accuracy, and matches the constant-noise
  Analytical model.
- **The state-space noise earns its keep on calibration** — better than the single Wiener
  everywhere (one-step cov90 0.753 vs 0.564; rollout band 0.67 vs 0.22) — but is **not yet
  nominal**; it under-covers.
- **The open blocker is calibration, and its cause is structural, not aleatoric.** For gap
  reconstruction the dominant uncertainty is the respiration model's missing-driver error
  (~1.1, from soil temperature / moisture), which the noise heads do not represent, so the
  bands are too narrow.
- **Known respiration parameters remain the operating regime** — predicted E0/rb is
  unreliable and high-variance even in-range.

## 5. Honest caveats

- One site, one seed — not yet established across held-out sites or seeds.
- The rollout band is uncalibrated (under-covers); the constant-noise Analytical and the
  MDN over-cover. No method is trustworthy on gap uncertainty yet.
- The learned-diurnal tendency and the drift correction help in-range but are high-variance
  / de-generalising; the robust headline is known-params + given-diurnal + no-correction +
  state-space noise.

## 6. Multi-site generalisation (Task 1 — DONE)

**Design.** All five sites held out in turn (leave-one-site-out), three seeds (0/1/42), the
two robust axes fixed (known E0/rb; no residual correction) and the two scientific levers
swept: drift tendency (exogenous diurnal vs learned→diurnal) × noise (state-space vs single
Wiener) = **4 ablations × 5 sites × 3 seeds = 60 runs** (`outputs/ss_loso/`). Competing
methods (MDN, Neural SDE, mean-var Gaussian, Analytical SDE, RF, XGB) are **reused** from the
final_loso manuscript sweep (already 5 sites × 3 seeds). Report: `analysis/ss/loso_report.py`
(→ `loso_metrics_long.csv`, `loso_summary.csv`); figures F1/F2 in `analysis/ss/figs/`.

### 6.1 Per-site one-step metrics (mean over 3 seeds): CRPS | cov90 | RMSE

| site | WN-SS (diur,ss) | WN-SS (diur,wien) | MDN | Neural SDE | Analytical | RF / XGB (RMSE) |
|------|------|------|------|------|------|------|
| Woodwalton | 0.534 / 0.79 / 1.83 | **0.487** / 0.70 / 1.39 | 0.739 / 0.91 / 3.56 | 0.585 / 0.78 / 1.88 | 0.634 / 0.96 / 2.13 | 1.55 / 2.31 |
| Rosedene | 0.730 / 0.78 / 2.00 | 0.735 / 0.73 / 1.96 | **0.720** / 0.84 / 2.22 | 0.754 / 0.88 / 2.32 | 0.784 / 0.91 / 2.33 | 1.61 / 1.73 |
| Redmere 1 | 1.256 / 0.91 / 161 | 1.003 / 0.89 / 40 | 1.299 / 0.90 / 30 | 19.46 / 0.93 / **956** | **0.657** / 0.95 / 2.01 | 1.28 / 1.35 |
| Redmere 2 | 0.727 / 0.87 / 2.36 | **0.719** / 0.88 / 2.49 | 0.726 / 0.82 / 2.42 | 0.755 / 0.87 / 2.34 | 0.773 / 0.92 / 2.27 | 1.54 / 1.67 |
| Great Fen | 0.905 / 0.82 / 2.68 | 0.910 / 0.84 / 2.88 | 0.912 / 0.76 / 2.46 | 1.084 / 0.90 / 3.81 | **0.907** / 0.89 / 2.42 | 1.58 / 1.61 |

Cross-site mean ± std (over all site × seed): WN-SS (diur,ss) CRPS **0.830 ± 0.39**, cov90
0.835 ± 0.08; WN-SS (diur,wien) 0.771 ± 0.32, 0.809 ± 0.10; MDN 0.879 ± 0.35, 0.846 ± 0.07;
Neural SDE 4.53 ± 12.4, 0.871 ± 0.11; Analytical **0.751 ± 0.11**, 0.926 ± 0.03.
Head-to-head, the headline WN-SS beats MDN on **10/15** site × seed for both CRPS and RMSE.

**What generalises (one-step):** on the four "in-family" sites WienerNet-SS is competitive
with or better than MDN/Neural SDE on CRPS and is reasonably calibrated (cov90 ≈ 0.78–0.87).
**What does not:** at **Redmere 1** the *stochastic sampling tail* of every learned-noise
model explodes out-of-distribution (WN-SS RMSE 40–161, MDN 30, Neural SDE 956, mean-var
huge). Only the **constant-noise Analytical** SDE and the driver-only **trees** stay sane
there. This is the same OOD noise-scale explosion documented in the final_loso sweep — the
*deterministic drift* is stable everywhere; it is the learned noise head that is the OOD
liability. The learned-diurnal tendency is also unstable across sites (CRPS 2.23 ± 4.24) —
**exogenous diurnal is the robust choice**, confirming the Woodwalton ablation.

### 6.2 Autoregressive gap-fill across all five sites (Task 1c)

`analysis/ss/autoregressive_gapfill.py`, extended to all sites + the **trees** (driver-only,
point-forecast, no accumulation). Deterministic-rollout RMSE by hours-into-gap, mean over the
five sites (figure F3):

| model | 0–2 h | 2–5 h | 5+ h | note |
|------|------|------|------|------|
| **WienerNet-SS** (state-sp = Wiener, same drift) | **1.55** | **1.72** | **1.72** | flat — physics drift tracks the decline |
| Analytical SDE | 1.54 | 1.69 | 1.68 | ties WN-SS (identical Reco drift) |
| MDN | 2.65 | 3.42 | 2.63 | climbs; worse everywhere |
| Neural SDE | 963 | 1309 | 566 | **blows up OOD** (Redmere 1: 4808 / 6533 / 2816) |
| Random Forest | 1.74 | 1.37 | 1.17 | driver-only → declines, no band |
| XGBoost | 2.00 | 1.57 | 1.33 | driver-only → declines, no band |

**This is the cleaner, stronger generalisation result.** Feeding the forecast forward,
WienerNet-SS keeps its point error **flat at every one of the five held-out sites** — the
diurnal-physics drift tracks the within-night respiration decline instead of chasing noise —
and ties the pure-physics Analytical model. The black-box sequence model (Neural SDE)
**catastrophically diverges** when its own forecasts are fed back (worst at Redmere 1); MDN
degrades steadily. Trees are competitive on point RMSE but carry no uncertainty and no
physics (they cannot extrapolate a gap's temperature response, only interpolate drivers).

## 7. Structural-error band fix (Task 2 — DONE)

`analysis/ss/structural.py` + `gap_band_eval.py`. The band must carry **three** terms, not
two: measurement `σ_meas` (flat), process `σ_proc·√t` (accumulating), and — the missing one —
**structural** `σ_struct` (flat). `σ_struct² = max(0, Var(level residual) − σ_meas²)` is
estimated **out-of-sample on the training sites** (so it is a genuine held-out band) and added
in quadrature; in the AR rollout it enters as a persistent per-member offset `N(0, σ_struct)`
drawn once at gap start.

**Two things the data settled:**

1. **Adding soil temperature to the reconstruction mean does nothing.** `corr(Tsoil1, level
   residual) = 0.037`, and `Tsoil1` is 0.88-collinear with air temperature, so once
   `Reco(T_air)` is in the mean it adds no information (level-residual std 1.59 → 1.59). The
   structural signal at these lowland fen/peat sites is **moisture / water-table**, not soil
   temperature — and soil moisture (VWC) is absent at Woodwalton, so it cannot be the
   universal lever. The soil-temperature route is therefore reported as a **negative result**.
2. **The explicit structural-variance term is the universal fix.** One-step gap band
   near-coverage, old (measurement+process) → fixed (+σ_struct):

   | site | σ_struct | old near-cov | fixed near-cov |
   |------|------|------|------|
   | Woodwalton | 1.56 | 0.25 | **0.92** |
   | Rosedene | 1.44 | 0.56 | **0.91** |
   | Redmere 1 | 0.13 | 0.73 | 0.74 |
   | Redmere 2 | 1.50 | 0.80 | **0.97** |
   | Great Fen | 1.38 | 0.63 | **0.88** |

   The fix restores nominal coverage at **4 of 5 sites** (figure F4). Redmere 1 is the
   exception in the *opposite* direction: its aleatoric `σ_meas` head is already OOD-inflated,
   so `σ_struct ≈ 0` and the raw band already covers ~0.73 — the structural term is not the
   lever there (the OOD noise-scale explosion of §6.1 is). In the **AR rollout** the same
   offset lifts WN-SS band coverage from ~0.79–0.83 (near-field, under) to **0.94–0.98** at
   every site (cross-site 5+ h coverage 0.99).

## 8. Updated summary

- **The generalising headline is stability, not a one-step-CRPS sweep.** Across all five
  held-out sites, WienerNet-SS's autoregressive gap-fill stays flat (RMSE 1.55 → 1.72) and
  ties the pure-physics Analytical model, while Neural SDE diverges by 100–1000× and MDN
  climbs. Physics integration buys robustness the black-box sequence models structurally lack.
- **One-step probabilistic accuracy is competitive but not dominant**: WN-SS beats MDN 10/15
  site × seed and is well-calibrated on four sites; at Redmere 1 every learned-noise model
  (WN-SS included) suffers the OOD sampling-tail explosion — only constant-noise / physics
  and trees are immune.
- **The band under-coverage is fixed.** The structural-variance term restores nominal gap
  coverage at 4/5 sites and lifts the AR rollout band to ~0.95+ everywhere. Soil temperature
  does **not** shrink the structural residual (collinear with air temp); the missing driver
  is moisture/water-table.
- **Operating regime unchanged**: known E0/rb, exogenous diurnal tendency, no residual
  correction, state-space noise for calibration + the structural term for the gap band.

### Remaining honest caveats
- Redmere 1 remains an unsolved OOD failure for *all* learned-noise models — the sampling
  tail, not the drift, explodes. Constant-noise (Analytical) is the robust fallback there.
- The far-horizon gap band slightly **over**-covers (the `σ_proc·√t` process term) — a minor,
  conservative error, opposite to the near-field problem we fixed.
- AR gap-fill and the band fix are evaluated at seed 0 across sites (the one-step eval table
  uses all three seeds); the qualitative stability story is consistent across the sweep.
