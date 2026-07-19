# WienerNet evaluation metrics — reference

Comprehensive description of every metric the evaluation pipeline computes,
written for manuscript use. Ordered **core probabilistic first** (the primary
evaluation axis), then SDE process-consistency, significance/context, marginal
distributional, and point accuracy (secondary). Each entry gives the definition
and formula, inputs, range/orientation, interpretation, what it discriminates,
caveats, and the code location.

---

## 0. Framing and the predictive object being scored

The primary claim of WienerNet is about the **quality of the one-step predictive
distribution over next-step NEE** — the SDE transition
p(NEE_{t+1} | drivers_t, NEE_t) — not point accuracy. Every model that emits a
distribution is scored on the probabilistic metrics; point metrics are reported
only as a secondary sanity axis (the point-optimised RF/XGB regressors are an
upper bound there, so it cannot discriminate the stochastic models).

**The predictive law is a location–scale family**, assembled per test point from
the model outputs:

| symbol | meaning | source |
|---|---|---|
| `y`   | observation, NEE_{t+1} | `gt["nee"]` |
| `μ`   | predictive location = bNEE + drift·dt (deterministic mean) | `preds["nee_mean"]` |
| `s`   | predictive scale = σ·√dt = `exp(nee_log_std)` | `preds["nee_std"]` |
| family | `gaussian` \| `student_t` \| `laplace` | model; Student-t iff a `log_nu` param exists |
| `ν`   | Student-t degrees of freedom = softplus(log_nu)+1 | model `log_nu` |

For **Gaussian** the scale `s` is the standard deviation; for **Student-t** `s`
is the *t-scale* (the SD is `s·√(ν/(ν−2))`), matching how the likelihood loss
parameterises it. Deterministic models (no `nee_std`) are scored as a degenerate
predictive distribution (a point mass at `μ`).

The same contract is emitted by the three **process-comparison baselines** (the
ablation ladder — analytical constant-diffusion SDE, no-physics heteroscedastic
network, neural SDE), so they are scored by the identical functions and sit on the
same axis as the increment and level WienerNet variants. The analytical baseline's
diffusion is a constant σ (one global value or one per site) with a Gaussian law,
so it uses the Gaussian path below; any baseline trained with the Student-t NLL is
scored with the Student-t path.

**Two predictive axes are reported.** (1) *Parametric* — the exact predictive law
above; correct and cheap for the likelihood (Gaussian / β-NLL / Student-t)
variants. (2) *Ensemble* — a family-agnostic predictive distribution built from
`M` repeated stochastic forward passes (`Trainer.predict_ensemble`, default
`M=100`), which resamples the latent and the diffusion noise each pass. The
ensemble is the **honest** axis for the sampling / MMD variants, whose parametric
`nee_std` under-represents their true spread; for the NLL variants it
cross-checks the parametric score. Both are written to `probabilistic.json`
(keys `overall` and `ensemble`).

Code: `wienernet/evaluation/probabilistic.py`,
`wienernet/evaluation/process_consistency.py`, driver
`scripts/evaluate.py::write_distribution_reports`.

---

## 1. Primary probabilistic metrics (headline)

### 1.1 CRPS — Continuous Ranked Probability Score

**Definition.** A strictly proper score for a full predictive distribution `F`
against a scalar observation `y`:

  CRPS(F, y) = ∫_{−∞}^{∞} ( F(x) − 𝟙{x ≥ y} )² dx = E|X − y| − ½ E|X − X′|,

where X, X′ are independent draws from `F`. It has the units of the target
(µmol m⁻² s⁻¹) and rewards **calibration and sharpness jointly**. For a point
forecast it reduces to the absolute error |μ − y|, so deterministic baselines are
scored on the same axis and are automatically penalised for having no spread.

**Closed form (Gaussian).** With z = (y − μ)/σ, Φ/φ the standard normal CDF/PDF:

  CRPS(𝒩(μ,σ), y) = σ · [ z(2Φ(z) − 1) + 2φ(z) − 1/√π ].

**Ensemble estimator.** For `m` sorted samples s_(1) ≤ … ≤ s_(m):

  CRPS ≈ (1/m) Σ_i |s_(i) − y| − (1/m²) Σ_i s_(i) (2i − 1 − m),

the sorted form of E|X−y| − ½E|X−X′| (O(m log m); reduces to |μ−y| for a
degenerate ensemble).

**Non-Gaussian families (Student-t, Laplace) — the CRPS is NOT hard-wired
Gaussian.** `crps(...)` dispatches on the model's predictive family, which
`scripts/evaluate.py` auto-detects: a `log_nu` parameter ⇒ Student-t with
ν = softplus(log_nu)+1, else Gaussian (Laplace is a supported family for the
planned Laplace-NLL models). Gaussian uses the closed form above; **every other
family is scored by drawing an exact-family ensemble** (`sample_predictive`,
default 400 draws — `standard_t(ν)` for Student-t, `laplace` for Laplace) and
applying the ensemble estimator. So a Student-t-trained model — the intended
primary, since the residuals are heavy-tailed — is scored with a genuine Student-t
CRPS, not a Gaussian approximation; likewise its NLL, PIT and coverage use the
Student-t density / CDF / quantiles (§§1.2–1.4). Two practical notes: (i) the
Student-t path currently guards ν > 2 (a finite-variance requirement in
`_check_family`); the empirical residuals favour ν ≈ 2, so a fit whose learned ν
falls to ≈2 would trip this — relax the guard to ν > 1 (CRPS needs only a finite
first moment) before scoring such a model; (ii) the non-Gaussian CRPS is an
*unbiased Monte-Carlo estimate* (400 samples), slightly noisier than the Gaussian
closed form — raise the draw count, or add the Student-t closed form (Jordan et
al. 2019), if bit-exactness is required.

**Range / orientation.** ≥ 0, lower is better; 0 only for a point mass exactly at
`y`. **Reported** per-site and pooled; per-row values stored as `pred_crps` in
`predictions.parquet`.

**What it discriminates.** The single headline number: a model that is sharp but
mis-located, or well-located but over/under-dispersed, is penalised. Unlike RMSE
it credits honest spread.

**Caveats.** Scale-dependent (report alongside the target's own spread; do not
compare across differently-scaled targets). The ensemble estimate has a small
upward bias that vanishes as `m → ∞`; `m=100–400` is adequate here (validated
against the Gaussian closed form to <2 %).

**Code.** `crps` (family dispatch), `crps_gaussian` (closed form), `crps_ensemble`
(estimator), `sample_predictive` (exact-family draws for Student-t / Laplace).

---

### 1.2 Predictive log-score / NLL

**Definition.** Mean negative log predictive density of the observation under the
model's predictive law:

  NLL = − (1/N) Σ_t log f(y_t ; μ_t, s_t, family, ν).

Gaussian, Laplace and Student-t densities are used per the model family (Student-t
uses location `μ`, scale `s`, dof `ν`). A strictly proper *local* score:
it depends only on the density at the realised value, so it rewards putting mass
exactly where observations land and punishes over-confident narrow predictions
very steeply.

**Range / orientation.** Real-valued, lower is better; only distributional models
can report it (undefined for a point forecast).

**What it discriminates.** Directly measures transition-density fit and is the
natural training-aligned score for the likelihood variants. Its steep tail
penalty is why it, unlike CRPS, sharply separates Gaussian from heavy-tailed
noise models when the residuals have heavy tails.

**Caveats.** Unbounded: a single observation far in a too-thin tail can dominate
the average (this sensitivity is the point — it exposes tail misspecification —
but report a robustified/median variant if a few gross outliers distort the
mean). Comparable across models only for the **same** observations and target
scale.

**Code.** `predictive_logpdf` (per-sample), `probabilistic_scores["nll"]`.

---

### 1.3 PIT / rank histogram (calibration shape)

**Definition.** The Probability Integral Transform evaluates the predictive CDF
at the observation, u_t = F_t(y_t). Under a correctly specified model the
{u_t} are i.i.d. Uniform(0,1). For an ensemble the analogue is the observation's
(randomised) **rank** among its `m` samples, normalised to (0,1).

**Summary statistics reported** (`pit_summary`):
- mean — ideal 0.5 (≠0.5 ⇒ location bias),
- variance — ideal 1/12 ≈ 0.0833,
- `ks_uniform` — Kolmogorov–Smirnov distance from Uniform(0,1) (0 = perfectly
  calibrated),
- histogram counts/edges (default 20 bins) for the figure.

**Interpretation of the histogram shape** (the direct picture of whether the
noise magnitude is right):
- **flat** → calibrated;
- **∪-shaped** (var > 1/12) → predictive spread too narrow (under-dispersion,
  over-confident);
- **∩-shaped / dome** (var < 1/12) → spread too wide (over-dispersion);
- **sloped / shifted mean** → systematic location bias.

**Range / orientation.** `ks_uniform` ∈ [0,1], lower better; var → 1/12,
mean → 1/2.

**What it discriminates.** The one plot that shows *how* calibration fails
(too-narrow vs too-wide vs biased), not just that it fails. Complements coverage
(a scalar per level) with the whole distributional shape.

**Caveats.** Ensemble rank PIT is discretised at 1/(m+1); ties are broken by
randomisation. PIT uniformity is necessary but not sufficient for full
calibration (it checks the marginal rank distribution).

**Code.** `predictive_cdf`, `pit_summary`; per-row `pred_pit` in
`predictions.parquet`.

---

### 1.4 Interval coverage + sharpness (calibration magnitude)

**Definition.** For each nominal level L ∈ {0.50, 0.90, 0.95} form the central
predictive interval [Q(α/2), Q(1−α/2)], α = 1−L, from the predictive quantiles
(parametric ppf, or empirical sample quantiles for the ensemble). Report:
- **coverage** = fraction of observations inside the interval (target = L),
- **sharpness** = mean interval width (smaller = more informative).

**Range / orientation.** Coverage ∈ [0,1] should match the nominal L; sharpness ≥
0, smaller is better **conditional on** coverage being met. The joint target is
*nominal coverage at the narrowest width*.

**What it discriminates.** The actionable calibration readout: coverage far below
nominal = under-dispersed / over-confident (the failure mode of the current
noise heads at high flux); above nominal = over-dispersed. Sharpness prevents the
degenerate "cover everything with huge intervals" solution.

**Caveats.** A single level hides shape (pair with PIT). Coverage is a
Bernoulli-averaged rate — report with N; per-flux-bin or per-stratum coverage
(see §3.3) is where high-flux under-dispersion becomes visible.

**Code.** `interval_coverage`, `predictive_interval`; ensemble version inside
`ensemble_scores`.

---

### 1.5 Ensemble scoring (family-agnostic predictive distribution)

**Definition.** Given an (N, m) forecast ensemble (m stochastic forward passes
per time step), compute CRPS (ensemble estimator), a randomised **rank PIT**
u_t = (b_t + U·(e_t+1))/(m+1) — b_t = #samples below y_t, e_t = #ties — and
**empirical** interval coverage/sharpness from sample quantiles. Reduces to
CRPS = |μ − y| and a degenerate PIT for a one-sample (deterministic) ensemble.

**Why.** The sampling/MMD variants realise their predictive distribution through
latent + noise sampling, not a calibrated `nee_std`; scoring their parametric
scale misrepresents them. The ensemble is the honest, model-agnostic predictive
law and lets deterministic, parametric, and sampling models be compared on one
axis.

**Range / orientation.** As per the component scores (CRPS lower better; rank PIT
→ uniform; coverage → nominal).

**Caveats.** Monte-Carlo error ∝ 1/√m in the tails; `m=100` default trades cost
for tail resolution. Requires the model to actually resample per pass (it does —
the reparameterised latent and diffusion `randn` are not gated by eval mode).

**Code.** `ensemble_scores`; `Trainer.predict_ensemble`.

---

## 2. SDE / process-consistency metrics

These test that the *stochastic process* is captured, beyond the marginal
one-step score.

### 2.1 Variance-vs-scale (quadratic-variation / diffusion consistency)

**Definition.** For growing window sizes k (in native 30-min steps) compute the
variance of the k-step increment of the **level** series, Var(X_{t+k} − X_t),
within contiguous nights, for both the observed NEE series and the
model-generated (rolled) series. A Wiener-type diffusion has increment variance
growing ~linearly with the window; a faithful model reproduces the observed
variance-vs-k curve. Report `var_obs`, `var_gen`, and `ratio = var_gen/var_obs`
per k.

**Range / orientation.** `ratio → 1` at every scale = matched diffusion; ratio < 1
= under-dispersed generator (too little diffusion), > 1 = over-dispersed.

**What it discriminates.** Direct evidence the **diffusion term** is right across
timescales — a marginal one-step score can look fine while the multi-step
variance accumulation is wrong.

**Caveats.** Increments are taken only within nights (no cross-night pairs, so
gaps do not contaminate). The "generated" series is the within-night cumulative
sum of predicted increments seeded at the observed boundary; because drift and
diffusion are exogenous there is no train/inference exposure mismatch.

**Code.** `variance_vs_scale` (uses `rollout.nightly_rollout`,
`assign_night_ids`).

---

### 2.2 Standardized-residual autocorrelation (whiteness)

**Definition.** Standardise the one-step residuals, z_t = (y_t − μ_t)/s_t, and
compute their within-night autocorrelation at lags ℓ = 1…L. A well-specified
transition leaves z **white** (ACF ≈ 0 at all lags) and **unit-variance**
(std(z) ≈ 1). Report the ACF table plus the summary {mean_z, std_z}.

**Range / orientation.** ACF ∈ [−1,1] → 0 at all lags; std_z → 1; mean_z → 0.

**What it discriminates.** Leftover autocorrelation signals **drift structure the
model missed** (a systematic component still living in the residual); std_z ≫ 1
signals gross under-dispersion (scale too small), std_z ≪ 1 over-dispersion.

**Caveats.** Computed within nights only. std_z is sensitive to a few points with
near-zero predicted scale (they inflate z); read it together with the coverage /
PIT, which are bounded.

**Code.** `standardized_residual_autocorr`.

---

### 2.3 Component-wise drift check

**Definition.** Aggregate both the observed one-step increment (y_t − NEE_t) and
the model's **deterministic** increment (μ_t − NEE_t = drift·dt [+ residual]) to
per-window means (default window = night), which averages out the measurement
noise to expose the conditional-mean signal, then regress predicted vs observed
window means. Report R² = 1 − SS_res/SS_tot, bias = mean(pred − obs), slope, and
the window count.

**Range / orientation.** R² → 1 and bias → 0 = the deterministic backbone
reproduces the denoised increment; slope → 1.

**What it discriminates.** Validates the **drift** (physics + learned residual)
*away from the measurement-noise floor* — at native 30-min resolution the drift
signal is buried under noise, so a coarse-scale conditional-mean target is the
only fair way to score it.

**Caveats.** R² here is the coefficient of the 1:1 agreement (SS around the
observed window means), not a free-slope fit; a biased-but-linear backbone shows
up as bias ≠ 0 with high correlation.

**Code.** `drift_check`.

---

### 2.4 Component-wise noise check

**Definition.** Compare the **predicted noise** to the **empirical residual**
(observation minus the deterministic prediction, y − μ). Report for both: variance,
skewness, excess kurtosis, and a 3-σ tail fraction; a distributional distance
(**energy distance**, §3.2) between the empirical residual and a one-draw-per-row
sample of the predicted noise; and a **PIT** of the residual under the predictive
noise law (mean 0, per-row scale `s`).

**Range / orientation.** Matching moments, small energy distance, and a uniform
noise-PIT = the stochastic component matches the real noise shape and tails.

**What it discriminates.** Confirms the noise term reproduces the *shape and
tails* of the true residual (heteroscedastic, heavy-tailed, possibly skewed), not
just its variance — the crux of the Gaussian-vs-Laplace/Student-t question.

**Caveats.** The predicted-noise moments use one draw per row (a marginal over the
test set); for a family with heavy tails the sample kurtosis is high-variance —
interpret with the family's theoretical tail, not the sample kurtosis alone.

**Code.** `noise_check`.

---

## 3. Significance and supporting context

### 3.1 Diebold–Mariano test on CRPS

**Definition.** Paired test of equal predictive accuracy between two models using
their per-timestep loss difference d_t = CRPS^A_t − CRPS^B_t. The statistic is

  DM = d̄ / √( V̂(d̄) ) ,  V̂(d̄) = (1/N)[ γ₀ + 2 Σ_{k=1}^{ℓ} (1 − k/(ℓ+1)) γ_k ],

a Newey–West (Bartlett-weighted) long-run variance with ℓ = h−1 lags for an
h-step forecast, times the Harvey–Leybourne–Newbold small-sample correction
√((N+1−2h+h(h−1)/N)/N); the p-value is two-sided from a t_{N−1} distribution.

**Orientation.** DM < 0 with small p ⇒ model A has significantly **lower** loss.

**What it discriminates.** Turns "model A's CRPS is lower" into a p-value,
pre-empting the "difference is within noise" objection. Applied to the per-row
`pred_crps` columns paired across runs on the same test set.

**Caveats.** Requires row-aligned per-timestep losses from the two models (same
test set / order — holds for a shared data config). Use ℓ > 0 (or block bootstrap)
if residuals are autocorrelated; the one-step default is ℓ = 0.

**Code.** `diebold_mariano`.

---

### 3.2 Energy distance (distributional distance helper)

**Definition.** Between two 1-D samples a, b:

  D²(a,b) = 2 E|A − B| − E|A − A′| − E|B − B′| ≥ 0,

zero iff the distributions match. Used inside the noise check and available as a
general two-sample distance (sub-samples to ≤3000 per side for tractability).

**Orientation.** ≥ 0, lower = closer distributions. **Caveat.** Scale-dependent
(units of the target); a robust, kernel-free alternative to MMD.

**Code.** `energy_distance`.

---

### 3.3 Stratified reporting (non-stationarity)

**Definition.** The full probabilistic score set (CRPS, NLL, PIT summary,
coverage/sharpness) recomputed within each level of each stratifying dimension:
**site / management regime**, **season**, and **time-of-night** (coarse hour
bins). Groups with fewer than `min_n` (default 100) rows are skipped.

**What it discriminates.** Shows the model stays calibrated as the dynamics change
— seasonal temperature range, within-night cooling, and site/water-table regime
— rather than being calibrated only on average. Per-stratum coverage is where
regime-specific miscalibration (e.g. high-flux summer under-dispersion) surfaces.

**Caveats.** Per-stratum N shrinks the coverage/PIT precision; report N per cell.

**Code.** `stratified_scores`; grouping assembled in
`scripts/evaluate.py::write_distribution_reports`.

---

### 3.4 Measurement-noise floor (point-error context)

**Definition.** The irreducible point-error floor implied by the
eddy-covariance random-error model σ(flux) = a + b·|flux| (Hollinger &
Richardson): bin the physics residual by |flux| = Reco, fit the per-bin residual
SD linearly in |flux|, then report the RMSE floor √(mean σ²) and the Gaussian MAE
floor mean(σ)·√(2/π), alongside the fitted a, b.

**What it discriminates.** Reported beside RMSE/MAE to show every model sits near
this floor — i.e. point accuracy cannot discriminate the models, motivating the
distributional axis. Also supplies the target heteroscedastic slope `b` the noise
head should reproduce.

**Caveats.** Assumes the residual approximates the random measurement error; it
also contains model structural error, so the floor is a lower bound on achievable
point accuracy, not an exact noise estimate.

**Code.** `measurement_noise_floor`.

---

## 4. Marginal distributional metrics (secondary)

These compare the **pooled one-step marginals** of predictions and observations.
They are blind to per-timestep calibration (a model with the right marginal but
wrong per-point spread scores well), so they are secondary sanity checks, not the
headline.

### 4.1 MMD (Maximum Mean Discrepancy, RBF kernel)

**Definition.** √(E[k(Y,Y′)] + E[k(Ŷ,Ŷ′)] − 2E[k(Y,Ŷ)]) with an RBF kernel
k(a,b)=exp(−γ‖a−b‖²); the kernel-embedding distance between the two marginals
(returns √MMD², clamped at 0 for numerical noise). **Orientation.** ≥ 0, lower =
more similar marginals. **Caveats.** Marginal only; γ- and subsample-dependent
(default γ=1, subsample 2000 for the O(n²) kernel); compare on the **same**
one-step quantity/scale the model predicts. **Code.** `mmd_rbf`.

### 4.2 Wasserstein-1 (earth-mover) distance

**Definition.** W₁(Y, Ŷ) = ∫ |F_Y(x) − F_Ŷ(x)| dx, the minimal transport cost
between the two marginals; bin-free and robust. **Orientation.** ≥ 0, lower
better, in target units. **Caveats.** Marginal only (blind to per-timestep
calibration). **Code.** `wasserstein`.

### 4.3 KL divergence (histogram) — **demoted / appendix**

**Definition.** Discrete KL between histograms of Y and Ŷ over shared bins.
**Status.** Unstable — requires binning a density and swings with bin count and
tail sparsity (the cause of its wild cross-site variation); reported only with an
explicit bin-sensitivity caveat, superseded by Wasserstein/MMD for marginal
fidelity and CRPS/NLL for the predictive distribution. **Code.**
`kl_divergence_histogram`.

---

## 5. Point-accuracy metrics (secondary sanity axis)

Reported on the deterministic prediction `μ` (`nee_mean`), not the stochastic
sample. Because the point-optimised RF/XGB regressors upper-bound this axis and
all models sit near the measurement-noise floor (§3.4), point accuracy is **not**
the discriminating metric — it is a sanity check that the mean is not broken.

| metric | definition | orientation | note |
|---|---|---|---|
| **MAE** | mean\|μ − y\| | ≥0, lower | robust to outliers |
| **RMSE** | √mean(μ − y)² | ≥0, lower | outlier-sensitive; compare to `rmse_floor` |
| **bias** | mean(μ − y) | →0 | signed; >0 = over-prediction on average |
| **R²** | 1 − SS_res/SS_tot | →1 | fraction of variance explained by the mean |

**Code.** `mae`, `rmse`, `bias`, `r2` (`wienernet/evaluation/metrics.py`),
bundled in `MetricBundle` and computed at every temporal resolution
(raw / daily / weekly / monthly / quarterly) via `evaluate_at_resolutions`.

---

## 6. Where the numbers are written

**Every metric is reported at global (pooled) and per-site resolution.** Per run
(`scripts/evaluate.py::evaluate_run`, into `<run>/metrics/`):
- `probabilistic.json` — `{family, nu, n, global{…}, per_site{site → {…}},
  stratified{season|time_of_night → …}}`, where each block `{…}` =
  `{probabilistic{crps,nll,pit,coverage}, ensemble{…}, measurement_noise_floor{…}}`.
  Site is a first-class per-site key; season and time-of-night are additional
  strata for the non-stationarity check.
- `process_consistency.json` — `{family, nu, global{…}, per_site{site → {…}}}`,
  each block = `{residual_autocorrelation{summary,acf}, drift_check, noise_check,
  variance_vs_scale}`.
- `predictions.parquet` — per-row `pred_crps`, `pred_pit` (+ existing gt/pred
  columns) for cross-run **Diebold–Mariano** pairing.
- `calibration.json` — legacy Gaussian z-score coverage (kept; superseded by
  `probabilistic.json`).
- `per_site.csv`, `long.csv`, `summary.csv` — point + marginal metrics at every
  resolution (cross-seed summaries via the reporter).

### 6.0 Aggregated-scale distributional scoring (`aggregated.json`)

`wienernet/evaluation/aggregated.aggregated_scores` groups the test rows by
(site, calendar-window) and scores **window-aggregated NEE** at weekly/monthly
resolution — the scale where the ~40 % half-hourly measurement noise cancels
(~1/√N) and the physics/temperature signal persists. Because the one-step
predictions are teacher-forced, aggregating them isolates (i) systematic bias in
the mean and (ii) whether the model's noise structure predicts the aggregated
uncertainty. Per resolution it writes point skill (RMSE / bias / R²), **ensemble**
CRPS / PIT / coverage on the aggregated quantity (from the same 100-pass ensemble,
aggregated per window), and the relative-uncertainty (error-cancellation) curve
`raw → weekly → monthly`. Emitted to `<run>/metrics/aggregated.json` for **every**
run. *(Empirically: both physics and the MDN reach R²≈0.99 weekly / 0.997 monthly
— aggregation lifts both, it does not by itself create a physics advantage; see
`generalization_and_noise_findings.md`.)*

### 6.1 Metric-illustration figures (emitted for EVERY run, into `<run>/metrics/plots/`)

Manuscript-grade, driven from the JSON reports (so reproducible without
re-inference), with descriptive site and model names — never code identifiers
(`wienernet/evaluation/calibration_plots.py`). These six emit on **every** eval
(no flag); the heavier per-site decomposition plots stay behind `--plots`:
- `calibration_pit_histograms.png` — PIT/rank histograms, global + per-site grid
  (§1.3): flat = calibrated, ∪ = under-dispersed, dome = over-dispersed.
- `calibration_coverage_reliability.png` — empirical vs nominal coverage, global
  emphasised + per-site lines against the 1:1 perfect-calibration diagonal (§1.4).
- `predictive_skill_by_site.png` — CRPS per site (lower better) with the pooled
  reference (§1.1).
- `diffusion_variance_scaling.png` — observed vs model increment variance across
  aggregation windows, with the linear-Wiener reference (§2.1).
- `residual_autocorrelation.png` — standardized-residual ACF vs lag with the 95%
  white-noise band (§2.2).
- `aggregated_skill.png` — the noise-cancellation curve (raw→weekly→monthly) +
  per-resolution aggregated R² and ensemble CRPS (§6.0).

Aggregation scale: because half-hourly relative measurement uncertainty is ~40 %
and the physics/temperature signal accumulates over days–weeks, calibration and
CRPS should be read at the native step **and** at weekly/monthly aggregation
(where relative uncertainty falls to ~6–9 %); `aggregated_scores` (§6.0) and
`evaluate_at_resolutions` provide the temporal buckets.
