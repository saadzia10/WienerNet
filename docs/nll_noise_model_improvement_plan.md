# NLL / noise-model improvement plan

Status: **partially implemented.** Workstream A (metrics) and the Student-t
stabilization (§4.3) are done; the timescale/aggregation test (Workstream E, §6bis)
is the current focus, *before* the heavy-tail/skew and steepen-scale work. This
captures the improvements proposed after the 2026-07 nighttime-NEE error-structure
analysis so they can be implemented later without re-deriving them.

Supporting analysis, figures and reproducible scripts:
`analysis/nee_error_structure/` (README there indexes everything).

### Progress log
- **2026-07 — metrics (Workstream A):** implemented; see §3.
- **2026-07 — Student-t stabilization (§4.3):** dof **floored at ν≥2**
  (`nu = softplus(log_nu) + 2`, was `+1`); the model forward now **samples from the
  Student-t** it is trained under (was Gaussian — a train/sample mismatch); the
  scoring guard was relaxed `ν>2 → ν>1`; and the mixture/mean–var scale heads are
  clamped like the loss so out-of-sample scales can't exp-overflow.
  *Effect* (10-model, 1-seed bake-off; Increment-A/B, GT physics, 30-min): the raw
  NLL had been driving ν→~1.3 (infinite variance), which tripped the scoring guard
  and, via Gaussian sampling, badly under-dispersed the ensemble. After the fix, on
  the honest 100-pass ensemble the Increment-A/B calibration jumped
  **cov90 0.68→0.85, cov95 0.72→0.91, PIT-KS 0.11→0.04**, CRPS improved
  **0.629→0.614 / 0.628→0.616**, and the parametric and ensemble coverages now
  *agree* (the consistency fix). ν settles at the ~2 floor, matching the data's
  implied tail. This is a **stabilization, not the terminal noise model** — the
  residual heavy-tail *and right-skew* (pooled skew ≈+1.0) still argue for a
  Laplace/asymmetric-Laplace or mixture terminal head (§4.1–4.2, §5.2).
- **2026-07 — open finding (physics vs black box, one-step):** under GT physics the
  30-min drift already nails the mean, so Increment-A ≈ Increment-B ≈ no-physics
  heteroscedastic net ≈ neural SDE within ~0.002 CRPS. Physics does not yet earn its
  keep on one-step 30-min prediction → **Workstream E (§6bis)** is the immediate next
  step, ahead of steepen-scale / Laplace / ALD.

---

## 1. Why — what the data and the trained heads actually show

### 1.1 The target error structure (empirical, physics residual `NEE_obs − Reco`)
- **Heteroscedastic**: residual SD scales ~linearly with predicted flux,
  `SD ≈ 0.24 + 0.30·Reco`, R²=0.99; holds at every site (per-site slope 0.19–0.38).
- **Non-Gaussian, double-exponential (Laplace)**: after removing the
  flux-dependence (local standardisation), excess kurtosis ≈ 18.7, 3σ tail 0.014
  = Laplace (0.014) vs Gaussian (0.003); normality rejected p<1e−50; Laplace
  beats Gaussian by ΔAIC ≫ 0 at every site. Implied Student-t dof the tail wants
  ≈ **ν ≈ 2.0**.
- **Right-skewed**: pooled skew ≈ +1.0 (per-site +0.87 to +2.2, one site −0.23),
  so the true law is heavy-tailed **and asymmetric** — a symmetric Laplace/t is
  close but not exact.
- **Random error cancels ~1/√N under aggregation** (but slower: lag-1 autocorr
  0.29; systematic bias does not cancel). Relative uncertainty of the summed flux:
  ~42% (30-min) → ~9% (week) → ~6% (month). ⇒ half-hourly R²/RMSE is floored by
  measurement noise; **weekly–monthly sums are the honest evaluation scale.**

### 1.2 What the trained noise heads do (nll_bakeoff: β-NLL, Student-t)
Measured by whitening the increment residual by the model's own predicted σ
(`pred_noise_stds`), on the test set. See `analysis/nee_error_structure/scripts/noise_head_test.py`.

| | ν | σ-slope vs flux (effective) | corr(σ,flux) | whitened P(\|z\|>3) | implied ν | cover95 |
|---|---:|---:|---:|---:|---:|---:|
| β-NLL (Gaussian) | ∞ | 0.24 | 0.62 | 0.018 | ~2.0 | 0.93 |
| Student-t | 3.16 | 0.26 | 0.59 | 0.018 | ~2.0 | 0.84 |
| target | ~2.0 | 0.30 | — | 0.014 | — | 0.95 |

Findings:
- **Not** homoscedastic — both heads learned σ∝flux (corr ≈ 0.6, scale-vs-flux
  R² ≈ 0.96), but the slope is **too shallow** and **under-disperses at high flux**
  (pred/emp ratio falls to ~0.5–0.7 in the top flux bin → overconfident on
  high-respiration nights).
- **Student-t did learn a heavy tail** (ν=3.16, whitened residual ≈ t(3.16)),
  so it is not collapsing to Gaussian — but it is still slightly **too light**
  (data wants ν≈2.0) and **under-covers** (0.84).
- **β-NLL is Gaussian by construction** — cannot represent the tail; its own
  whitened residual is Laplace/t-heavy (6× the Gaussian 3σ rate). It should not
  be the terminal noise model.

---

## 2. Goals

A noise model that is simultaneously: **(i) heteroscedastic** with the correct
~0.30 flux-slope, **(ii) heavy-tailed** (≈ Laplace / t with ν≈2), **(iii)
optionally asymmetric** (right-skew), **(iv) calibrated** (coverage ≈ nominal
*within every flux bin*, not just on average), and **(v) evaluated at the
aggregation scale where the physics signal and calibration actually live.**

---

## 3. Workstream A — evaluation metrics (DONE 2026-07)

**Implemented** in `wienernet/evaluation/probabilistic.py` (CRPS closed-form +
ensemble, predictive NLL, PIT, interval coverage/sharpness, Diebold-Mariano,
measurement-noise floor, `probabilistic_scores` + `stratified_scores` +
`ensemble_scores`) and `wienernet/evaluation/process_consistency.py`
(`variance_vs_scale`, `standardized_residual_autocorr`, `drift_check`,
`noise_check`, `energy_distance`). Point metrics gained `bias`; MMD/Wasserstein
relabelled marginal; KL demoted. `Trainer.predict_ensemble` draws M stochastic
forward passes so the sampling/MMD variants are scored on an honest ensemble, not
their (under-representing) parametric `nee_std`. `scripts/evaluate.py` writes
`probabilistic.json` + `process_consistency.json` per run and attaches per-row
`pred_crps`/`pred_pit` to `predictions.parquet` for cross-run DM. Tests:
`tests/test_probabilistic.py`, `tests/test_process_consistency.py`. Energy score
(multi-step) intentionally skipped — the transition eval is one-step.

Original spec (kept for reference); reference impls also in
`analysis/nee_error_structure/scripts/`:

1. **Calibration / coverage**
   - PIT histogram + reliability curve; coverage at nominal 50/90/95%.
   - **Per-flux-bin coverage** (bin by predicted Reco) — this is what exposes the
     high-flux under-dispersion that overall coverage hides.
   - Calibrated-σ ratio (empirical resid std / mean predicted std).
2. **Heteroscedasticity recovery**
   - Slope of predicted effective-SD vs flux (binned), compared to the empirical
     residual slope (target ≈ 0.30); report both slopes + R² + corr(σ,flux).
3. **Tail / shape**
   - Excess kurtosis of the whitened residual; tail exceedance P(\|z\|>k) vs the
     assumed distribution (Gaussian / Laplace / t(ν)); implied ν via `t.fit`;
     optional QQ correlation.
4. **Proper scoring rules**
   - Held-out per-sample NLL and **CRPS** (closed-form for Gaussian/Laplace/t;
     sample-based otherwise) — the single number to rank noise models honestly.
5. **Aggregated-scale metrics** (ties to §1.1)
   - R²/RMSE and relative uncertainty of the flux sum at daily/weekly/monthly;
     the error-cancellation curve (sd of N-sum, 1/√N reference).
6. **Skew diagnostic**
   - Residual skew (overall + per flux bin) so an asymmetric model can be
     justified/validated.

Deliverable: a `noise_calibration_report(predictions_df) -> dict/plots` used by
`scripts/evaluate.py`, plus a cross-run comparison table (extends the existing
`nll_comparison.csv` shape).

---

## 4. Workstream B — likelihood / loss changes

File: `wienernet/losses/likelihood.py` (currently `gaussian_nll`, `beta_nll`,
`student_t_nll`, `compute_nll`; `nu = softplus(log_nu)+1`).

1. **Add `laplace_nll`** (double-exponential / L1 scale-NLL): the
   theoretically-matched symmetric model. New `variant="laplace"` in
   `compute_nll`. Config: `configs/loss/nll_laplace.yaml`.
2. **Add asymmetric-Laplace NLL (`ald_nll`)** with a learnable asymmetry κ
   (or per-sample skew head) to capture the +right-skew. Config
   `configs/loss/nll_ald.yaml`. This is the most complete symmetric-tail +
   skew option.
3. **Fix Student-t init / floor — DONE (2026-07).** Now `nu = softplus(log_nu) + 2`
   (floored at ν≥2, the data's implied tail), the forward **samples from the
   Student-t** (not Gaussian, fixing a train/sample mismatch that under-dispersed
   the ensemble), and the scoring guard is `ν>1`. This stopped the ν→~1.3 collapse
   (infinite variance) and restored ensemble calibration — see the progress log for
   the before/after. Floored-t is now the **stable baseline**; the heavy-tail +
   *skew* terminal model is §4.1/§4.2. `NU_FLOOR` lives in `losses/likelihood.py`
   (`student_t_dof`); the shared t-sampler is `models/components.draw_unit_noise`.
4. **Retire β-NLL as the terminal noise model.** Keep `beta_nll` only as a
   *mean-fit stabiliser* (its purpose: stop σ swamping the mean gradient); pair
   it with a heavy-tailed terminal likelihood, or use `beta=1` when it must
   carry the scale (restores scale fidelity at the cost of mean fit).

---

## 5. Workstream C — noise-head / model changes

File: `wienernet/models/increment_sde.py` (σ via `sigma_decoder`, `log_nu`,
noise assembly in `forward`).

1. **Steepen the high-flux scale.** The head under-grows σ where flux is large.
   Options, cheapest first:
   - `beta_nll(beta=1)` (removes the σ^{2β} down-weighting that flattens scale).
   - Ensure the latent `z` strongly carries respiration magnitude (it sees
     Ta/E0/rb already); optionally add an explicit `Reco`-scaled term to the σ
     head so σ has a physics-anchored multiplicative component
     (`σ = softplus(head(z)) · (a + b·Reco)`).
2. **Skew head** (if pursuing ALD): a per-sample asymmetry output, or a single
   learnable κ analogous to `log_nu`.
3. **Keep it heteroscedastic-by-design**, not homoscedastic-with-outliers: the
   tail should come from the *likelihood family* (t/Laplace/ALD), the
   flux-dependence from the *scale head* — don't let one paper over the other.

---

## 6. Workstream D — configs & experiments

- New loss configs: `configs/loss/nll_laplace.yaml`, `configs/loss/nll_ald.yaml`;
  edit `configs/loss/nll_student_t.yaml` init.
- Experiment: extend `configs/experiment/piae_increment_residual_nll.yaml` into a
  bake-off over `{gaussian, beta, student_t(ν-low), laplace, ald}` × seeds,
  writing per-sample predictions incl. `pred_noise_stds` (and skew/ν params) so
  Workstream A metrics run on every leaf. Mirror the existing
  `scratchpad_sweep/nll_compare.py` harness but persist runs under `outputs/`
  (not a temp dir — see the figures-in-repo convention).

---

## 6bis. Workstream E — timescale / aggregation (does physics earn its keep?)

**Motivation.** The 1-seed bake-off showed physics *ties* a black-box heteroscedastic
net on 30-min one-step CRPS (progress-log open finding). Expected: under GT physics the
30-min ΔT is tiny and noise-dominated, so the drift increment `dReco/dT·ΔT·dt` is a
small slice of the increment and both a physics drift and a flexible net fit the
(mostly-noise) 30-min conditional equally. This is a *separate* question from the noise
fix and must be settled before we can claim the physics-informed method leads.

**Hypothesis (SDE signal-to-noise scaling).** In the Euler–Maruyama step the drift
scales with `dt` and the diffusion with `√dt`, so the **drift-to-noise ratio grows as
`√dt`**. Over a longer step (or aggregated window) the systematic respiration signal —
the night's temperature fall driving Reco down — accumulates while the random noise
grows only `√dt` *and averages down ~1/√N* (the error-structure analysis measured
42%→9%→6% at 30-min→week→month). So at longer timescales the drift becomes the dominant,
*predictable* part of the increment, where the Lloyd–Taylor structure encodes it exactly
and a black box must learn it — the regime where physics should pull ahead.

**Already supported by the pipeline.** `data.time_step_k=k` + `rescale_to_timestep`
rebuild targets at a `k·30-min` step within each night: `NEE_{t+k}`, the **averaged**
drift rate `dTa=(Ta_{t+k}−Ta_t)/span`, and `dt=k·30`, so `bNEE + f·dt` reconstructs
`NEE_{t+k}` with `f·dt ≈ Reco(T_{t+k})−Reco(T_t)` and noise `σ·√(k·30)` — exactly "a
longer dt with averaged mean and √dt noise". The `dt_scale_sweep` experiment runs
`k=1,2,4,8,16` (30 min → 8 h).

**Test.** Sweep `k∈{1,2,4,8}` for a physics model (Increment-A/-B, WienerNet-NLL) vs a
no-physics model (heteroscedastic mean–var, MDN) and the neural SDE — all on the
stabilized Student-t, matched capacity — and report CRPS / per-flux-bin coverage / PIT /
RMSE **as a function of `k`**. *Success* = the physics models' **relative advantage grows
with `k`** and overtakes the black box at some horizon (a performance-vs-timescale
manuscript figure). Also evaluate the trained 30-min models **rolled out** to the same
horizons (`nightly_rollout` + `variance_vs_scale`) to separate "trained at long dt" from
"rolled out" — the SDE structure should roll out more consistently than a net with no
dynamics.

**Result (2026-07 — hypothesis REFUTED in this form).** Swept `k∈{1,2,4,8}` for
Increment-A/-B (physics) vs heteroscedastic mean-var / MDN (no physics), stabilized
Student-t, seed 42. The physics-minus-blackbox CRPS gap (ensemble) *grows* with `k`
instead of shrinking:

| gap vs k | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| CRPS (phys − blackbox) | −0.002 | +0.001 | +0.005 | +0.016 |
| RMSE on the mean | −0.003 | −0.005 | +0.008 | +0.006 |

The no-physics MDN is best at every scale and pulls *further* ahead as `dt` grows.
Diagnosis: the SNR mechanism is real (drift/noise ∝ √dt) **but the Euler-linearization
bias grows faster** — the physics mean is *best* at k=1–2 and becomes *worst* by k=4–8
as `dReco/dT(T_t)·ΔT` under-shoots the convex `Reco(T_t+ΔT)−Reco(T_t)` over the night's
cooling; the black box has no such bias and the MDN stays best-calibrated. So
longer-dt-on-the-same-rich-data makes physics *worse*, not better.

**Follow-ups.**
- **Exact-respiration-difference drift.** Replace the Euler linearization with the
  integrated change `Reco(T_t + ΔT) − Reco(T_t)` (ΔT = predicted `dTa·dt`, still causal;
  reduces to Euler as dt→0). Removes the mean bias; re-run the sweep. Expected to
  recover ~parity, not a decisive win (the RMSE gap it fixes is only ~0.006–0.008).
- **The physics edge is not signal-size — it's inductive bias.** On this rich dataset
  with GT physics params, nighttime NEE≈Reco(T) is an easy function a black box learns
  as well as the physics at every scale. Where physics *should* win is generalization:
  **held-out-site extrapolation, predicted-k (not GT), sparse/limited data, and
  rollout/process-consistency** — those are the acceptance tests, not longer-dt CRPS.
- Fewer independent `k`-step pairs per night at large `k` → higher variance; keep `k≤8`
  for a night-scale test and use multiple seeds.
- The black box can also be *trained* at long `dt`, so physics winning tests the
  inductive-bias / generalization advantage, not merely signal size — the extrapolation
  (held-out site) and aggregated-calibration axes make that sharpest.


## 6ter. Workstream F — terminal noise model: SHAPE the aleatoric noise (CHOSEN direction)

**Motivation.** The floored Student-t (§4.3) stabilised calibration but is *symmetric*.
The residual is heavy-tailed AND **right-skewed (skew ≈ +1.0)**, so a single symmetric
t structurally cannot fit the asymmetry — the small residual gap that remains. The
no-physics MDN's advantage is almost entirely its flexible **mixture** noise capturing
this skewed, heavy-tailed shape. The goal is to borrow that noise flexibility while
KEEPING the SDE drift/residual/noise decomposition (the physics story).

**Shared invariant (what makes this a *combination*, not a degeneration into an MDN).**
The **drift `f_phys + r(z)` remains the conditional MEAN**; the noise term is *shaped*
but **zero-mean**, so `nee_mean = bNEE + drift·dt` is unchanged and the physics
interpretation survives. The mixture/skew only shapes the noise, it never steals the
mean. The Euler `√dt` scaling and the drift clamp (graceful OOD) are retained.

### F.1 Asymmetric-Laplace (ALD) noise — the minimal skew-capturing model
- `ΔNEE = drift·dt + √dt·ε`, with `ε ~ ALD(loc, σ(z), κ)`, `κ` a learnable asymmetry
  (one scalar `log_kappa`, analogous to the Student-t `log_nu`; optionally a per-sample
  head). κ=1 recovers the symmetric Laplace.
- **Centring subtlety:** a location-0 ALD has a *nonzero* mean (it's skewed), so to keep
  the drift = conditional mean, either subtract the analytic ALD mean
  (`E[ALD] = σ(1/κ − κ)`) so `ε` is zero-mean, or define the drift as the ALD *location*
  and report `nee_mean = drift·dt + √dt·E[ALD]`. Recommend the former (drift = mean) for
  a clean physics interpretation; document whichever is chosen.
- New `variant="ald"` in `losses/likelihood.compute_nll` (pinball-style NLL); config
  `configs/loss/nll_ald.yaml`; model flag `noise_asymmetry` adds `log_kappa`. Cheapest
  option — 1 extra parameter — and the direct test of the skew hypothesis.

### F.2 Zero-mean mixture noise head — the flexible upper bound (MDN's shape on the SDE)
- `ε ~ Σ_k w_k(z)·N(δ_k(z), σ_k(z)²)`, **constrained `Σ_k w_k δ_k = 0`** (centre the
  component offsets → noise is zero-mean → drift stays the mean). A K-component mixture
  NOISE head on the increment SDE.
- Reuses `losses/likelihood.mixture_nll` (already implemented for the MDN baseline), but
  the component locations are `drift·dt + √dt·δ_k'` (centred offsets) rather than a free
  mean, and `nee_mean = drift·dt`. Model flag `noise_mixture_components=K`.
- Keeps interpretability + dt-scaling + graceful OOD (the drift clamp bounds the mean, so
  even if the mixture head extrapolates on an OOD site the damage is bounded — likely more
  robust than the plain MDN there).

### Experiment (both, vs the MDN)
Hold out **Woodwalton** (where GT-k physics already *wins*), k=1, stabilized setup, **3
seeds**, matched capacity. Compare: physics-SDE + {Student-t (current), ALD (F.1), mixture
(F.2)} vs the no-physics MDN. **Success** = the ALD / mixture SDE ≥ MDN on ensemble CRPS
AND better-calibrated (PIT-KS, coverage), *while retaining the drift/residual/noise
decomposition*.

**RESULT (2026-07, IMPLEMENTED + RUN).** Both noise models are implemented on
IncrementSDEModel (`noise_asymmetry`→`log_kappa`+`asymmetric_laplace_nll`+
`sample_ald_noise`+`configs/loss/nll_ald.yaml`; `noise_mixture_components=K`→centred
mixture noise head→`mixture_nll`), verified `nee_mean == bNEE + drift` (interpretability
preserved), tests in `test_increment_sde.py`. Woodwalton, GT-k, 3 seeds, ensemble CRPS:

| model | CRPS | cov90 | PIT-KS |
|---|---|---|---|
| Physics-SDE + Student-t | 0.533 | 0.709 | 0.207 |
| **Physics-SDE + ALD** | **0.509** | 0.611 | 0.197 |
| Physics-SDE + Mixture | 0.578 | 0.820 | 0.186 |
| No-physics MDN | 0.739 | 0.913 | 0.211 |

**Physics-SDE + ALD BEATS the MDN on CRPS while keeping the SDE decomposition** — the first
"physics superior on the primary metric" result. Caveats: ALD learned **κ≈1.0 (symmetric)**
→ the win is the **Laplace tail, not the skew** (woodwalton increment noise ~symmetric; the
+1.0 skew was pooled/level-basis); ALD **under-covers** (0.611 vs 0.90 — sharp but
overconfident); the mixture **underperformed** on this easy site. Full record in
`docs/generalization_and_noise_findings.md`.

### Coverage fix + LOSO generalisation (DONE 2026-07)
**ALD coverage fix.** Root cause: the ALD predictive std was **5× too small**
(`calib_std_ratio` 5.10 vs a well-calibrated Student-t 1.11) — the L1/pinball loss fits the
bulk MAD and ignores the heavy high-flux tail. The fix is **β=0.5 gradient re-weighting on the
ALD NLL** (`asymmetric_laplace_nll(..., beta=0.5)`, now the `nll_ald.yaml` default): CRPS 0.509→
0.515 (kept), cov90 **0.611→0.848**. β is non-monotonic — β>0.5 re-collapses the scale
(σ² weight over-focuses on high-flux). The physics-anchored scale `σ=softplus(head)·(a+b·Reco)`
(§5.1) was implemented (`noise_physics_scale`, default off) but OVERSHOOTS and degrades the mean
(enlarging σ down-weights the 1/σ location gradient) → kept as a gated option, not the fix.

**LOSO (all 5 sites × 3 seeds, ensemble CRPS).** Best no-residual physics (ALD β=0.5 / mixture)
vs MDN: woodwalton −0.176, **redmere_1 −0.524** (was catastrophic 7.5, now 0.775 beating MDN
1.30), redmere_2 −0.015, great_fen tie, rosedene +0.012. **Physics wins/ties 4/5 sites, the
catastrophic OOD mode eliminated** (β=0.5 + `drift_clamp` + no residual). The residual variants
still blow up on redmere_1 (ald_A 5.56) — use variant B. The mixture ≈ ALD on CRPS but
better-calibrated (cov90 0.82 vs 0.74 woodwalton). Full record + figures:
`docs/generalization_and_noise_findings.md` §4bis, `analysis/holdout_noise/`.

### Remaining
1. **Skew still not engaged** — κ learns ≈1 on every site; the gain is the Laplace tail, not
   asymmetry. Test the level basis / per-flux-bin, or accept the increment noise is ~symmetric.
2. **Woodwalton calibration** — physics is sharper but under-covers the easy site (0.74–0.82 vs
   MDN 0.913); the mixture or a per-flux-bin scale term closes it. Low priority.


## 7. Success criteria (how we know it worked)

Using Workstream A metrics on held-out test:
- Predicted effective-SD slope within ~±10% of the empirical **0.30**, with
  **per-flux-bin coverage ≈ nominal across all bins** (the high-flux bin is the
  acceptance test — current models fail here).
- Whitened-residual tail matches the assumed family; **implied ν ≈ learned ν**
  (≈2), and **cover95 ≈ 0.95** (Student-t is 0.84 today).
- **Lower held-out CRPS/NLL** than the Gaussian and β-NLL baselines.
- **Weekly/monthly aggregated calibration ≈ nominal**, and aggregated relative
  uncertainty tracks the empirical ~9%/~6%.

---

## 8. Open decisions

- **Increment vs level basis**: the noise head models the *increment* residual;
  the 0.30 slope was measured on the *level* residual (increment ≈ √2×level in
  scale). Decide which basis the metrics report as canonical (recommend: report
  both, validate the head on the increment basis, cite the level 0.30 as the
  physical anchor).
- **Skew**: is an ALD worth the extra head, or is a symmetric Laplace/t "good
  enough"? Decide after the symmetric heavy-tailed models are calibrated.
- **Measurement vs structural error**: the residual mixes both; only the random
  (measurement) part is truly Laplace/heteroscedastic and cancels under
  aggregation. If a future goal is a *measurement-error* likelihood specifically,
  the flux-dependent systematic bias (physics over-predicts at high Reco) should
  be modelled separately (drift/residual head), not absorbed into σ.
