# WienerNet‑SS — Evaluation Methodology

*A self‑contained specification of how WienerNet‑SS is evaluated: the two evaluation stages
(distributional scoring of the one‑step predictive law, and autoregressive gap‑filling), what
each metric measures and why it is included, and the theory of the **structural‑variance band**
— a native feature of the physics‑decomposed noise model that makes the gap‑filling uncertainty
honest. This document is source material for the manuscript **Evaluation** section. It specifies
methodology and theory; it deliberately reports no experimental numbers.*

---

## 0. Why two stages, and why distributions

A respiration model is used for two things: (i) as a probabilistic transition model whose
one‑step predictive distribution should be *correct* (calibrated and sharp), and (ii) as a
**gap‑filler** that reconstructs multi‑hour to multi‑day stretches of missing flux with
honest uncertainty. These stress different properties, so we evaluate both:

* **Stage 1 — distributional evaluation** of the one‑step predictive law `p(NEE_{t+1} | x_t)`.
* **Stage 2 — gap‑filling evaluation** of the autoregressive, multi‑step reconstruction and its
  uncertainty band.

The primary axis is **distributional, not point**. Point metrics (RMSE/MAE) are reported but
cannot discriminate good models here, because nighttime NEE has a large **irreducible
measurement‑noise floor**: under the Hollinger–Richardson random‑error model the flux error
scale grows with flux magnitude, `σ(flux) = a + b·|flux|`, giving an RMSE floor
`√(mean σ²)` that *no* point model can beat. We estimate this floor and report it beside
RMSE/MAE precisely to show the point axis is saturated — which is why a model must instead be
judged on the *distribution* it predicts.

Every model that emits a predictive law is scored on the **same** axis. The law is a
location–scale family (`gaussian` / `student_t` / `laplace`) with per‑row location `μ` (the
deterministic prediction `NEE_t + drift·Δt`) and scale `s = σ_eff`; models available only as
samples (repeated latent+noise draws, or the MDN mixture) are scored from an **ensemble**.
Deterministic models are scored too — CRPS reduces to absolute error — so they sit on the same
axis and are automatically penalised for having no spread.

---

## Stage 1 — Distributional evaluation of the one‑step predictive law

### 1.1 CRPS — the headline proper score

The **Continuous Ranked Probability Score** is the primary metric. For a predictive CDF `F` and
observation `y`,

```
CRPS(F, y) = ∫_ℝ ( F(u) − 1{u ≥ y} )² du  =  E|X − y| − ½ E|X − X'| ,   X,X' ~ F  iid .
```

It is a **strictly proper** score (minimised only by the true distribution) in the *units of the
data*, and it rewards calibration and sharpness **jointly**: a forecast is penalised both for
being off‑centre and for being over‑ or under‑dispersed. We use the closed form for a Gaussian
law, `CRPS = σ[ z(2Φ(z)−1) + 2φ(z) − 1/√π ]` with `z = (y−μ)/σ`, and the sorted‑ensemble
estimator `E|X−y| − ½ E|X−X'|` for non‑Gaussian / sample‑only laws. For a degenerate (point)
forecast it collapses to `|μ − y|`, which is why deterministic baselines are comparable. CRPS is
the number we rank models by.

### 1.2 Log‑score / NLL

The mean negative log predictive density `−mean log p(y | μ, s)`. Also strictly proper, but —
unlike CRPS — **unbounded**: it diverges when an observation lands in a near‑zero‑density tail,
so it is dominated by the worst‑covered points. It is reported as a **tail‑sensitivity**
complement to CRPS: a model can have good CRPS yet poor NLL if its tails are too thin (a real
risk for the heavy‑tailed NEE residual), which is exactly the failure the Student‑t / ALD noise
families guard against.

### 1.3 PIT — the calibration test

The **Probability Integral Transform** is `PIT = F(y | μ, s)`, the predictive CDF evaluated at
the observation (rank‑based for ensembles). If the predictive law is correct, `PIT ~ Uniform(0,1)`
— a flat histogram with mean `½` and variance `1/12`. The *shape* of the deviation is
diagnostic:

* **U‑shaped** PIT (variance `> 1/12`) ⇒ observations fall in the tails too often ⇒ the model is
  **under‑dispersed** (bands too narrow) — the characteristic failure of a free noise head
  trained by MSE.
* **Dome‑shaped** PIT (variance `< 1/12`) ⇒ **over‑dispersed** (bands too wide).
* We summarise with the **KS distance** of the PIT from uniform (`ks_uniform`, 0 = perfect) and
  report mean/variance and the histogram.

PIT answers "is the *whole shape* of the predicted distribution right?", where interval
coverage (1.4) answers it only at chosen quantiles.

### 1.4 Interval coverage and sharpness

For nominal levels (50 %, 90 %, 95 %) we report the **empirical coverage** — the fraction of
observations inside the central predictive interval — and the **sharpness**, the mean interval
width. The pair is the operational reading of calibration: a trustworthy gap‑filler needs its
90 % interval to contain ≈ 90 % of observations (coverage ≈ nominal) with the **narrowest**
width that achieves it. Coverage without sharpness is trivial (an infinite band covers
everything); we always report them together, and sharpness is what breaks ties between two
equally‑calibrated models.

### 1.5 Ensemble scoring — the honest axis for sampling models

Some models (the MMD‑trained variant, the mixture/MDN baseline, any sampler) have a parametric
`nee_std` that under‑represents their true spread. For these we draw an ensemble (repeated
latent+noise forward passes) and compute CRPS, a **rank‑based PIT**, and coverage/sharpness
directly from the samples. This scores each model on the distribution it *actually generates*,
not on a parametric summary it may not honour — so no model is flattered or penalised by the
choice of readout.

### 1.6 Diebold–Mariano significance

Because CRPS differences between good models can be small, we test them. The **Diebold–Mariano**
test operates on the *paired, per‑timestep* CRPS series `d_t = CRPS^A_t − CRPS^B_t`, testing
`H₀: E[d] = 0` (equal predictive accuracy) with a Newey–West (HAC) long‑run variance to account
for the serial correlation of nighttime residuals, and the Harvey–Leybourne–Newbold small‑sample
correction. It converts "model A has lower mean CRPS" into "model A is *significantly* better
(p‑value)", which is what a manuscript claim of superiority requires.

### 1.7 Process‑consistency diagnostics — is the *stochastic process* captured?

Scoring the marginal one‑step law is necessary but not sufficient: a model can be marginally
calibrated yet mis‑specify the *process*. Four diagnostics check the SDE structure:

* **Variance‑vs‑scale (quadratic variation).** Within nights, compute `Var(NEE_{t+k} − NEE_t)`
  for growing windows `k`, for both the observed series and the model‑generated (rolled) series.
  This is the empirical **quadratic‑variation / diffusion‑scaling** curve. A single Wiener
  diffusion predicts variance growing linearly with the window; the observed curve reveals
  whether the increment variance actually accumulates that way (it largely does *not* for
  30‑minute NEE — the motivation for the state‑space measurement+process split in the approach).
  We report the generated/observed variance **ratio** per window.
* **Standardised‑residual whiteness.** The standardised one‑step residual `z = (y − μ)/s` should,
  under correct specification, be **white** (autocorrelation ≈ 0 at all within‑night lags) and
  **unit‑variance** (`std z ≈ 1`). Leftover autocorrelation in `z` signals **drift structure the
  model missed** (a mis‑specified conditional mean); `std z ≠ 1` signals a mis‑scaled noise. We
  report the within‑night ACF and `(mean z, std z)`.
* **Drift check.** Aggregate both the observed increment and the model's deterministic increment
  to coarse window means (averaging the measurement noise *out*), and regress predicted‑vs‑observed
  window means (`R²`, bias, slope). This validates the **drift backbone** away from the noise
  floor — the physics drift is a weak per‑step signal that only becomes visible once the noise is
  averaged down.
* **Noise check.** Compare the predicted noise law to the **empirical residual** `y − μ`
  (moments: variance, skew, excess kurtosis, 3‑σ tail mass; an **energy distance**; and a PIT of
  the residual under the zero‑mean, per‑row‑scale predictive law). This confirms the stochastic
  component matches the real noise *shape and tails*, not just its variance — the check that
  justifies the heavy‑tailed, skewed ALD family.

### 1.8 Interpretability diagnostic — the drift/residual/noise clean‑up

A dedicated diagnostic verifies the decomposition the approach claims. With the residual
correction on and the noise forced zero‑mean, the temperature‑dependent physics misfit should
move **out of the noise and into the named residual**:

```
corr(noise, T) → 0    and    noise mean → 0        (the noise is pure aleatoric)
corr(residual, T)  carries the temperature‑correlated misfit .
```

Reporting `corr(noise, T)` vs `corr(residual, T)` (and the noise mean) shows the split is real —
the noise is not silently absorbing structure that belongs to the drift. This is what licenses
reading the three terms as physics / named‑misfit / aleatoric.

### 1.9 Stratified scoring — generalisation and non‑stationarity

All of the above are computed **globally** and **stratified** (per site, per season, per
flux/temperature regime), with a minimum sample count per stratum. Stratification is how we test
that calibration **holds as the dynamics change** — across held‑out sites (leave‑one‑site‑out)
and across seasons — rather than only on average. A model whose global CRPS is good but whose
per‑site calibration collapses on an unseen site is not a trustworthy gap‑filler, and only the
stratified view exposes it.

---

## Stage 2 — Gap‑filling evaluation

### 2.1 The autoregressive protocol

Gap‑filling is the target application: a contiguous stretch of `NEE` is missing while the
**drivers remain measured** through the gap. We evaluate it as the model is actually used —
autoregressively, method‑agnostically:

1. **Anchor** at the last observed `NEE` before the gap.
2. **Forecast** one step ahead with the model, `NEE_{t+1} = NEE_t + drift·Δt + noise` (6 in the
   approach doc).
3. **Feed the forecast forward** as the `NEE_t` input for the next step, using the **real
   drivers** at every step (only `NEE` is synthetic).
4. Repeat to the end of the gap.

Every increment‑form model can be driven this way (each predicts `NEE_{t+1} = NEE_t + …`,
including the mixture/MDN whose component means are `NEE_t + decoder`), so the comparison is
fair across physics and black‑box models. Because WienerNet‑SS's drift and diffusion are
**exogenous** (functions of drivers, not of `NEE_t`), its rollout is a plain cumulative sum of
increments from the anchor — there is no train/inference exposure mismatch. Two horizons are
scored:

* **Within‑night rollout** (short gaps): step‑by‑step to the end of the night, scored by
  hours‑into‑gap.
* **Nightly‑mean reconstruction** (week+ gaps): the forecastable target over long gaps is each
  night's *mean* flux (the 30‑minute structure is measurement noise and averages out), which the
  physics predicts directly from that night's measured temperature via `Reco(T)` — no
  autoregression, so it does not drift with gap length.

### 2.2 Point accuracy vs gap length — the stability test

We report the **deterministic‑rollout RMSE binned by hours‑into‑gap** (0–2 h, 2–5 h, 5+ h). The
diagnostic question is *how the error behaves as the gap grows*:

* a model whose drift **tracks the within‑night respiration decline** keeps RMSE approximately
  **flat** as the gap lengthens (the physics carries the trajectory);
* a persistence‑like or noise‑chasing model's RMSE **climbs**;
* an unstable autoregressive model **diverges** (compounding its own errors).

This is where physics integration pays off, and the binning makes the difference between "stable"
and "diverging" legible rather than hidden in a single averaged RMSE.

### 2.3 The gap‑fill uncertainty band

The rollout also produces an **uncertainty band**: an ensemble of noisy trajectories (each member
feeds its own noisy forecast forward) whose central interval is scored for **coverage by
hours‑into‑gap**. As the gap lengthens the band should widen to keep coverage near nominal — but
*how* it should widen is the subtle part, and is the subject of §2.4.

### 2.4 The structural‑variance band — a native feature of the decomposition

**This is a capability of the physics‑decomposed noise model, not a post‑hoc patch.** It follows
directly from writing the gap‑fill variance in the same measurement/process language as the SDE.

**The reconstruction and its residual.** Over a gap the conditional‑mean flux is reconstructed
from the *measured temperature* as the physics level `m(x_t) = a + b·Reco(T_t)` (the mean rollout
of the exact‑`Reco` drift telescopes to exactly this — temperature is known through the gap). The
observed flux scatters around it, `NEE_t = m(x_t) + η_t`, and the honest question is the variance
of `η_t`.

**Three variance components with three different time‑scalings.** The gap‑fill predictive
variance decomposes as

```
Var[η_t]  =  σ_meas²        (measurement — flat in gap time)
          +  σ_proc² · τ    (process/Wiener — accumulates with elapsed gap time τ)
          +  σ_struct²      (structural — flat in gap time)                          (★)
```

The first two are exactly the state‑space heads from the approach (§3.5): measurement error
(independent per sample, so flat) and the accumulating Wiener innovation (∝ elapsed time). The
**third term is the new one and is essential in the gap regime**:

> **Structural variance `σ_struct²`** is the variance of the part of the true conditional mean
> that the physics level `m(x)` *cannot represent* because `Reco(T_air)` omits drivers of
> respiration — soil temperature, soil moisture, water table. It is **epistemic** (missing‑driver)
> uncertainty, and it is **flat in gap time**: a persistent level offset for a given gap, not a
> random walk.

**Why the aleatoric heads alone under‑cover a gap — and why one‑step scoring never sees it.**
One‑step, the drift anchors on the *true* `NEE_t`, so the missing‑driver signal is carried by the
anchor and is largely invisible: the aleatoric heads `(σ_meas, σ_proc)` are trained on the
conditional spread *given the true previous value*, which does not include `σ_struct`. In a gap
the anchor is gone — the reconstruction leans only on temperature — so the structural error is
**exposed and dominates**. A band built from `σ_meas` and `σ_proc` alone is therefore
structurally too narrow over a gap, no matter how well the aleatoric heads are calibrated
one‑step. Term (★) is what the model must add.

**Estimation — out‑of‑sample, so the band is honest.** `σ_struct` is estimated from the
**level residual** of the reconstruction, with the measurement contribution removed so it is not
double‑counted:

```
σ_struct²  =  max( 0,  Var[ NEE − m(x) ]  −  σ_meas² ) ,
```

computed on the **training sites** (the sites the model saw) and applied to the held‑out site, so
the reported band is a genuine out‑of‑sample uncertainty rather than a quantity fit to the rows
being scored. In the autoregressive rollout it is realised as a **persistent per‑member offset**
`~ N(0, σ_struct)` drawn once at the gap start (flat across the gap), added on top of the
per‑step aleatoric spread; equivalently, the one‑shot reconstruction band is
`√(σ_meas² + σ_struct² + σ_proc²·τ)`.

**Two physical levers on the same term.** Because `σ_struct` is defined through the level
residual, it exposes two complementary ways to improve the gap band, both physically meaningful:
(a) **shrink** it by adding measured drivers (soil temperature/moisture, water table) to the
respiration mean `m(x)`, reducing the missing‑driver residual; and (b) **represent** whatever
remains by carrying the `σ_struct` term. This is why (★) is a *feature*: only a model that
reconstructs an explicit physics level can define a level residual, name its missing‑driver
component, separate it from measurement and process variance by their different time‑scalings,
and estimate it out‑of‑sample. A black‑box forecaster has no reconstruction level against which
a structural residual is even defined, so it cannot express this uncertainty — it can only
inflate a single opaque spread.

**What we report.** For the gap band we report coverage **near** the gap start (first night) and
**far** (last day), with and without the structural term, at several gap lengths. The reading is
that the aleatoric‑only band is nominal one‑step but under‑covers over a gap, and that adding the
structural term restores coverage toward nominal across gap lengths — with the process term
(`σ_proc²·τ`) making the far‑horizon band slightly conservative, which is the safe direction.

---

## 3. Summary — the evaluation contract

| stage | question | primary metrics |
|---|---|---|
| point (context) | is the point axis saturated? | RMSE/MAE vs the **measurement‑noise floor** |
| distributional | is the one‑step predictive law right? | **CRPS** (rank), NLL (tails), **PIT/KS** (calibration shape), coverage+**sharpness** |
| process | is the *SDE process* captured? | variance‑vs‑scale, standardised‑residual whiteness, drift‑check, noise‑check |
| interpretability | is the decomposition real? | `corr(noise,T)→0` vs `corr(residual,T)` |
| generalisation | does calibration hold off‑distribution? | all of the above **stratified** per site (LOSO) / season |
| gap‑filling | is the reconstruction stable and honestly bounded? | RMSE **vs hours‑into‑gap** (stability); band **coverage** with the **structural‑variance** term (★) |

A model passes only if it is calibrated (PIT/coverage) *and* sharp (CRPS/sharpness) *and*
process‑consistent, holds these under leave‑one‑site‑out stratification, and — in the gap regime
— stays stable with a band that covers because it accounts for measurement, process **and**
structural variance.
