# WienerNet‑SS — Approach & Method

*A self‑contained specification of the WienerNet‑SS model: the generative process, the
respiration physics, the drift/residual/noise decomposition, the two interchangeable noise
laws (single‑Wiener and state‑space), the likelihood families, and the exact quantity each
learnable component is anchored against in training. This document is the source material for
the manuscript **Approach / Methods** section. It deliberately contains **no experimental
results** — only the model and its justification.*

---

## 1. Modelling philosophy

Nighttime net ecosystem CO₂ exchange (NEE) at a flux tower is, to first order, ecosystem
respiration `Reco`, a smooth, temperature‑driven quantity, observed every 30 minutes through
a heavy, skewed, heteroscedastic measurement/turbulence noise. WienerNet‑SS models the
**one‑step transition** of NEE as a physics‑informed stochastic differential equation (SDE)
discretised by Euler–Maruyama, so that three scientifically distinct quantities are
represented by three separate, individually interpretable terms:

1. a **physics drift** — the analytic change in respiration implied by the Lloyd–Taylor model
   as temperature evolves over the step;
2. an optional **residual drift correction** — a learned, bounded term that absorbs *systematic*
   departures of the true conditional mean from the Lloyd–Taylor physics (missing‑process
   structure), kept small by construction so the physics explains first;
3. a **stochastic noise term** — the genuinely aleatoric part (measurement + turbulence),
   modelled either as a single Wiener diffusion or as a **state‑space** measurement+process
   decomposition, with a calibrated, heavy‑tailed, skewed predictive law.

The design goal is an **interpretable** predictive distribution: the mean is physics plus a
named correction, and the spread is a calibrated aleatoric law — never a black box whose
"uncertainty" is an unfalsifiable by‑product. Every learnable piece is anchored to a
physically meaningful target (§6), which is what lets the decomposition be read scientifically.

---

## 2. Problem setup and notation

Let `x_t` be the vector of exogenous drivers at time `t` (air temperature `T_t`, radiation,
humidity/VPD, friction velocity `u*`, soil temperature, etc.), `Δt` the time to the next
sample (minutes; per‑row, so the model composes across aggregation scales), and `NEE_t` the
observed flux. The learning target is the next observation `NEE_{t+1}` (equivalently the
increment `ΔNEE_t = NEE_{t+1} − NEE_t`).

Respiration follows the **Lloyd–Taylor** model in the **Reichstein (2005)** parameterisation:

```
Reco(T; E0, rb) = rb · exp( E0 · ( 1/(Tref + T0) − 1/(T + T0) ) )              (1)
```

with reference temperature `Tref = 10 °C` and `T0 = 46.02` used as `(T + T0)` in the
denominator (the codebase stores `T0` positive; this is the `T0 = −46.02 °C` convention). `E0`
(activation energy) and `rb` (base respiration) are the two site/period parameters. During the
night GPP ≈ 0, so `NEE ≈ Reco` and the *conditional mean transition of NEE is the transition
of Reco*. The analytic temperature sensitivity is

```
dReco/dT = rb · exp( E0·(1/(Tref+T0) − 1/(T+T0)) ) · E0 / (T + T0)^2 .          (2)
```

Throughout, `E0, rb` are supplied as **known parameters** (fit offline per site/period by
REddyProc‑style estimation) — the primary operating regime — although the model can also
predict them from a head (an ablation).

---

## 3. The generative model

### 3.1 Encoder and heads

A shared encoder maps the **exogenous** drivers to a latent code

```
z = Encoder(x_t) .
```

`NEE_t` is deliberately **excluded** from the encoder input: the boundary value enters only as
the integration anchor (§3.6), never as a feature, so there is no teacher‑forcing / exposure
bias between training (true `NEE_t`) and autoregressive inference (forecast `NEE_t`). From `z`
the model reads a small number of heads, each of which is separately supervised (§6):

| head | symbol | role |
|---|---|---|
| tendency | `g_φ(z)` → `(dT/dt)ᵖ` | predicted temperature tendency (the drift's `dT/dt`) |
| residual | `r_φ(z)` | optional drift‑misfit correction (per‑minute rate) |
| measurement scale | `s_meas(z)` | aleatoric noise scale (single‑Wiener σ, or state‑space measurement σ) |
| process scale | `s_proc(z)` | state‑space Wiener‑process scale (state‑space noise only) |
| (parameters) | `(E0,rb)ᵖ(z)` | optional; inert when `E0,rb` are known |

Scales are mapped through `softplus(·)` (strictly positive) with a small floor.

### 3.2 Physics drift — the exact respiration difference

Given the tendency `(dT/dt)` (its source is §3.3), the temperature change over the step is
`ΔT = (dT/dt)·Δt`. WienerNet‑SS uses the **exact respiration difference** as the drift, rather
than the Euler linearisation `dReco/dT · ΔT`:

```
f_phys = [ Reco(T + ΔT; E0, rb) − Reco(T; E0, rb) ] / Δt        (per‑minute rate)   (3)
```

Because `Reco` is convex in `T`, the linearisation `dReco/dT·(dT/dt)` (2) systematically
under/over‑shoots as `Δt` grows; the exact difference (3) removes that convexity bias and
reduces to (2) as `Δt → 0`. Dividing by `Δt` expresses `f_phys` as a per‑minute *rate*, so the
same clamp/residual/integration machinery (§3.4–3.6) is agnostic to the step size.

### 3.3 Temperature tendency — the diurnal‑physics decomposition

The drift needs `dT/dt`. The observed 30‑minute temperature change is dominated by
unpredictable turbulence; only its **diurnal** component is forecastable. WienerNet‑SS
therefore separates the tendency into a predictable diurnal part (which drives the drift) and a
turbulent part (which belongs in the noise). Three sources are supported; the **primary
approach uses the learned‑diurnal head**:

* **learned‑diurnal (primary):** a head predicts the tendency, `(dT/dt)ᵖ = g_φ(z)`, and is
  MSE‑anchored (§6) to the **physics diurnal tendency** `Δ̄T(site, month, hour)` — the
  per‑site month‑by‑hour climatology of the observed temperature change. The head thus learns
  to *predict* the smooth, driver‑conditioned diurnal tendency rather than memorising a
  lookup, while the physics prior keeps the prediction on the forecastable manifold.
* **exogenous diurnal (ablation):** feed the climatology `Δ̄T` directly as the tendency (no
  head) — a pure lookup, retained only for comparison.
* **learned‑observed (legacy):** anchor the head to the raw observed `dTa` — this makes the
  head chase turbulence and is not used in the primary model.

The diurnal climatology can equivalently be written analytically as a sinusoid,
`Δ̄T(t) ≈ −π (A_day/24)·sin(2π (t − t_max)/24)` with `A_day` the diurnal amplitude and `t_max`
the time of the daily maximum; the empirical month×hour climatology is the nonparametric form
actually used.

### 3.4 Residual drift correction (on/off ablation — part of the approach)

Lloyd–Taylor at air temperature is an approximation: real respiration also responds to soil
temperature, moisture and water table. WienerNet‑SS can add a learned correction to the drift,

```
drift = f_phys + r_φ(z)         (residual ON)   or   drift = f_phys   (residual OFF)   (4)
```

`r_φ(z)` is a per‑minute rate. It is **anchored toward zero** by an L2 penalty (§6), i.e. it is
*physics‑first*: the Lloyd–Taylor drift must explain the bulk of the transition, and `r_φ`
only absorbs the **systematic**, temperature‑correlated misfit the physics cannot. Whether the
correction helps is site‑dependent, so both `residual ON` and `residual OFF` are part of the
approach and reported as an ablation. (When `r_φ` is on and the noise is forced zero‑mean, the
temperature‑dependent misfit provably moves *out* of the noise and *into* `r_φ`: the diagnostic
`corr(noise, T) → 0` while `corr(r_φ, T)` carries the misfit — §6, and the evaluation doc.)

**Graceful‑OOD drift clamp.** The combined drift rate is soft‑bounded,
`drift ← c·tanh(drift / c)` with `c = 1 °(flux)·min⁻¹`. For real drift (`≈ ±0.17` per minute)
this is ≈ identity, but it saturates rather than letting a head that extrapolates to garbage on
a far‑out‑of‑distribution site drive the *exponential* Lloyd–Taylor term to a catastrophic
prediction. It degrades an OOD failure to a bounded increment instead of a blow‑up.

### 3.5 Noise — two interchangeable laws (both part of the approach)

The diffusion term carries the aleatoric spread. WienerNet‑SS supports two structurally
different noise laws; **both are primary** and are reported for the NEE problem because they
answer different questions.

**(a) Single‑Wiener diffusion.** One scale `σ(z) = s_meas(z)`. The increment noise is
`σ · √Δt`, i.e. the classical Brownian assumption that increment variance grows linearly with
the step:

```
Var[noise increment] = σ(z)^2 · Δt .                                            (5a)
```

**(b) State‑space (measurement + process).** The observed increment mixes two physically
distinct sources: independent **measurement** error on each of the two endpoints, and a genuine
**process** (Wiener) innovation accumulated over the step. Modelling them separately gives the
increment variance

```
Var[noise increment] = 2·σ_meas(z)^2 + σ_proc(z)^2 · Δt .                       (5b)
```

The factor **2** is exact under the standard eddy‑covariance random‑error model
(Hollinger–Richardson): each of `NEE_t`, `NEE_{t+1}` carries independent measurement error of
scale `σ_meas`, so their difference has variance `σ_meas² + σ_meas² = 2σ_meas²`. This term is
**flat in `Δt`** (measurement error does not accumulate); the process term is **∝ `Δt`** (a
true random walk). The split is identified by training across **varying per‑row `Δt`** (the
per‑row `Δt` column, exercised by the dt‑aggregation design): a flat‑plus‑linear variance‑vs‑`Δt`
law separates the two components, where a single Wiener head (5a) is forced to fit a pure line
through the origin. Physically, (5b) says most of the 30‑minute increment "noise" is
*independent measurement error*, not accumulating diffusion — which the state‑space form can
express and the single Wiener form cannot.

This same measurement/process decomposition — a *flat* term and a *`Δt`‑accumulating* term, each
with a distinct physical meaning — is what later lets the gap‑filling procedure add a third,
*structural* variance term with its own time‑scaling, giving an honest multi‑day gap band. That
extension is an evaluation‑/application‑time construction (it estimates the structural component
from a held‑out reconstruction residual, not a trained head), so its full theory lives in the
evaluation methodology rather than here; the model‑side enabler is exactly the decomposition (5b).

Both laws feed a common predictive scale used by the likelihood (§4): with `σ_eff` the noise
**increment** standard deviation,

```
single‑Wiener :  σ_eff = σ(z)·√Δt          (equivalently  log σ_eff = log σ + ½ log Δt)
state‑space   :  σ_eff = √( 2 σ_meas^2 + σ_proc^2 · Δt )        (already an increment std)
```

### 3.6 The Euler–Maruyama transition and the predictive law

Collecting drift (4) and noise (5), one Euler–Maruyama step from the observed boundary is

```
NEE_{t+1} = NEE_t + drift·Δt + ε·σ_eff ,        ε ~ (zero‑mean unit law, §4)     (6)
```

(the `√Δt` is inside `σ_eff`). The **conditional (deterministic) mean** and **conditional
scale** on the `NEE_{t+1}` target scale are therefore

```
μ(x_t)      = NEE_t + drift·Δt = NEE_t + (f_phys + r_φ)·Δt        (no noise)      (7)
s(x_t)      = σ_eff                                                              (8)
```

The noise is **zero‑mean** in the primary model, so the drift *is* the conditional mean and the
noise is pure spread — the mean and the scale are the two orthogonal sufficient statistics the
likelihood consumes (§4). `μ` and `s` are exactly the quantities scored by the evaluation
distributional metrics; the full ensemble predictive law is obtained by sampling `ε` (§4) and
integrating (6). Because drift and diffusion are **exogenous** (functions of `x_t`, not of
`NEE_t`), the multi‑step / gap‑filling rollout is a plain cumulative sum of increments seeded at
the boundary, with no train/inference exposure mismatch (see the evaluation doc).

### 3.7 Robustness by construction — the physics‑anchored mean

A direct, **a‑priori** consequence of the physics drift is that WienerNet‑SS's *conditional
mean* is structurally insensitive to distribution shift or sensor corruption in the driver set.
From (7) the conditional mean is

```
μ(x_t) = NEE_t + [ Reco(T + ΔT; E0, rb) − Reco(T; E0, rb) ]  ( + r_φ·Δt ) ,
```

which depends on the drivers **only** through (i) the air temperature `T`, (ii) the known (or
REddyProc‑anchored) parameters `E0, rb`, and (iii) the bounded diurnal tendency `ΔT`. It does
**not** depend on the remaining exogenous drivers — the turbulence, energy‑balance and humidity
channels — except through the smooth, MSE‑anchored tendency head. A large distribution shift, or
an outright sensor fault, in any **non‑temperature** driver therefore **cannot move the
conditional mean**: the physics carries the point prediction. This is a structural guarantee that
the black‑box mean — a free function of *all* drivers — does not have, and it is the mechanism by
which the model's point forecast stays stable on a site that is grossly out‑of‑distribution in a
non‑physics input.

Two bounded safeguards reinforce it:

* the **drift clamp** `drift ← c·tanh(drift/c)` (§3.4) caps the per‑step drift rate, so even a
  tendency or residual head that extrapolates on an out‑of‑range input degrades to a *bounded*
  increment rather than being amplified through the exponential `Reco`; and
* keeping the learned surface on the mean **minimal** — physics drift plus at most one bounded
  correction — limits the exposure of the mean to any single corrupted input (an argument, on the
  mean side, for the residual‑off primary).

The predictive **scale** is a separate matter and carries no such guarantee: the aleatoric scale
heads are learned functions of *all* drivers, so their out‑of‑distribution behaviour depends on
the noise law (single‑Wiener vs the extra `Δt`‑amplified process head of the state‑space form)
and on the regularising effect of the auxiliary diurnal task — modelling choices whose empirical
OOD behaviour is characterised in the results, not architectural guarantees. The clean separation
is deliberate: **the physics makes the mean robust by construction; the noise law and the
auxiliary supervision are the levers for the robustness of the spread.**

---

## 4. Noise families (the likelihood laws)

The unit innovation `ε` in (6) — and, equivalently, the conditional law of `NEE_{t+1}` given
`(μ, s)` — is one of a small family of **location–scale** laws. The scale `s` is trained by the
corresponding negative log‑likelihood (§5), and the same family is used to *sample* `ε` so the
generated predictive distribution is consistent with the trained likelihood. With
`u = (y − μ)/s`:

| family | conditional density of `NEE_{t+1}` | learnable shape | captures |
|---|---|---|---|
| **Gaussian** | `N(μ, s²)` | — | symmetric, light tails |
| **β‑NLL (Gaussian)** | `N(μ, s²)`, reweighted objective | — | as Gaussian, robust mean fit |
| **Student‑t** | `μ + s·t(ν)` | `ν = softplus(log ν)+2` | heavy tails |
| **Asymmetric Laplace (ALD)** | `∝ (1/s)·(κ/(1+κ²))·exp(−ρ_κ(u))` | `κ = exp(log κ)` | heavy tails **and** right skew |
| **Mixture (K‑component)** | `Σ_k w_k · N(μ_k, s_k²)` | offsets/scales/weights | arbitrary shape (baseline) |

with the ALD check function `ρ_κ(u) = κ·u` for `u ≥ 0` and `−u/κ` for `u < 0` (the pinball
loss). `κ = 1` is the symmetric Laplace; `κ < 1` gives the heavier **right** tail observed in
NEE residuals. The **primary noise family for the NEE problem is the ALD** — the residual of
NEE about the respiration mean is empirically heavy‑tailed *and* right‑skewed (a symmetric
Student‑t misses the skew) — with Gaussian, β‑NLL and Student‑t retained as loss ablations. The
K‑component mixture is the flexible‑shape head used for the no‑physics baseline; kept zero‑mean
so the drift still owns the conditional mean.

For numerical safety the log‑scale is clamped to `[−7, 5]`, the Student‑t dof is floored at
`ν = 2` (finite mean, so CRPS/PIT are defined), and `log κ ∈ [−3, 3]`.

---

## 5. Training objective

Training minimises a single **config‑driven composite loss** — a sum of terms, each of which
can be zeroed to ablate it without touching code. The **primary objective** for WienerNet‑SS is

```
L = L_NLL  +  λ_td · MSE_tendency  +  λ_res · ‖r_φ‖²  ( +  λ_k · MSE_{E0,rb} )   (9)
```

with the parameter‑anchor term inert in the known‑parameter regime. Each piece is below; §6
states what it anchors against.

**5.1 Conditional likelihood `L_NLL` (the stochastic objective).** The negative log‑likelihood
of `NEE_{t+1}` under the chosen family (§4) with location `μ` (7) and scale `s` (8), e.g. the
Gaussian

```
L_NLL = mean[ ½ (y − μ)² / s²  +  log s  +  ½ log 2π ] ,
```

and analogously for Student‑t, ALD (pinball `ρ_κ` in place of the quadratic) and the mixture
(`−logΣ_k`). **Why a likelihood and not `MSE(μ + noise, y)`:** for the MSE objective,
`E[ (μ + noise − y)² ] = (μ − y)² + Var[noise]`, which is minimised by `Var → 0` — MSE *kills*
the noise, so a free noise head can only be kept alive by matching it to an external *prior*
(the original WienerNet used an MMD‑to‑Gaussian penalty), and then the noise reflects the prior,
not the true conditional variance. The likelihood instead **rewards calibration**: the
`(y−μ)²/s²` term penalises a too‑small scale and the `log s` term penalises a too‑large scale,
so `s` settles at the *real* conditional spread with nothing to tune. Because location and scale
are orthogonal sufficient statistics in these families, the deterministic drift/residual (mean)
and the aleatoric noise (scale) **do not fight**.

**β‑NLL reweighting.** Optionally each per‑sample Gaussian/ALD NLL is multiplied by the detached
weight `stop_grad(s^{2β})` (Seitzer et al., 2022). Plain heteroscedastic NLL down‑weights the
mean gradient by `1/s²` in high‑variance regions, degrading the mean fit precisely on the
high‑flux nights that matter; `β = 0` is the plain NLL, `β = 1` makes the mean gradient behave
like MSE (most stable), and `β = 0.5` (the default) balances a sharp scale against a well‑fit
mean. It is the lever that keeps the tail‑robust ALD from collapsing its scale onto the bulk
median absolute deviation.

**5.2 Independent supervision (always on).**
* `MSE_tendency = ‖ g_φ(z) − Δ̄T ‖²` — the learned tendency head is supervised to the physics
  diurnal target `Δ̄T` (learned‑diurnal), or to observed `dTa` if no diurnal target is provided.
* `‖r_φ‖²` — L2 on the residual head (§3.4), the *physics‑first* regulariser.
* `MSE_{E0,rb}` — supervises the optional parameter head to REddyProc values, with anchor
  normalisation (dividing by the target variance) so the large‑scale `E0` term cannot dominate;
  **inert in the known‑parameter primary regime**.

**5.3 Legacy / baseline terms (config‑available, off in the primary model).** Point `MSE_nee`;
drift‑to‑increment `MSE_drift`; the original WienerNet distributional matcher `MMD_noise`
(match the noise sample to a zero‑mean prior) and `MMD_nee`; and a VAE latent `KL`. These
reconstruct the pre‑likelihood WienerNet and the MMD‑trained variant for ablation, and are
**replaced by `L_NLL`** in WienerNet‑SS.

---

## 6. What each component is anchored against (summary)

This is the table the interpretability argument rests on — every learnable quantity has an
explicit, physically meaningful supervisory target.

| component | learnable object | anchored against | mechanism |
|---|---|---|---|
| **drift tendency** | `g_φ(z)` = `(dT/dt)ᵖ` | **physics diurnal tendency** `Δ̄T(site,month,hour)` | `MSE_tendency` |
| **respiration params** | `(E0,rb)ᵖ` (optional) | **REddyProc** offline fits | `MSE_{E0,rb}` (inert when known) |
| **residual correction** | `r_φ(z)` | **zero** (kept a small, physics‑first correction) | `‖r_φ‖²` (L2) |
| **noise — single Wiener** | `σ(z)` | the **observed next‑step spread**: `σ√Δt` is fit so the conditional law of `NEE_{t+1}` matches the data | `L_NLL` |
| **noise — state‑space** | `σ_meas(z)`, `σ_proc(z)` | the **observed increment variance‑vs‑`Δt` law**: the flat term to the measurement (independent‑endpoint) part `2σ_meas²`, the ∝`Δt` term to the accumulating process part; jointly identified through the likelihood over varying `Δt` | `L_NLL` |
| **noise shape** | `κ` (ALD), `ν` (Student‑t), mixture params | the **residual shape** (skew/tails) that maximises the conditional likelihood | `L_NLL` |

Two points the manuscript should make explicit:

* **The noise scales are anchored to the *data*, not to a prior.** Unlike the MMD‑to‑Gaussian
  training of the original WienerNet (which forces the noise toward an assumed shape), the NLL
  anchors `σ` / `(σ_meas, σ_proc)` to the actual conditional spread of `NEE_{t+1}` about the
  drift. The state‑space split additionally uses the *physical* prior that measurement error is
  independent between samples (hence the exact factor 2 and the flat‑in‑`Δt` term) while the
  process term is the only genuinely accumulating (Wiener) part.
* **The residual is anchored to zero**, so it is a *diagnostic*: whatever temperature‑correlated
  signal it ends up carrying is, by construction, structure the Lloyd–Taylor physics missed —
  which is exactly what makes the drift/residual/noise split scientifically readable.

---

## 7. Model summary and variant matrix

WienerNet‑SS is one model with a small set of switches. The **primary approach** fixes:
known `E0,rb`; **learned‑diurnal** tendency (a predicted, physics‑anchored `dT/dt`); the
**exact respiration‑difference** drift (3); the graceful‑OOD clamp; the **ALD** likelihood with
`β = 0.5`; zero‑mean noise. The two axes reported for the NEE problem are:

| axis | options (both reported) | what it changes |
|---|---|---|
| **noise law** | single‑Wiener (5a) · state‑space (5b) | whether increment variance is pure `∝Δt` diffusion or measurement+process |
| **residual correction** | OFF (drift = `f_phys`) · ON (drift = `f_phys + r_φ`) | whether a bounded, zero‑anchored learned term absorbs the physics misfit |

with the likelihood family (Gaussian / β‑NLL / Student‑t / ALD / mixture) as a further loss
ablation. All switches leave the interpretable drift/residual/noise decomposition (§1) intact;
they only change how the noise accumulates and whether the drift carries a learned correction.

---

### Notation appendix

| symbol | meaning |
|---|---|
| `NEE_t`, `NEE_{t+1}` | observed flux at step `t`, target at `t+1` |
| `Δt` | per‑row step length (minutes) |
| `x_t`, `z` | exogenous drivers, latent code `Encoder(x_t)` |
| `T`, `Tref`, `T0` | air temperature, reference (10 °C), offset (46.02, used as `T+T0`) |
| `E0`, `rb` | Lloyd–Taylor activation energy, base respiration |
| `Reco`, `dReco/dT` | respiration (1) and its temperature sensitivity (2) |
| `(dT/dt)`, `Δ̄T` | temperature tendency; physics diurnal tendency (climatology) |
| `f_phys`, `r_φ` | physics drift rate (3), residual correction (4) |
| `σ`, `σ_meas`, `σ_proc`, `σ_eff` | single‑Wiener scale; state‑space measurement & process scales; effective increment std |
| `μ`, `s` | conditional mean (7) and scale (8) fed to the likelihood |
| `κ`, `ν` | ALD asymmetry, Student‑t degrees of freedom |
| `β` | β‑NLL gradient‑reweighting exponent |
