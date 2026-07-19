# Process-model comparison baselines

A ladder of baselines for the study's primary claim — the quality of the one-step
predictive distribution of nighttime NEE, p(NEE_{t+1} | drivers_t, NEE_t). Each
baseline removes exactly one ingredient of the full model (physics drift, learned
misfit correction, learned state-dependent noise, SDE integration), so together
they isolate which ingredient earns its place.

**Shared setup (fairness).** Every baseline uses the same meteorological drivers,
the same chronological train/test split, the same input normalisation, and the
same one-step target and scale as the full model. Each emits a predictive
distribution over next-step NEE and is therefore scored on the same proper scores
(CRPS, predictive log-score, PIT calibration, interval coverage) and the same
process-consistency diagnostics. The learned baselines are trained with the same
likelihood objective as the full model and are given the same network width and
depth, so any difference in skill reflects model *structure* rather than capacity
or training. Results are reported across multiple random seeds (mean ± standard
error) for the learned baselines.

## Baseline 1 — Analytical SDE (constant diffusion)

The textbook stochastic respiration model and the cheapest rung of the ladder:
pure physics with a constant noise level and no learning. The drift is the
analytic time-derivative of the Lloyd–Taylor respiration function evaluated at the
observed temperature and its rate of change — exactly the drift the full model
integrates — and the one-step increment is formed by the same Euler–Maruyama step,

  ΔNEE = f_phys · Δt + σ · √Δt · ε,   ε ~ 𝒩(0, 1),

with NEE_{t+1} = NEE_t + ΔNEE. The single free quantity is the constant diffusion
level σ, estimated from the spread of the training-set increment misfit (observed
increment minus the physics-drift increment); because the increment noise scales
with √Δt, σ is estimated as the time-step-normalised root-mean-square of that
misfit, which coincides with its maximum-likelihood value. Two forms are provided:
a single global σ, and one σ per site (calibrated within each site) to accommodate
the differing noise magnitudes across the fen towers. Being pure physics with a
fixed noise level, this baseline is deterministic given the data and carries no
seed dependence.

This baseline is the reference point for the core claim: it shows exactly what the
learned misfit correction and the learned state-dependent noise add over textbook
physics with constant noise. In practice it is well calibrated for a single step
but its independent, constant-magnitude increments accumulate variance over a
night faster than the observations do — the deficiency the learned, state-dependent
noise is designed to remedy.

## Baseline 2 — No-physics heteroscedastic model

A black-box neural model that maps the same drivers directly to a predictive
distribution over the next-step increment, with no physics, no drift, and no SDE
integration (in particular, no √Δt time-step scaling of the noise). It is the key
ablation: it isolates whether embedding the physics and the SDE structure adds
anything beyond simply learning a flexible, heteroscedastic conditional
distribution. Two forms are provided:

- **Mean–variance form.** The network outputs a predicted mean and a predicted
  spread of the increment, i.e. a single conditional Gaussian (or Student-t) per
  time step. This is the simplest heteroscedastic baseline.

- **Mixture-density form.** The network instead outputs a small mixture of
  components (means, spreads, and mixing weights), allowing a skewed or
  heavy-tailed, non-Gaussian predictive shape. Because a single mean and spread
  cannot describe a mixture, it is trained with the mixture likelihood and scored
  from samples of its predictive distribution (the same sample-based scoring used
  for the stochastic full-model variants); a moment-matched mean and variance are
  retained only for point-error reporting.

Both forms share the full model's network width and depth, so the comparison
reflects the presence or absence of physical and dynamical structure, not model
size. The mixture form fits the heavy-tailed, right-skewed increment distribution
substantially better than the single-Gaussian form, consistent with the empirical
error structure of the flux data.

## Baseline 3 — Neural SDE (data-driven drift and diffusion)

A direct, observation-space neural stochastic differential equation in which both
the drift and the diffusion are free neural functions of the state, with no
physics prior, integrated with the same one-step Euler–Maruyama scheme as the full
model,

  ΔNEE = g(state) · Δt + σ(state) · √Δt · ε,   ε ~ 𝒩(0, 1),

and trained on the observed next-step values with the same likelihood objective.
It keeps the SDE time-step structure (the drift scales with Δt and the diffusion
with √Δt), but — unlike the analytical baseline — replaces the physics drift with a
black box, and — unlike the no-physics heteroscedastic model — retains the dynamical
integration. It carries no misfit decomposition: the entire drift is learned
freely, with no physical parameters.

This baseline substantiates the study's critique of purely data-driven neural
SDEs. Its unconstrained drift can produce occasional unstable predictions, and its
diffusion is prone to the variance-collapse pathology of heteroscedastic
likelihood training (a few points driven to a near-zero predicted spread), so it
is the baseline that most benefits from a robust likelihood variant. Its
independent increments also over-accumulate variance across a night, like the
constant-diffusion baseline, because nothing in the free drift enforces the bounded,
mean-reverting behaviour of the underlying respiration process.

## Training objectives (loss functions)

Every baseline is trained by maximum likelihood — the same objective family as the
full model — so the comparison is on equal footing. The applicable and recommended
likelihoods differ by baseline:

| Baseline | Applicable likelihoods | Recommended |
|---|---|---|
| Analytical SDE (constant diffusion) | Gaussian; Student-t (heavy-tailed option) | Gaussian |
| No-physics, mean–variance | Gaussian; variance-stabilised Gaussian; Student-t | variance-stabilised or Student-t |
| No-physics, mixture density | Mixture (Gaussian or Student-t components) | Mixture, Student-t components for heavy tails |
| Neural SDE | Gaussian; variance-stabilised Gaussian; Student-t | variance-stabilised or Student-t |

A few consequences of the model structure make some choices inapplicable or
redundant. For the **analytical SDE**, the drift carries no learnable parameters
and the single constant noise level is already at its likelihood-optimal value, so
the variance-stabilised Gaussian objective (which only re-weights the mean-fit
gradient) reduces to the ordinary Gaussian one — exactly so for the fixed-length
consecutive steps used here — and offers nothing extra; a mixture likelihood does
not apply, as there is one constant noise level rather than a mixture. For the **mixture-density** baseline, the predictive law is a mixture
that no single mean-and-spread can represent, so the single-component likelihoods
(ordinary and variance-stabilised Gaussian) cannot score it — only the mixture
likelihood applies. The **mean–variance** and **neural SDE** baselines both learn a
per-point spread, so they benefit from the variance-stabilised Gaussian (which
prevents the mean fit from degrading where the variance is large) or the Student-t
(which matches the heavy-tailed residuals); the neural SDE in particular is the
most exposed to variance collapse under the plain Gaussian likelihood and is
therefore the strongest case for a robust variant.

For a like-for-like comparison, train the four baselines and the full model with
the **same** likelihood family (e.g. Student-t if that is the headline objective);
the mixture baseline uses the mixture form of that same family.

The non-likelihood loss terms used elsewhere in the framework do not apply to the
baselines: the physics-parameter anchors and the temperature-rate and
drift-versus-observed-increment anchors are inert (the analytical baseline uses the
observed physics parameters directly rather than predicting them, and the two
neural baselines contain no physics); the boundary/level reconstruction term is
inert (all baselines integrate onto the observed boundary rather than reconstruct
the level); the explicit misfit penalty applies only to the full model's misfit
head; the distribution-matching noise-prior term is superseded by the likelihood
itself; and the variational-latent regulariser does not apply, as the baselines use
deterministic encoders. Only the likelihood terms above are active.

---

**Summary of the ladder.** Analytical SDE = physics drift + constant noise, no
learning. No-physics heteroscedastic = a flexible conditional density, no dynamical
structure. Neural SDE = learned drift + learned noise, no physics. The full model =
physics drift + learned misfit correction + learned state-dependent noise + SDE
integration. Comparing the four on identical inputs, splits, targets and scores
isolates the contribution of each ingredient.
