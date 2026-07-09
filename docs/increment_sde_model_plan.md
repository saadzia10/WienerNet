# Plan: Increment-based PIAE / PIAE-reg SDE model

Handoff design doc for implementing a **new, separate** model type that reformulates
WienerNet as a proper Euler–Maruyama increment predictor. The existing
`WienerNetModel` stays untouched; this is a new model file + presets.

---

## 1. The problem we are solving

WienerNet models nighttime NEE as a Wiener-process SDE. The **current**
`WienerNetModel` (`wienernet/models/wienernet.py`) computes:

```
nee_raw = nee_decoder(z)            # black-box LEVEL reconstruction of NEE_t
noise   = mu(z) + eps*sigma(z)      # VAE-style reparam head (has a learned mean)
bnee    = nee_raw + noise
drift   = dReco/dT(E0,rb) * dT/dt   # physics increment rate (Lloyd-Taylor)
nee_pred = bnee + drift*dt          # (dt Euler step, see §3)
```

**Symptom (drift displacement / noise translation):** the "predictions without
noise" line (`nee_raw + drift`) sits *systematically above* ground truth, and the
predicted noise is centred *negative* to pull it back down. The noise is doing
two jobs: (a) zero-mean stochasticity, and (b) a learned **bias / physics-misfit
correction**. So the SDE decomposition is not identifiable — drift vs noise is a
mushy split.

**Root causes found this session:**
- The noise head has a learned conditional mean `mu(z)`, and the MMD-noise prior
  mean was set to the empirical mean of `NEE - NEE_phy = -0.238`, i.e. the model
  was *explicitly trained* to put a non-zero (physics-misfit-shaped) mean into the
  noise. `nee_raw` sits elevated to compensate.
- "GT noise" `= NEE - NEE_phy` conflates the **physics misfit** `b(state)` with the
  **true aleatoric noise**. The misfit is state-dependent: `corr(pred_noise, Ta)`
  reached +0.75 on bad seeds; per-site means differ. The decomposition is also
  **seed-unstable** (one seed clean, another leaks the misfit into the noise).
- The physics drift is a **long-timescale** signal (see §4): at 30 min it is
  near-inert and the increment is diffusion-dominated.

---

## 2. What is ALREADY DONE (do not redo — reuse these)

### 2a. Two noise-mean fixes (both implemented, tested, committed-ready)
1. **Structural zero-mean noise** — `HeadsConfig.noise_zero_mean` flag in
   `wienernet/models/wienernet.py`. When true, `fc_mu` is not built, so
   `noise = eps*sigma(z)` and `E[eps|z]=0` by construction. Wired through
   `scripts/train.py` and `scripts/evaluate.py`. Experiment
   `configs/experiment/zeromean_noise.yaml` sets it + zeroes the prior mean.
2. **Empirical centred noise prior** — `training.noise_prior.kind: {gaussian|empirical}`.
   `make_empirical_noise_prior(residuals, center=True)` in
   `wienernet/losses/composite.py` matches noise to the **centred train-only**
   residual pool (`bundle.noise_residuals`) instead of a parametric Gaussian, so
   it preserves the true skewed shape with a zero mean. Experiment
   `configs/experiment/empirical_noise.yaml`.

**Finding:** the empirical prior collapsed per-site noise means (e.g. redmere_1
`-1.14 -> -0.04` on a good seed) but does not force *conditional* zero-mean; the
structural fix is the robust one. They are complementary. The new model should
inherit the structural zero-mean noise (`noise_zero_mean`) for its aleatoric term.

### 2b. dt / Euler fix (implemented, tested)
- The data's `dNEE`, `dTa` are per-**minute** rates. The model now does an Euler
  step `nee_pred = bnee + drift*dt`, keeping `drift` a per-minute rate.
  `WienerNetConfig.dt` (default 30) + a per-row `dt` **column** from the data
  (default, config fallback). Pipeline `add_time_derivatives`
  (`data_pipeline/partitioning.py`) now emits `dt` and uses `total_seconds()/60`
  (the old `.dt.components.minutes` was buggy). Existing parquets patched.
- **Consequence found:** with the fix the drift is load-bearing (`corr(drift*dt,
  ΔNEE)=0.41` at 120 ep) but only because `pred_dtemp` becomes *unfaithful*
  (`corr(pred_dtemp, dTa)=-0.12`) — a free knob fitting 30-min noise. The honest
  fix is to make `pred_dtemp` a faithful forecast of the temperature change (keep
  `mse_temp_derivative`), accepting a weak per-step drift.

### 2c. Timescale finding + multi-scale sweep (implemented, tested)
- `notebooks/04_timescale_analysis.ipynb` +
  `wienernet/evaluation/timescale.py::increment_correlation_by_lag`:
  `corr(ΔNEE, ΔReco)` rises **0.014 (30 min) -> 0.235 (8 h)**. Physics is a
  long-timescale signal; at 30 min the increment is diffusion-dominated. Even at
  8 h it is only ~6% of variance. Physics is strong at the **level**
  (`NEE ≈ Reco(T)`), weak at the short-scale **increment**.
- **Per-scale models**, not a multi-scale loss: `wienernet/data/rescale.py::
  rescale_to_timestep(df, k)` rebuilds targets at a `k*30`-min step **within each
  night** (rebuilds `NEE_next=NEE_{t+k}`, `dNEE`, `dTa`, `dt=k*30`; drops rows
  whose `t+k` crosses a night boundary). Config `data.time_step_k` (null=30 min);
  experiment `configs/experiment/dt_scale_sweep.yaml` sweeps `k=1,2,4,8,16`.
  Works for PIAE, AE, RF, XGB (baseline entrypoint now passes `time_step_k`). The
  new increment model MUST compose with this (use per-row `dt`).

---

## 3. Key design decisions for the new model (already agreed)

1. **Predict the increment, integrate the boundary.** The model predicts
   `dNEE = drift + noise`; the forecast is `NEE_{t+1} = NEE_t + dNEE` via
   Euler–Maruyama. **`bNEE` (= observed `NEE_t`) is the integration boundary
   only** — it is NOT reconstructed. So the new model **drops** `nee_decoder`,
   `nee_raw`, `bnee`, and the `mse_bnee` term.
2. **Exogenous increment / no exposure bias.** Drift and diffusion are functions
   of exogenous state `(T, dT/dt, E0, rb, drivers)` — **not** of `NEE_t`. So drop
   `bNEE` from the encoder input (`WienerNetModel` currently feeds
   `cat(x, bNEE, k)`; the new model feeds exogenous features only). This makes
   free-running multi-step rollout clean: increments accumulate with no
   train/inference mismatch. The physics self-anchors the level to `Reco(T)`, so
   long rollouts don't run away.
3. **Faithful `dTemp`.** Because forecasting has no future drivers, the model
   predicts `dT/dt` and it must be a *faithful* forecast of `dTa` (keep
   `mse_temp_derivative`), not a free knob. The physics drift is then honest
   (weak per-step); the residual + noise carry the rest.
4. **Euler–Maruyama scaling.** Drift scales with `dt`, diffusion with `sqrt(dt)`:
   `dNEE = f*dt + sigma*sqrt(dt)*Z`. Use the per-row `dt` column (composes with the
   dt-scale sweep). At fixed dt these are constants, but include them explicitly
   for cross-scale correctness.
5. **Training = one-step teacher forcing.** Each transition is independent: use GT
   `NEE_t` as boundary, predict `NEE_{t+1}`. Multi-step rollout is an *evaluation*
   mode (seed with GT `NEE_{t0}`, then feed the model's own accumulated NEE).
6. **Keep physics identifiable.** Anchor `E0, rb` with `mse_E0/mse_rb` (REddyProc
   fits). Regularize the residual (L2) so physics explains first and the residual
   is a small correction — else it collapses `Reco` into a black box (the current
   failure, relocated).

---

## 4. The new model — architecture and three versions

New file `wienernet/models/increment_sde.py` with a class (e.g.
`IncrementSDEModel`) + a config dataclass, following the `WienerNetModel` /
`HeadsConfig` / `WienerNetConfig` pattern. Reuse physics from
`wienernet/physics/lloyd_taylor.py` (`reco`, `dreco_dT`, `sde_drift`;
Reichstein convention, `T0 = 46.02`, denominators `(T + T0)`).

### Shared forward skeleton
```
z      = encoder(x_exogenous)                       # x excludes NEE_t
E0,rb  = k_head(z)                                   # -> mse_E0/mse_rb (REddyProc)
dtdt   = temp_deriv_head(z)                          # -> mse_temp_derivative (faithful dTa)
f_phys = dReco/dT(T, E0, rb) * dtdt                  # physics drift RATE (per minute)
# version-specific drift + noise below ...
dNEE   = drift_term * dt + noise_term * sqrt(dt)
nee_pred = bNEE_observed + dNEE                      # Euler-Maruyama; -> mse_nee vs NEE_{t+k}
```

### Version A — Residual-based PIAE  (`piae_increment_residual`)
- `r = residual_head(z)` — a state-dependent **drift-misfit** correction.
- `sigma = softplus(sigma_head(z))`; `noise = eps*sigma` with `eps~N(0,1)`
  (**zero-mean**, i.e. `noise_zero_mean=True` semantics).
- `drift_term = f_phys + r`; `noise_term = sigma`.
- `dNEE = (f_phys + r)*dt + sigma*sqrt(dt)*Z`.
- Losses: `mse_nee` (increment/`NEE_{t+k}`), `mse_drift` (drift vs `dNEE` target,
  optional), `mse_E0/mse_rb`, `mse_temp_derivative`, `mmd_noise` (aleatoric shape,
  can use the empirical centred prior), **`residual_l2 = lambda*mean(r^2)`** (new
  term — sweep `lambda`).
- Interpretation payoff: `r(z)` is a publishable diagnostic of *where Lloyd-Taylor
  is biased* as a function of state; noise is genuine aleatoric.

### Version B — Non-residual PIAE  (`piae_increment`)
- **No residual head.** The **noise term keeps a learned mean** (fc_mu present,
  `noise_zero_mean=False`) so it absorbs the physics misfit — the "existing noise
  term learning physics misfit" behaviour, but in the increment formulation.
- `drift_term = f_phys`; `noise_term = mu(z) + eps*sigma(z)`.
- `dNEE = f_phys*dt + (mu(z) + eps*sigma(z))*sqrt(dt)*Z`  *(decide sqrt(dt) placement
  for the mean vs stochastic part — see §6)*.
- Losses: as A minus `residual_l2`; `mmd_noise` on the (non-centred) noise.
- This is the honest baseline that shows what the noise-as-misfit costs.

### Version C — Reg-only, no residual, drift direct  (`piae_reg_increment`)
- **PIAE-reg analog.** Predicts the drift and adds it to the **observed** NEE via
  Euler; no residual, no noise-based level reconstruction.
- `dNEE = f_phys*dt` (optionally `+ sigma*sqrt(dt)*Z` if a stochastic term is
  wanted; the reg spirit is deterministic drift). `nee_pred = bNEE_obs + dNEE`.
- Losses: `mse_nee`, `mse_drift`, `mse_E0/mse_rb`, `mse_temp_derivative`. No
  `mmd_bnee`, no residual.

All three: `bNEE` is observed boundary only; increment is exogenous; uses per-row
`dt`; zero-mean aleatoric noise where noise is present (except B). Add registry
presets in `wienernet/models/registry.py` and matching
`configs/model/piae_increment*.yaml`. Add `configs/loss/increment*.yaml` (enable
`mse_nee`, `mse_drift`, `mse_E0/rb`, `mse_temp_derivative`, `mmd_noise`,
`residual_l2`; disable `mse_bnee`, `mmd_bnee`).

---

## 5. Implementation checklist

- [ ] `wienernet/models/increment_sde.py`: `IncrementSDEModel` + config dataclass,
      flags: `residual: bool`, `noise: bool`, `noise_zero_mean: bool`,
      `stochastic: bool` (C can be deterministic). Forward returns dict with
      `drift`, `residual`, `noise`, `noise_mu/logvar`, `k`, `temp_derivative`,
      `dnee_pred`, `nee_pred` (and `sigma`). Exogenous input (no bNEE).
- [ ] Physics: reuse `sde_drift`; add nothing to lloyd_taylor unless needed.
- [ ] `residual_head` + `sigma_head` (softplus) building blocks (reuse
      `build_mlp_with_head` from `components.py`).
- [ ] Loss: add `residual_l2` term to `wienernet/losses/composite.py` (guarded by
      weight, no-op at 0). Reuse `mse_nee`, `mse_drift`, `mse_E0/rb`,
      `mse_temp_derivative`, `mmd_noise`. The increment `mse_nee` compares
      `bNEE_obs + dNEE` to `NEE_{t+k}` — verify `nee_pred` semantics match.
- [ ] `sqrt(dt)` noise scaling in the forward (per-row `dt` from batch).
- [ ] Registry presets + `configs/model/*` + `configs/loss/increment*.yaml`.
- [ ] `scripts/train.py` / `evaluate.py`: model build must handle the new class
      (may need a builder switch on `cfg.model.variant` or a shared
      `build_model`). Ensure `time_step_k` (dt-scale sweep) still flows.
- [ ] Encoder input change: drop `bNEE` from features for the increment model
      (input_dim changes; keep `k`/E0,rb as conditioning or predict-only — decide).
- [ ] Tests (`tests/test_models.py` / new): output shapes; `nee_pred = bNEE +
      dNEE`; residual off/on; zero-mean noise for A; `sqrt(dt)`/`dt` scaling;
      per-row dt override; overfit-single-batch; gradient flow; checkpoint resume.
- [ ] Eval: standard `evaluate.py` works (it predicts `nee_pred`). Add a
      **multi-step rollout** eval (seed GT `NEE_{t0}`, accumulate) as the honest
      SDE test — the drift's value is cumulative, not per-step. Also report
      `corr(residual, Ta)` and `corr(noise, Ta)` to show the misfit moved from
      noise (B) to residual (A).

---

## 6. Open decisions to make at build time

1. **`sqrt(dt)` for the noise mean (version B).** The stochastic part scales with
   `sqrt(dt)`; a *learned mean* on the noise is really a deterministic drift
   correction and should scale with `dt`. Cleanest: in B, treat `mu(z)` as part of
   the drift (`drift += mu(z)`, scaled by `dt`) and keep only `eps*sigma*sqrt(dt)`
   as stochastic — which makes B ≈ A with an unregularized, unnamed residual.
   Decide whether B keeps `mu` in the noise (literal "noise learns misfit") or
   folds it into drift.
2. **Encoder input.** Confirm dropping `bNEE`; decide whether `E0,rb` are inputs
   (REddyProc conditioning) or predict-only. Recommend predict-only from drivers
   + keep `mse_E0/rb` anchor, to stay fully exogenous.
3. **`residual_l2` strength** — the key hyperparameter of version A; sweep it
   (e.g. 0.1, 1, 5, 10). Too low → residual eats physics; too high → can't correct.
4. **Reg version C stochasticity** — deterministic drift only, or with a diffusion
   term? "Predicts drift directly and adds to observed NEE" reads deterministic.
5. **Multi-scale interaction.** Confirm the increment model + `dt_scale_sweep`
   gives the manuscript "performance vs dt" for these new variants too.

---

## 7. Validation targets

- `nee_pred` (one-step) RMSE/R² should be comparable to the current model — this
  is a re-parameterisation, not a re-fit.
- **Decomposition cleanup:** `corr(noise, Ta) -> 0` at every seed (A); `residual`
  carries the temperature-dependent misfit; predicted-noise mean ≈ 0 per site.
- **Faithful physics:** `corr(pred_dtemp, dTa)` should be positive (not the -0.12
  free-knob regime).
- **Rollout:** integrating over a night should track the `Reco(T)` trend; compare
  A (residual) vs B (noise-misfit) vs C (reg) trajectories.
- **Manuscript axis:** run each version through `dt_scale_sweep` (k=1..16) and
  report metrics vs horizon alongside PIAE/AE/RF/XGB.

---

## 8. Repo pointers (for the fresh chat)

- Existing model: `wienernet/models/wienernet.py` (`WienerNetModel`,
  `HeadsConfig`, `WienerNetConfig`); presets `wienernet/models/registry.py`.
- Physics: `wienernet/physics/lloyd_taylor.py`.
- Losses: `wienernet/losses/composite.py` (`compute_losses`,
  `make_gaussian_noise_prior`, `make_empirical_noise_prior`).
- Trainer: `wienernet/training/trainer.py`. Data: `wienernet/data/loader.py`
  (`build_dataloaders`, `DataBundle`), `dataset.py`, `rescale.py`,
  `features.py` (`physics_nee_numpy`, `compute_noise_statistics`).
- Data columns: `NEE`, `NEE_next` (target), `bNEE`=`NEE` (boundary), `dNEE`,
  `dTa`, `dt`, `E0`, `rb`, `Ta`, `site`, `DateTime`. Targets are per-minute rates
  (`dNEE = ΔNEE/dt`).
- Tests: run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`.
- GPU currently unusable in this env (CUDA err 999 + empty `CUDA_VISIBLE_DEVICES`);
  train on CPU (`device=cpu`, ~6 min for 120 ep / seed).
</content>
