#!/usr/bin/env bash
# dt-scale sweep across all methods. k in {1,2,4}, seeds {0,1,2}.
#   - Older WienerNet (piae, piae-reg, pivae): noise-mean fixes OFF and ON (two versions).
#   - AE / VAE baselines (no noise head): single version.
#   - Increment methods (A/B/C): GT E0/rb via ground_truth + NO k-head (predict_k=false),
#     E0/rb NOT fed to the encoder, dTa off. (A keeps the noise fixes; B is learned-mean
#     noise by identity; C is deterministic.)
#   - RF / XGB: flat features already include E0/rb; dTa not a feature.
# Inputs for ALL torch level models: include_k=true (E0/rb in), include_dtemp=false (dTa out).
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate pytorch
cd /home/cognitia/Desktop/Work/PhD/WienerNet

ROOT=outputs/dt_sweep
EPOCHS=${EPOCHS:-120}
KS=${KS:-"1 2 4"}
SEEDS=${SEEDS:-"0 1 2"}
mkdir -p $ROOT
echo "sweep: EPOCHS=$EPOCHS  KS=[$KS]  SEEDS=[$SEEDS]  -> $ROOT"

# include_k=true / include_dtemp=false for every torch LEVEL model (E0/rb in, dTa out)
INP="model.inputs.include_k=true model.inputs.include_dtemp=false"
# noise-mean fixes (structural zero-mean + empirical centred prior)
FIX="model.heads.noise_zero_mean=true training.noise_prior.kind=empirical training.noise_prior.center=true"

torch_run(){ local label="$1"; shift
  for K in $KS; do for S in $SEEDS; do
    local d=$ROOT/$label/k${K}_s${S}
    if timeout 900 python scripts/train.py "$@" seed=$S data.time_step_k=$K device=cuda \
         mlflow.enabled=false training.num_epochs=$EPOCHS hydra.run.dir=$d \
         > $ROOT/${label}_k${K}s${S}.log 2>&1; then echo "OK   $label k$K s$S"
    else echo "FAIL $label k$K s$S (see $ROOT/${label}_k${K}s${S}.log)"; fi
  done; done
}

baseline_run(){ local label="$1"; local model="$2"
  for K in $KS; do for S in $SEEDS; do
    local d=$ROOT/$label/k${K}_s${S}
    if timeout 900 python scripts/train_baseline.py model=$model seed=$S data.time_step_k=$K \
         mlflow.enabled=false hydra.run.dir=$d > $ROOT/${label}_k${K}s${S}.log 2>&1; then echo "OK   $label k$K s$S"
    else echo "FAIL $label k$K s$S (see $ROOT/${label}_k${K}s${S}.log)"; fi
  done; done
}

# ---- Older WienerNet: noise fixes OFF and ON ----
torch_run wnet_piae_nofix     model=piae_sde_sampling     loss=full $INP
torch_run wnet_piae_fix       model=piae_sde_sampling     loss=full $INP $FIX
torch_run wnet_piaereg_nofix  model=piae_sde_reg_sampling loss=reg  $INP
torch_run wnet_piaereg_fix    model=piae_sde_reg_sampling loss=reg  $INP $FIX
torch_run wnet_pivae_nofix    model=pivae_sde_sampling    loss=full loss.weights.kl_latent=0.001 $INP
torch_run wnet_pivae_fix      model=pivae_sde_sampling    loss=full loss.weights.kl_latent=0.001 $INP $FIX

# ---- AE / VAE baselines (no noise head) ----
torch_run ae   model=ae  loss=ae  $INP
torch_run vae  model=vae loss=vae $INP

# ---- Increment methods (GT E0/rb, no k-head; E0/rb NOT fed, dTa off) ----
GTNOHEAD="model.physics_k_source=ground_truth model.predict_k=false loss.weights.mse_E0=0 loss.weights.mse_rb=0 loss.normalize_anchors=false"
torch_run inc_residual     experiment=piae_increment_residual                 # already GT-K no head + noise fixes
torch_run inc_nonresidual  model=piae_increment     loss=increment $GTNOHEAD  # learned-mean noise (B identity)
torch_run inc_reg          model=piae_reg_increment loss=increment $GTNOHEAD  # deterministic (C)

# ---- Tree baselines (flat features incl E0/rb) ----
baseline_run rf  rf
baseline_run xgb xgb

echo "SWEEP DONE"
