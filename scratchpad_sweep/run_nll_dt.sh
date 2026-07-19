#!/usr/bin/env bash
# NLL dt-sweep: beta-NLL vs the old MSE+MMD baseline at 30m/2h/6h (k=1/4/12),
# seeds 0,1,2. Then evaluate (nee_mean metrics + calibration + new mean+band plots)
# and aggregate the noise-vs-residual balance across horizons.
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate pytorch
export MKL_THREADING_LAYER=GNU MPLBACKEND=Agg
cd /home/cognitia/Desktop/Work/PhD/WienerNet
ROOT=outputs/nll_dt_sweep
EP=120
mkdir -p $ROOT

run(){ local label="$1"; shift
  for K in 1 4 12; do for S in 0 1 2; do
    local d=$ROOT/$label/k${K}_s${S}
    if timeout 900 python scripts/train.py "$@" seed=$S data.time_step_k=$K device=cuda \
         mlflow.enabled=false training.num_epochs=$EP hydra.run.dir=$d \
         > $ROOT/${label}_k${K}s${S}.log 2>&1; then echo "OK   $label k$K s$S"
    else echo "FAIL $label k$K s$S (see $ROOT/${label}_k${K}s${S}.log)"; fi
  done; done
}

run nll_beta         experiment=piae_increment_residual_nll loss=nll_beta
run mse_mmd_baseline experiment=piae_increment_residual
echo "TRAIN DONE"
python scratchpad_sweep/nll_dt_agg.py > $ROOT/aggregate.out 2>&1
echo "AGG EXIT $?"
tail -40 $ROOT/aggregate.out
echo "ALL DONE"
