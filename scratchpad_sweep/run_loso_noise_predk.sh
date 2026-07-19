#!/usr/bin/env bash
# Quick predicted-k LOSO check: same as run_loso_noise.sh but the respiration
# parameters E0/rb are PREDICTED by the k-head (anchored by mse_E0/rb) instead of
# supplied ground-truth. Tests whether the physics CRPS win survives when the black
# box's information advantage is removed (our method must estimate E0/rb itself).
# Only the two winning no-residual heads: ALD(beta=0.5) and 3-comp mixture.
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate pytorch
cd /home/cognitia/Desktop/Work/PhD/WienerNet

ROOT=outputs/loso_noise_predk
EPOCHS=${EPOCHS:-120}
SITES=${SITES:-"woodwalton redmere_1 redmere_2 great_fen rosedene"}
SEEDS=${SEEDS:-"0 1 42"}
NPAR=${NPAR:-5}
mkdir -p $ROOT
echo "LOSO-noise PREDICTED-k check: SITES=[$SITES] SEEDS=[$SEEDS] NPAR=$NPAR -> $ROOT"

# predicted k: predict_k=true + physics_k_source=predicted (the piae_increment defaults);
# mse_E0/rb anchors stay on (nll_* configs). zero-mean noise, dTa predicted, drift bound on.
COMMON="model.physics_k_source=predicted model.predict_k=true model.noise_zero_mean=true \
model.predict_temp_derivative=true \
mlflow.enabled=false data.num_workers=0 data.split_strategy=site_holdout \
training.scheduler.name=none training.num_epochs=$EPOCHS device=cuda"

declare -A MODEL=( [ald_B]=piae_increment [mix_B]=piae_increment )
declare -A LOSS=(  [ald_B]=nll_ald        [mix_B]=nll_gaussian )
declare -A EXTRA=( [ald_B]=""             [mix_B]="model.noise_mixture_components=3" )

gate(){ while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do wait -n; done; }
one(){ local label="$1" site="$2" seed="$3"; local d=$ROOT/${label}_${site}_s${seed}
  local log=$ROOT/${label}_${site}_s${seed}.log
  if python scripts/train.py model=${MODEL[$label]} loss=${LOSS[$label]} ${EXTRA[$label]} \
        $COMMON data.holdout_site=$site seed=$seed hydra.run.dir=$d > $log 2>&1 \
     && python scripts/evaluate.py --run $d --checkpoint last >> $log 2>&1; then
       echo "OK   $label $site s$seed"
  else echo "FAIL $label $site s$seed (see $log)"; fi
}

for label in ald_B mix_B; do
  for site in $SITES; do for seed in $SEEDS; do
    gate; one "$label" "$site" "$seed" &
  done; done
done
wait
echo "LOSO-NOISE PREDK CHECK DONE -> $ROOT"
