#!/usr/bin/env bash
# Leave-one-site-out sweep for the terminal noise models (Workstream F + coverage fix).
# For every held-out site x seed, train our increment SDE with each noise head and
# evaluate on the held-out site (leakage-free: site_holdout split, constant LR, eval last.pth).
#
# Models (all GT-k so the drift is comparable; dTa always predicted). The ALD
# under-coverage fix is beta=0.5 reweighting on the ALD NLL (nll_ald default) — it
# lifts cov90 0.61->0.85 while keeping the CRPS win; a woodwalton grid showed the
# physics-anchored scale overshoots/destabilises and beta>0.5 collapses the scale,
# so beta=0.5 is the sweet spot (see docs). Configs:
#   ald_B   : piae_increment          (no residual)  + ALD noise (beta=0.5)
#   mix_B   : piae_increment          (no residual)  + 3-comp Gaussian mixture noise
#   ald_A   : piae_increment_residual (residual r(z))+ ALD noise (beta=0.5)
#   mix_A   : piae_increment_residual (residual r(z))+ 3-comp Gaussian mixture noise
# The ALD / mixture head is on the NOISE branch only; r(z) stays a deterministic drift
# correction. The no-physics MDN baseline already exists in outputs/loso/ (reused).
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate pytorch
cd /home/cognitia/Desktop/Work/PhD/WienerNet

ROOT=outputs/loso_noise
EPOCHS=${EPOCHS:-120}
SITES=${SITES:-"woodwalton redmere_1 redmere_2 great_fen rosedene"}
SEEDS=${SEEDS:-"0 1 42"}
NPAR=${NPAR:-5}
mkdir -p $ROOT
echo "LOSO-noise sweep: EPOCHS=$EPOCHS SITES=[$SITES] SEEDS=[$SEEDS] NPAR=$NPAR -> $ROOT"

# common overrides: GT-k (no k-head), zero-mean noise, dTa predicted
COMMON="model.physics_k_source=ground_truth model.predict_k=false model.noise_zero_mean=true \
model.predict_temp_derivative=true \
mlflow.enabled=false data.num_workers=0 data.split_strategy=site_holdout \
training.scheduler.name=none training.num_epochs=$EPOCHS device=cuda"

# label -> (model, loss, extra) config triples
declare -A MODEL=( [ald_B]=piae_increment          [mix_B]=piae_increment \
                   [ald_A]=piae_increment_residual  [mix_A]=piae_increment_residual )
declare -A LOSS=(  [ald_B]=nll_ald      [mix_B]=nll_gaussian \
                   [ald_A]=nll_ald      [mix_A]=nll_gaussian )
declare -A EXTRA=( [ald_B]=""                               [mix_B]="model.noise_mixture_components=3" \
                   [ald_A]=""                               [mix_A]="model.noise_mixture_components=3" )

gate(){ while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do wait -n; done; }

one(){ local label="$1" site="$2" seed="$3"; local d=$ROOT/${label}_${site}_s${seed}
  local log=$ROOT/${label}_${site}_s${seed}.log
  if python scripts/train.py model=${MODEL[$label]} loss=${LOSS[$label]} ${EXTRA[$label]} \
        $COMMON data.holdout_site=$site seed=$seed hydra.run.dir=$d > $log 2>&1 \
     && python scripts/evaluate.py --run $d --checkpoint last >> $log 2>&1; then
       echo "OK   $label $site s$seed"
  else echo "FAIL $label $site s$seed (see $log)"; fi
}

for label in ald_B mix_B ald_A mix_A; do
  for site in $SITES; do for seed in $SEEDS; do
    gate; one "$label" "$site" "$seed" &
  done; done
done
wait
echo "LOSO-NOISE SWEEP DONE -> $ROOT"
