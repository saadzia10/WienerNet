#!/usr/bin/env bash
# Fills in the residual-correction-ENABLED arm of the primary approach, to sit alongside the
# no-residual runs already in outputs/ss_loso/. Primary axes fixed: learned-diurnal tendency
# (predicted dT/dt, MSE-anchored to the physics diurnal), known E0/rb. Swept:
#   noise : state-space (measurement+process)  vs  single Wiener
# Residual drift-correction ON (model.residual=true). 2 noise x 5 sites x 3 seeds = 30 runs.
# Named  {site}_s{seed}_ldiur_res_{ss|wien}  (the existing no-residual runs are
# {site}_s{seed}_ldiur_{ss|wien}). Same parent dir, same eval, so the report picks them up.
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
cd /home/cognitia/Desktop/Work/PhD/WienerNet
ROOT=outputs/ss_loso; EPOCHS=${EPOCHS:-120}; NPAR=${NPAR:-5}
mkdir -p $ROOT
SITES=${SITES:-"woodwalton rosedene redmere_1 redmere_2 great_fen"}
SEEDS=${SEEDS:-"0 1 42"}

COMMON="model=piae_increment loss=nll_ald model.noise_zero_mean=true model.exact_reco_drift=true \
model.predict_temp_derivative=true model.physics_k_source=ground_truth model.predict_k=false \
model.drift_tendency=learned_diurnal model.residual=true mlflow.enabled=false data.num_workers=0 \
data.split_strategy=site_holdout training.scheduler.name=none training.num_epochs=$EPOCHS device=cuda"

gate(){ while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do wait -n; done; }
one(){ local name="$1"; shift; local d=$ROOT/$name
  if python scripts/train.py $COMMON "$@" hydra.run.dir=$d > $ROOT/$name.log 2>&1 \
     && python scripts/evaluate.py --run $d --checkpoint last >> $ROOT/$name.log 2>&1; then
       echo "OK   $name"; else echo "FAIL $name (see $ROOT/$name.log)"; fi
}

for site in $SITES; do
  for seed in $SEEDS; do
    for nmode in ss wien; do
      NOV="model.noise_state_space=$([ "$nmode" = ss ] && echo true || echo false)"
      gate; one "${site}_s${seed}_ldiur_res_${nmode}" data.holdout_site=$site seed=$seed $NOV &
    done
  done
done
wait
echo "SS LOSO RESIDUAL SWEEP DONE -> $ROOT"
