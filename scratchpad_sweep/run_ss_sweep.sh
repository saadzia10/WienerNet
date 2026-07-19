#!/usr/bin/env bash
# WienerNet-SS ablation: all 16 combinations of the four new axes, Woodwalton held out,
# 1 seed. Constant across combos: exact_reco_drift (the physics-integration drift) and the
# ALD noise family. The four toggles:
#   k   : predict_k (predicted E0/rb + MSE anchor)  vs  ground-truth E0/rb
#   dt  : drift_tendency = diurnal (exogenous physics)  vs  learned_diurnal (head -> diurnal)
#   r   : residual drift correction on/off
#   n   : noise_state_space (measurement+process) vs current single Wiener head
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate pytorch
cd /home/cognitia/Desktop/Work/PhD/WienerNet
ROOT=outputs/ss_sweep; EPOCHS=${EPOCHS:-120}; NPAR=${NPAR:-5}; SEED=${SEED:-0}
mkdir -p $ROOT
COMMON="model=piae_increment loss=nll_ald model.noise_zero_mean=true model.exact_reco_drift=true \
model.predict_temp_derivative=true mlflow.enabled=false data.num_workers=0 \
data.split_strategy=site_holdout data.holdout_site=woodwalton training.scheduler.name=none \
training.num_epochs=$EPOCHS device=cuda seed=$SEED"

gate(){ while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do wait -n; done; }
one(){ local name="$1"; shift; local d=$ROOT/$name
  if python scripts/train.py $COMMON "$@" hydra.run.dir=$d > $ROOT/$name.log 2>&1 \
     && python scripts/evaluate.py --run $d --checkpoint last >> $ROOT/$name.log 2>&1; then
       echo "OK   $name"; else echo "FAIL $name (see $ROOT/$name.log)"; fi
}

for kmode in gtk pk; do
  if [ "$kmode" = gtk ]; then KOV="model.physics_k_source=ground_truth model.predict_k=false"
  else KOV="model.physics_k_source=predicted model.predict_k=true"; fi
  for dtmode in diur ldiur; do
    if [ "$dtmode" = diur ]; then DOV="model.drift_tendency=diurnal"
    else DOV="model.drift_tendency=learned_diurnal"; fi
    for rmode in res nores; do
      ROV="model.residual=$([ "$rmode" = res ] && echo true || echo false)"
      for nmode in ss wien; do
        NOV="model.noise_state_space=$([ "$nmode" = ss ] && echo true || echo false)"
        gate; one "${kmode}_${dtmode}_${rmode}_${nmode}" $KOV $DOV $ROV $NOV &
      done
    done
  done
done
wait
echo "SS SWEEP DONE -> $ROOT"
