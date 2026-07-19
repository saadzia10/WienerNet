#!/usr/bin/env bash
# WienerNet-SS multi-site generalisation: does the Woodwalton win hold across all 5
# held-out sites x 3 seeds? We fix the two axes that were clearly best on Woodwalton
# (GT-k >> pred-k; no residual generalises better) and sweep the two SCIENTIFIC levers:
#   dt tendency : diurnal (exogenous physics climatology)  vs  learned_diurnal (head->MSE->diurnal)
#   noise       : state-space (measurement+process)         vs  single Wiener
# -> 4 ablations x 5 sites x 3 seeds = 60 runs, all into outputs/ss_loso/.
# Competing methods (MDN/NeuralSDE/mean-var/analytical/RF/XGB) are REUSED from
# outputs/final_loso/ (already 5 sites x 3 seeds) — no need to retrain them.
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
model.residual=false mlflow.enabled=false data.num_workers=0 data.split_strategy=site_holdout \
training.scheduler.name=none training.num_epochs=$EPOCHS device=cuda"

gate(){ while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do wait -n; done; }
one(){ local name="$1"; shift; local d=$ROOT/$name
  if python scripts/train.py $COMMON "$@" hydra.run.dir=$d > $ROOT/$name.log 2>&1 \
     && python scripts/evaluate.py --run $d --checkpoint last >> $ROOT/$name.log 2>&1; then
       echo "OK   $name"; else echo "FAIL $name (see $ROOT/$name.log)"; fi
}

for site in $SITES; do
  for seed in $SEEDS; do
    for dtmode in diur ldiur; do
      if [ "$dtmode" = diur ]; then DOV="model.drift_tendency=diurnal"
      else DOV="model.drift_tendency=learned_diurnal"; fi
      for nmode in ss wien; do
        NOV="model.noise_state_space=$([ "$nmode" = ss ] && echo true || echo false)"
        gate; one "${site}_s${seed}_${dtmode}_${nmode}" \
          data.holdout_site=$site seed=$seed $DOV $NOV &
      done
    done
  done
done
wait
echo "SS LOSO SWEEP DONE -> $ROOT"
