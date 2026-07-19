#!/usr/bin/env bash
# FINAL manuscript sweep — full 5-site leave-one-site-out (no data reduction), GT-k
# throughout (predicted-k confirmed to underperform: parameter-estimation bottleneck).
# Everything lands in ONE parent: outputs/final_loso/<label>_<site>_s<seed>/.
#
# Roster (all matched-capacity, 120 epochs, constant LR, eval last.pth, ensemble scored):
#   WienerNet (our best):
#     wienernet_laplace   physics SDE + heavy-tailed (Laplace/ALD) noise, beta=0.5 variance-match
#     wienernet_mixture   physics SDE + 3-component zero-mean mixture noise
#   Noise / loss ablation (same physics SDE backbone, no residual):
#     abl_gaussian        Gaussian (Wiener) diffusion
#     abl_beta            beta-NLL Gaussian
#     abl_studentt        Student-t diffusion
#     abl_laplace_b0      Laplace/ALD with beta=0 (the coverage fix OFF -> shows beta=0.5 matters)
#     abl_mmdnoise        MMD prior-matched noise (learned-mean) — the original WienerNet loss, on the SDE
#   Structural ablation:
#     abl_laplace_resid   Laplace noise + residual drift head (shows residual de-generalises)
#     abl_mixture_resid   mixture noise + residual drift head
#     abl_determin        deterministic physics drift, no aleatoric noise
#   Prior / physics references:
#     prior_analytical    analytical SDE (physics drift + calibrated constant noise); seed-invariant
#     prior_old_wienernet original level WienerNet + full MMD loss (published configuration)
#   No-physics competitors:
#     comp_mdn            mixture-density network (strongest black box)
#     comp_meanvar        mean-variance heteroscedastic net
#     comp_neuralsde      neural SDE (free drift + diffusion)
#     comp_rf, comp_xgb   tree baselines (point predictors; point-skill table only)
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate pytorch
cd /home/cognitia/Desktop/Work/PhD/WienerNet

ROOT=outputs/final_loso
EPOCHS=${EPOCHS:-120}
SITES=${SITES:-"woodwalton redmere_1 redmere_2 great_fen rosedene"}
SEEDS=${SEEDS:-"0 1 42"}
NPAR=${NPAR:-6}
mkdir -p $ROOT
echo "FINAL LOSO sweep: EPOCHS=$EPOCHS SITES=[$SITES] SEEDS=[$SEEDS] NPAR=$NPAR -> $ROOT"

TC="mlflow.enabled=false data.num_workers=0 data.split_strategy=site_holdout training.scheduler.name=none training.num_epochs=$EPOCHS device=cuda"
# trees have no torch training/scheduler/device knobs
TREEC="mlflow.enabled=false data.num_workers=0 data.split_strategy=site_holdout"
GTK="model.physics_k_source=ground_truth model.predict_k=false model.predict_temp_derivative=true"

# label -> full override string (site/seed/rundir appended per run). ENTRY marks the entrypoint.
declare -A OV ENTRY
OV[wienernet_laplace]="model=piae_increment loss=nll_ald $GTK model.noise_zero_mean=true";                                    ENTRY[wienernet_laplace]=torch
OV[wienernet_mixture]="model=piae_increment loss=nll_gaussian model.noise_mixture_components=3 $GTK model.noise_zero_mean=true"; ENTRY[wienernet_mixture]=torch
OV[abl_gaussian]="model=piae_increment loss=nll_gaussian $GTK model.noise_zero_mean=true";                                     ENTRY[abl_gaussian]=torch
OV[abl_beta]="model=piae_increment loss=nll_beta $GTK model.noise_zero_mean=true";                                            ENTRY[abl_beta]=torch
OV[abl_studentt]="model=piae_increment loss=nll_student_t $GTK model.noise_zero_mean=true";                                   ENTRY[abl_studentt]=torch
OV[abl_laplace_b0]="model=piae_increment loss=nll_ald loss.likelihood.beta=0.0 $GTK model.noise_zero_mean=true";              ENTRY[abl_laplace_b0]=torch
OV[abl_mmdnoise]="model=piae_increment loss=increment $GTK model.noise_zero_mean=false";                                      ENTRY[abl_mmdnoise]=torch
OV[abl_laplace_resid]="model=piae_increment_residual loss=nll_ald $GTK model.noise_zero_mean=true";                           ENTRY[abl_laplace_resid]=torch
OV[abl_mixture_resid]="model=piae_increment_residual loss=nll_gaussian model.noise_mixture_components=3 $GTK model.noise_zero_mean=true"; ENTRY[abl_mixture_resid]=torch
OV[abl_determin]="model=piae_reg_increment loss=increment $GTK";                                                              ENTRY[abl_determin]=torch
OV[prior_old_wienernet]="model=piae_sde_sampling loss=full";                                                                  ENTRY[prior_old_wienernet]=torch
OV[comp_mdn]="model=hetero_mdn loss=nll_student_t";                                                                           ENTRY[comp_mdn]=torch
OV[comp_meanvar]="model=hetero_mlp loss=nll_gaussian";                                                                        ENTRY[comp_meanvar]=torch
OV[comp_neuralsde]="model=neural_sde loss=nll_gaussian";                                                                      ENTRY[comp_neuralsde]=torch
# seed-invariant physics reference (deterministic drift + calibrated const noise) -> 1 seed
OV[prior_analytical]="model=analytical_sde loss=nll_gaussian";                                                                ENTRY[prior_analytical]=torch_1seed
# tree baselines (point predictors; write their own metrics.json, no ensemble eval)
OV[comp_rf]="model=rf";   ENTRY[comp_rf]=tree
OV[comp_xgb]="model=xgb"; ENTRY[comp_xgb]=tree

LABELS="wienernet_laplace wienernet_mixture abl_gaussian abl_beta abl_studentt abl_laplace_b0 abl_mmdnoise abl_laplace_resid abl_mixture_resid abl_determin prior_analytical prior_old_wienernet comp_mdn comp_meanvar comp_neuralsde comp_rf comp_xgb"

gate(){ while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do wait -n; done; }

one(){ local label="$1" site="$2" seed="$3"; local d=$ROOT/${label}_${site}_s${seed}; local log=$d.log
  local ent=${ENTRY[$label]}
  if [ "$ent" = "tree" ]; then
    if python scripts/train_baseline.py ${OV[$label]} $TREEC data.holdout_site=$site seed=$seed hydra.run.dir=$d > $log 2>&1; then
      echo "OK   $label $site s$seed"; else echo "FAIL $label $site s$seed"; fi
  else
    # torch: for prior_old_wienernet don't force TC's epochs override to clash; TC already has epochs
    if python scripts/train.py ${OV[$label]} $TC data.holdout_site=$site seed=$seed hydra.run.dir=$d > $log 2>&1 \
       && python scripts/evaluate.py --run $d --checkpoint last >> $log 2>&1; then
      echo "OK   $label $site s$seed"; else echo "FAIL $label $site s$seed"; fi
  fi
}

for label in $LABELS; do
  ent=${ENTRY[$label]}
  seeds="$SEEDS"; [ "$ent" = "torch_1seed" ] && seeds="0"
  for site in $SITES; do for seed in $seeds; do
    gate; one "$label" "$site" "$seed" &
  done; done
done
wait
echo "FINAL LOSO SWEEP DONE -> $ROOT"
