#!/usr/bin/env bash
# Waits for the ss_loso sweep to finish, then runs the full downstream pipeline:
#   1. loso_report.py          -> per-site x seed eval CSVs + printed tables
#   2. gap_band_eval.py        -> NEXT-2 structural band fix across all 5 sites
#   3. autoregressive_gapfill  -> AR gap-fill (all methods + trees) across all 5 sites
#   4. loso_plots.py           -> publication figures
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
cd /home/cognitia/Desktop/Work/PhD/WienerNet
LOG=outputs/ss_loso_downstream.log

# wait for the sweep to signal completion (or all 60 runs accounted for)
while ! grep -q "SS LOSO SWEEP DONE" outputs/ss_loso_run.log 2>/dev/null; do
  done=$(grep -cE "^OK|^FAIL" outputs/ss_loso_run.log 2>/dev/null || echo 0)
  [ "$done" -ge 60 ] && break
  sleep 60
done
echo "sweep done ($(grep -cE '^OK' outputs/ss_loso_run.log) OK / $(grep -cE '^FAIL' outputs/ss_loso_run.log) FAIL) — running downstream" | tee $LOG

echo -e "\n########## 1. LOSO EVAL REPORT ##########" | tee -a $LOG
python analysis/ss/loso_report.py 2>&1 | tee -a $LOG

echo -e "\n########## 2. GAP-FILL BAND (structural fix) ##########" | tee -a $LOG
python analysis/ss/gap_band_eval.py --device cpu 2>&1 | tee -a $LOG

echo -e "\n########## 3. AUTOREGRESSIVE GAP-FILL (all methods + trees) ##########" | tee -a $LOG
python analysis/ss/autoregressive_gapfill.py 2>&1 | tee -a $LOG

echo -e "\n########## 4. PUBLICATION FIGURES ##########" | tee -a $LOG
python analysis/ss/loso_plots.py 2>&1 | tee -a $LOG

echo -e "\nDOWNSTREAM PIPELINE COMPLETE" | tee -a $LOG
