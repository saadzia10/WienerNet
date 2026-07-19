#!/usr/bin/env bash
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
cd /home/cognitia/Desktop/Work/PhD/WienerNet
LOG=outputs/ss_loso_residual_downstream.log
while ! grep -q "SS LOSO RESIDUAL SWEEP DONE" outputs/ss_loso_residual_run.log 2>/dev/null; do
  done=$(grep -cE "^OK|^FAIL" outputs/ss_loso_residual_run.log 2>/dev/null || echo 0)
  [ "$done" -ge 30 ] && break
  sleep 60
done
echo "residual sweep done ($(grep -cE '^OK' outputs/ss_loso_residual_run.log) OK / $(grep -cE '^FAIL' outputs/ss_loso_residual_run.log) FAIL)" | tee $LOG
echo -e "\n### full LOSO report incl. residual arm ###" | tee -a $LOG
python analysis/ss/loso_report.py 2>&1 | tee -a $LOG
echo -e "\nRESIDUAL DOWNSTREAM COMPLETE" | tee -a $LOG
