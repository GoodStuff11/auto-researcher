#!/bin/bash
# Login-node timing probe of `fix_test.py --mode complete` on the 4x3 open 5+5 cluster: 4 processes,
# 2-minute fit budget, output in results_probe/ and logs/complete_probe_N10_login.txt. Sizes the Slurm job.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/share/apps/software/anaconda3/bin/python3 fix_test.py --mode complete --Ne 10 --procs 4 --budget-min 2 \
    --wave 64 --outdir results_probe > logs/complete_probe_N10_login.txt 2>&1
