#!/bin/bash
#SBATCH --job-name=P2.1r15_polish_V15
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# P2.1 route R15.1 (revival of R12.3, L51): polished re-score of half_F16 against full_F4 at V = 15 (instance 25).
# Rule: RULE.md (KILL if rho_pol(15) > 1.25; PASS <= 1.15 licenses the V = 12, 14 polish as a new test; else AMBIGUOUS).
# Per arm (sequential, one CPU): ED + calibration, recover the scored state (half_F16: exact SR resume from the step-500
# checkpoint to step 526, replay verified bit-exact in the probe), replay budget, L-BFGS polish (<= 300 it or 480 CPU-s),
# polished budget. Results: results/polish_<arm>_inst25.json (the script refuses to overwrite); verdict printed by aggregate.
# Wall time from local probes (probe_*_output.txt, single thread): per arm setup+recovery <= 46 s, budget 8-9 s (x2),
#   L-BFGS 480 s cap (+ <= 3 s for the iteration in progress) -> <= 555 s per arm, 1110 s for both = 18.5 min;
#   +30% = 24 min -> 00:25:00 (cap 0.42 CPU-h).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/share/apps/software/anaconda3/bin/python3
${PY} polish.py --mode polish --arm full_F4
${PY} polish.py --mode polish --arm half_F16
${PY} polish.py --mode aggregate
