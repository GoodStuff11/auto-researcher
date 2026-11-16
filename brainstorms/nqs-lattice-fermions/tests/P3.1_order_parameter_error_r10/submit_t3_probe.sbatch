#!/bin/bash
#SBATCH --job-name=P3.1r10_t3probe
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.1 Round 10, T3 step 1 (2026-11-16): target probe for the 10.1d T3 on the 4x3 OBC 4+4 t-t'-J cluster.
# Trains the width-W ground state of one seed (1200 SR steps, P2.1 r3 settings), scores the tails, saves the
# parameters (reused by step 2) and stops. 6 tasks x 2 CPUs x 1 h = 12 CPU-h allocated, about 3 CPU-h expected.
# Gate afterwards (seconds, any node):
#   python3 t3_scrd.py --checkpoint target --aggregate $(ls results/t3/t3_4x3_w${W}_s*_probe.json | paste -sd,)
# TARGET-OK (>= 9 scored items) -> submit_t3.sbatch with the same W; NO-TARGET -> rerun this file with W=3, then W=2
# (pre-registered width ladder; leaf, "T3 design for route 10.1 (2026-11-16)").
# Submit from this directory: sbatch --export=ALL,W=6 submit_t3_probe.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
W=${W:-6}
SEEDS=(11 22 33 44 55 66)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
/share/apps/software/anaconda3/bin/python3 t3_scrd.py --mode run --size 4x3 --seed "$S" --gs-width "$W" \
    --stop-after A --out "results/t3/t3_4x3_w${W}_s${S}_probe.json"
