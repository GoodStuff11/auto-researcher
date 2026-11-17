#!/bin/bash
#SBATCH --job-name=P2.1r14_halfF16_V15
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# P2.1 route R12.3: the one remaining task of Round 12d (job 706240, task 3, cancelled at step 150 with no
# checkpoint): `half` at F=16 on instance 25 (5x3 OBC 7+7, V=15). Decides comparison A (half_F16 vs full_F4).
# Code unchanged: ../P2.1_subcubic_amplitudes_r12d/ladder_d.py --task 3 (ladder.train, t1_sub.decide unchanged).
# Result lands in ../P2.1_subcubic_amplitudes_r12d/results/half_F16/t1_half_inst25_s11.json next to V=8,12,14,
# checkpoints every 250 steps in .../half_F16/ckpt/ (a resubmission resumes). The script refuses to overwrite.
# Wall time: measured 1.43 CPU-s/step at V=15 (706240 task 3, steps 0-150); worst case 6000 steps = 8580 s
#   + ~2 min ED/setup/budget = 2.42 h, +30% = 3.15 h -> 03:10:00. The SR CPU cap (10800 s = 3.0 h) also ends
#   the run inside this limit if the node is slower. Expected (plateau at 650-1600 steps): 0.3-0.7 h.
# Verdict: cd ../P2.1_subcubic_amplitudes_r12d && python3 ladder_d.py --mode aggregate   (comparison A)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/share/apps/software/anaconda3/bin/python3
R12D="${SLURM_SUBMIT_DIR}/../P2.1_subcubic_amplitudes_r12d"
cd "${R12D}"
${PY} ladder_d.py --mode train --task 3
