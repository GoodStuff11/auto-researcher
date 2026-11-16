#!/bin/bash
#SBATCH --job-name=P3.1r10_t3
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=07:00:00
#SBATCH --array=0-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.1 Round 10, T3 step 2 (2026-11-16): decisive test of route 10.1d with trained Hylleraas response networks
# (m = 3 per pass, T = 4, width W, shift E_L1R - 0.1 t) on the 4x3 OBC 4+4 cluster, seeds 11..66, plus the
# matched-compute foils F0 (untrained copies), F1 (ground state trained further), F2 (width 2W) and F3 (critic
# 2026-11-16: the same pipeline with observable-free Krylov sources), plus a sampled-Ritz replay (reported). The ground
# state is reloaded from step 1 (submit_t3_probe.sbatch). Rule, early stop and decision tree: t3_scrd.py
# docstring and the leaf ("T3 design for route 10.1 (2026-11-16)"). JSON rewritten after every pass.
# 6 tasks x 2 CPUs x 7 h = 84 CPU-h allocated; about 35 CPU-h expected (5.8 CPU-h per task; F3 adds 1.25).
# Only after the probe gave TARGET-OK for this W. Submit from this directory: sbatch --export=ALL,W=6 submit_t3.sbatch
# Early-stop check (when every task has written SCRdNN_T2):
#   python3 t3_scrd.py --checkpoint pass2 --aggregate $(ls results/t3/t3_4x3_w${W}_s??.json | paste -sd,)
# Verdict afterwards:
#   python3 t3_scrd.py --aggregate $(ls results/t3/t3_4x3_w${W}_s??.json | paste -sd,)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
W=${W:-6}
SEEDS=(11 22 33 44 55 66)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
test -f "results/t3/params/gs_4x3_w${W}_s${S}.npz" || { echo "missing ground state of seed $S: run the probe first"; exit 1; }
/share/apps/software/anaconda3/bin/python3 t3_scrd.py --mode run --size 4x3 --seed "$S" --gs-width "$W" \
    --out "results/t3/t3_4x3_w${W}_s${S}.json"
