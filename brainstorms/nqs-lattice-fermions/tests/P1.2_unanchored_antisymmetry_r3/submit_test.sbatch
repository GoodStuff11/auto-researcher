#!/bin/bash
#SBATCH --job-name=P1.2r3_FRD
#SBATCH --partition=kim
#SBATCH --array=0-107
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P1.2 round 3: symmetry- and spin-projected FRD-K / FRD-X versus projected NNBF x Jastrow.
# CPU only. One array task per (instance, seed, arm, SR shift):
#   t = 0..107 -> instance = t // 36 (0: 4x3 open 4+4; 1: 4x3 cylinder 4+4; 2: 6x2 ladder 5+5;
#   all U=8, t'=-0.25), r = t % 36, seed = (1234, 2345, 3456)[r // 12],
#   arm = (ctrl_PS, frdK_PS, frdX_PS, ctrl_U, frdK_U, frdX_U)[(r % 12) // 2],
#   shift = (1e-3, 1e-2)[r % 2]. 1500 SR steps each.
# Cost (measured with --mode timing, 8 threads, on a shared compute node; timing_output.txt):
#   instance 0 (D=245025, |G|=4): ctrl_PS 4.0, frdK_PS 9.0, frdX_PS 6.2, ctrl_U 5.5, frdK_U 5.7,
#   frdX_U 5.0 s/step -> worst 1500 x 9.0 s = 3.8 h.
#   instance 2 (D=627264, |G|=24): ctrl_PS 10.4, frdK_PS 30.2, frdX_PS 15.6, ctrl_U 9.7,
#   frdK_U 15.6, frdX_U 11.6 s/step -> worst 1500 x 30.2 s = 12.6 h (1.9x margin to 24 h).
#   Plus setup (ED + in-sector Lanczos, ~1-4 min), budgets (P1.1: ~4 min each on D=627264;
#   2-3 per task) and the sampled arm (minutes).
#   Memory: P1.1 peaked at 3.4 GB on D=627264; FRD adds (65536 x E) neighbour tables per chunk
#   (< 1 GB); 16G gives > 3x margin.
# Submit from this directory so the logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
# Afterwards (any node, seconds): python3 micro_test.py --aggregate results
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --ctrl-width 32 --sr-iters 1500 --sr-lr 0.05 --sr-samples 2000 \
    --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3 --nb-diag 8 --ns-diag 1000
