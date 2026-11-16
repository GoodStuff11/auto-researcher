#!/bin/bash
#SBATCH --job-name=p22r3_eloc
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-26
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r3/p22r3_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r3/p22r3_%A_%a.err
# CPU only. P2.2 round-3 decisive test.
#   R3.1 window-input backflow x Jastrow (exact local energy by window keys)
#   R3.2 window-keyed partially-frozen control variate (PFCV) on the NNBF-J state
#   (SHRJ: external-input-shrunk NNBF-J, diagnostic only)
#   controls: EXACT / SCREEN / SS on the unrestricted NNBF-J state (same seed).
# H_n rings at R = 2.8 bohr, 6-31G** (with p functions), active spaces of
# valence + p-type + remaining MOs:  N = 6: K = 8 10 12 14 16;  N = 8: K = 9 10 11 12.
# Task t -> instance t / 3, arm t % 3; each task runs seeds 0 1 2.
# Timings are single-thread (OMP/BLAS threads = 1 inside micro_test.py) so
# that estimator wall-clock costs are comparable; the 2nd CPU absorbs noise.
# Cost estimate (measured, login node, single thread, 400 L-BFGS its, 400 samples;
# precheck_N6_K8-12.log): training 3 s / 22 s / 82 s per arm-seed at dim
# 3136 / 14400 / 48400 (~0.2 s per iteration at 48400), analysis 1-10 s per 400
# samples.  Scaling with dim x nnz/row to the largest tasks (N=6 K=16, dim
# 313600; N=8 K=12, dim 245025): ~1.5 s/iteration -> ~40 min per arm-seed at
# 1500 its, ~2.5 h per task (3 seeds) incl. ED and 2000-sample analysis;
# peak memory ~8 GB (H with ~2e8-4e8 nonzeros).
# Verdict (pre-committed, printed by):
#   python3 micro_test.py --aggregate <this dir>
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r3
cd "$DIR"
NS=(6 6 6 6 6 8 8 8 8)
KS=(8 10 12 14 16 9 10 11 12)
ARMS=(NNBFJ WIBJ SHRJ)
I=$((SLURM_ARRAY_TASK_ID / 3))
A=${ARMS[$((SLURM_ARRAY_TASK_ID % 3))]}
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --n-atoms "${NS[$I]}" --n-active "${KS[$I]}" --bond-length 2.8 \
    --arms "$A" --seeds 0 1 2 --hidden 48 --maxiter 1500 --mu 1e-2 \
    --n-samples 2000 --bias-tol 1.6e-4 \
    --out "$DIR/results_N${NS[$I]}_K${KS[$I]}_${A}_${SLURM_ARRAY_JOB_ID}.json"
