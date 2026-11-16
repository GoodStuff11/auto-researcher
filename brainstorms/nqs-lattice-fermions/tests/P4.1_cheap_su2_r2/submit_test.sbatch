#!/bin/bash
# P4.1 round-2 decisive test: exact-SU(2) routes (pcg = projected charge-conditioned Gaussian,
# scf = spin-charge factorised SU(2) MPS) versus a parameter-matched spin-resolved NNBF, the
# mVMC-like foil (P_S P_K, Gutzwiller + charge Jastrow + doublon-holon) and the round-1 r1
# ablation. 4x3 PBC Hubbard, N_up = N_dn = 4, U = 8, exact sums (dim 245025).
# Array 0-29: instance = task / 15  (0: t' = -0.25, P1.1 instance 1, sign half only;
#                                     1: t' = -0.33, in-sector avoided crossing, gap 0.273 t),
#             arm = (task % 15) / 3 in (foil pcg scf nnbf r1), seed = task % 3.
# Same budget for every arm: L-BFGS <= 3000 iterations, <= 16 h wall clock, same seeds,
# same J(c) correlator, same K=0 projection; matched arms (pcg, scf, nnbf, r1) within
# ~5% of the pcg parameter count (2149 at --hidden 8).
# Cost estimate (measured on the login node with `--mode time`, 4x3 N=8 at t'=-0.25, one
#   energy+gradient, shared 20-core node): foil 1.8 s, pcg 2.9 s, nnbf 3.4 s, r1 2.7 s,
#   scf 23.1 s (D = 13 SU(2) MPS). L-BFGS uses ~1.1 evaluations per iteration, so 3000
#   iterations take 1.5-3.5 h for foil/pcg/nnbf/r1; scf reaches the common 16 h wall cap
#   after ~2300 iterations (equal wall-clock budget, logged in the JSON as "iters").
#   ED (S=0, K=0, 2 levels) ~1-2 min; observables ~5 min; sign budget (10 fixed-node eigsh +
#   4 L-BFGS-B starts on D = 245025, as in P1.1 job 329057) ~1-2 h, twice for nnbf
#   (raw and post-hoc projected). Worst case ~20 h < 24 h.
# Afterwards (login node, seconds):
#   python3 micro_test.py --mode report --tp -0.25
#   python3 micro_test.py --mode report --tp -0.33
#SBATCH --job-name=P4.1_su2_r2
#SBATCH --partition=kim
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r2/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r2/logs/%x_%A_%a.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r2
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
TPS=(-0.25 -0.33)
ARMS=(foil pcg scf nnbf r1)
ID=${SLURM_ARRAY_TASK_ID}
TP=${TPS[$((ID / 15))]}
ARM=${ARMS[$(((ID % 15) / 3))]}
SEED=$((ID % 3))
cd "$DIR"
$PY -u micro_test.py --mode run --Lx 4 --Ly 3 --Ne 8 --U 8 --tp "$TP" --kproj 1 \
    --ansatz "$ARM" --seed "$SEED" --hidden 8 --mps-mult 2,2,1,1 \
    --maxiter 3000 --max-hours 16 --chunk 16384 --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3
