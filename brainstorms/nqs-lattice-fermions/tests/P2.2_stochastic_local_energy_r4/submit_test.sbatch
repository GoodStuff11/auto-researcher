#!/bin/bash
#SBATCH --job-name=p22r4_keyx
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-33
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r4/p22r4_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r4/p22r4_%A_%a.err
# CPU only. P2.2 round-4 decisive test, two separate parts.
#  Tasks 0-3  : Part S (ED-free scaling), distorted H8 chain R=2.0, bases
#               (ns,np) = (2,1) (3,2) (4,3) (5,4) -> K = 40 72 104 136 MOs (minus
#               linear dependences); all five arms; 200 CISD-like samples, 12 of
#               them with full wall-clock timing of EXACT/SCREEN/TRUNC.
#  Tasks 4-33 : Part A (accuracy, ED). Instance A1 = H6 distorted chain R=3.6
#               (HF weight 0.53), K = 8 10 12; instance A2 = H8 distorted chain
#               R=3.4 (HF weight 0.47), K = 9 10 11. One task per (instance, K, arm);
#               seeds 0 1 2; hidden 64; 3000 L-BFGS iterations on the exact energy.
# Single-thread timings (OMP/BLAS threads = 1 inside micro_test.py).
# Cost estimate: Part S largest task (K=136) ~1 h integrals + ~1.5 h estimators,
# peak ~25 GB. Part A largest task (H8, K=11, dim 108900) ~1-2 s per L-BFGS
# iteration -> ~1.5 h per seed, ~5 h per task. Measured on the login node:
# Part S K=40 (20 samples, 5 arms) 39 s; Part A H6 K=8 (1 seed, 5 arms,
# 1500 iterations) 84 s.
# Verdict (pre-committed; printed by):
#   python3 micro_test.py --aggregate <this dir>
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r4
cd "$DIR"
PY=/share/apps/software/anaconda3/bin/python3
T=$SLURM_ARRAY_TASK_ID
if [ "$T" -lt 4 ]; then
    BS=("2,1" "3,2" "4,3" "5,4")
    $PY micro_test.py --part S --n-atoms 8 --bond-length 2.0 --basis-series "${BS[$T]}" \
        --n-samples 200 --n-wall 12 \
        --out "$DIR/results_S_b${T}_${SLURM_ARRAY_JOB_ID}.json"
else
    I=$((T - 4))
    INST=$((I / 15)); R=$((I % 15)); KI=$((R / 5)); AI=$((R % 5))
    ARMS=(NNBFJ WIBJ CGK XSK FJ)
    if [ "$INST" -eq 0 ]; then NA=6; BL=3.6; KS=(8 10 12); else NA=8; BL=3.4; KS=(9 10 11); fi
    $PY micro_test.py --part A --n-atoms $NA --bond-length $BL --n-active "${KS[$KI]}" \
        --arms "${ARMS[$AI]}" --seeds 0 1 2 --hidden 64 --maxiter 3000 --n-samples 2000 \
        --out "$DIR/results_A_i${INST}_K${KS[$KI]}_${ARMS[$AI]}_${SLURM_ARRAY_JOB_ID}.json"
fi
