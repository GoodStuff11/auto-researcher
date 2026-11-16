#!/bin/bash
#SBATCH --job-name=p22r5
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-43
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r5/p22r5_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r5/p22r5_%A_%a.err
# CPU only. P2.2 round-5 decisive test, critic-fix revision (route R5.1, WIBJ).
# Part L (route R5.2, ALE) is DEFERRED and not in this array.
#  Tasks  0-35 : Part W (accuracy, ED, L-BFGS on the exact energy), instance x arm:
#                instances H4 R=4.4 K = 12 16 20 24 (the K-trend) and H6 R=3.6 K = 12 14;
#                arms NNBFJ (seeds 0-2), NNBFJNF (seeds 3-5, noise floor), WIBJ (0-2),
#                NNBFJh4 and NNBFJh8 (minimal-contrast foils, 0-2), CASJ (window
#                multi-determinant x pair Jastrow, no NN, 0-2); hidden 64, hJ 32,
#                6000 L-BFGS iterations.
#  Tasks 36-43 : Part S (ED-free cost), distorted H8 chain R=2.0, bases (2,1) (3,2) (4,3)
#                (5,4) -> K = 40 72 104 136; per basis two tasks: main (NNBFJ + WIBJ,
#                timed in the same task) and foils (NNBFJh4, NNBFJh8, CASJ op counts);
#                n-wall 60.
# Cost estimates (scaled from pilot 411919: H4 K=20, 3000 its, ~900 s per seed):
#   H4 K=24 ~1 h per seed; H6 K=14 (dim 132496) ~2 h per seed; CASJ on H6 K=14
#   (0.7 s per L-BFGS iteration measured at K=12, ~3x at K=14) -> <= ~10 h per task;
#   Part S K=136 main ~4 h.
# Verdict (pre-committed; printed by):
#   python3 micro_test.py --aggregate <this dir>
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r5
cd "$DIR"
PY=/share/apps/software/anaconda3/bin/python3
T=$SLURM_ARRAY_TASK_ID
J=$SLURM_ARRAY_JOB_ID
if [ "$T" -lt 36 ]; then
    WA=(NNBFJ NNBFJNF WIBJ NNBFJh4 NNBFJh8 CASJ); ARM=${WA[$((T % 6))]}; C=$((T / 6))
    case $C in
        0) NA=4; BL=4.4; K=12 ;;
        1) NA=4; BL=4.4; K=16 ;;
        2) NA=4; BL=4.4; K=20 ;;
        3) NA=4; BL=4.4; K=24 ;;
        4) NA=6; BL=3.6; K=12 ;;
        5) NA=6; BL=3.6; K=14 ;;
    esac
    $PY micro_test.py --part W --n-atoms $NA --bond-length $BL --n-active $K --w-arms $ARM \
        --hidden 64 --hj 32 --maxiter 6000 --n-samples 2000 \
        --out "$DIR/results_W_H${NA}_K${K}_${ARM}_${J}.json"
else
    I=$((T - 36)); BS=("2,1" "3,2" "4,3" "5,4"); B=${BS[$((I / 2))]}; GS=(main foils); G=${GS[$((I % 2))]}
    $PY micro_test.py --part S --n-atoms 8 --bond-length 2.0 --basis-series "$B" --s-arms $G \
        --n-samples 200 --n-wall 60 \
        --out "$DIR/results_S_b$((I / 2))_${G}_${J}.json"
fi
