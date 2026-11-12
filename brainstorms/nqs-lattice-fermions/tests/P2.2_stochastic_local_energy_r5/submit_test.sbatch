#!/bin/bash
#SBATCH --job-name=p22r5
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-83
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r5/p22r5_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r5/p22r5_%A_%a.err
# CPU only. P2.2 round-5 decisive test (single-thread timings inside micro_test.py).
#  Tasks  0-67 : Part L (route R5.2, anchored local energy), H4 distorted chain
#                R=4.4 (HF weight ~0.46-0.49), K = 16 (tasks 0-33) and K = 20
#                (tasks 34-67); one estimator arm per task (34 arms: EXACT,
#                EXACTNF noise floor, 4 SCREEN, 6 SS, 4 ALE0, 2 ALE1, every
#                non-EXACT arm also in +B equal-step-cost mode); SR seeds 0 1 2
#                (EXACTNF: 10 11 12). Common warm start: 400 L-BFGS its of NNBF-J
#                (hidden 32, hJ 16). B = 500, budget = 200 EXACT steps.
#  Tasks 68-79 : Part W (route R5.1, WIBJ accuracy), instance x K x arm:
#                H4 R=4.4 K = 20, 24 and H6 R=3.6 K = 12, 14; arms NNBFJ (seeds
#                0-2), NNBFJNF (NNBFJ seeds 3-5, noise floor), WIBJ (seeds 0-2);
#                hidden 64, 6000 L-BFGS iterations on the exact energy.
#  Tasks 80-83 : Part S (route R5.1, ED-free cost), distorted H8 chain R=2.0,
#                bases (2,1) (3,2) (4,3) (5,4) -> K = 40 72 104 136, arms NNBFJ, WIBJ.
# Cost estimates (scaled from the login-node pre-checks: Part L K=12 ~0.1 s per
# EXACT SR step at B=500): Part L K=20 <= ~6 h per task (+B tasks with B=2000
# ~10 s per step); Part W H6 K=14 (dim 132496) ~2.5 h per seed -> ~8 h per task;
# Part S K=136 ~3 h.
# Verdict (pre-committed; printed by):
#   python3 micro_test.py --aggregate <this dir>
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy_r5
cd "$DIR"
PY=/share/apps/software/anaconda3/bin/python3
T=$SLURM_ARRAY_TASK_ID
J=$SLURM_ARRAY_JOB_ID
BASE=(SCREEN:1e-3 SCREEN:3e-3 SCREEN:1e-2 SCREEN:3e-2
      SS:1e-3:4 SS:1e-3:64 SS:1e-2:4 SS:1e-2:64 SS:3e-2:4 SS:3e-2:64
      ALE0:4:4 ALE0:4:32 ALE0:16:4 ALE0:16:32 ALE1:4:4 ALE1:16:32)
LARMS=(EXACT EXACTNF)
for a in "${BASE[@]}"; do LARMS+=("$a" "$a+B"); done   # 2 + 32 = 34 arms
if [ "$T" -lt 68 ]; then
    KS=(16 20); K=${KS[$((T / 34))]}; A=${LARMS[$((T % 34))]}
    $PY micro_test.py --part L --n-atoms 4 --bond-length 4.4 --n-active "$K" --arm "$A" \
        --warm-iters 400 --batch 500 --n-ref 200 --seeds 0 1 2 \
        --out "$DIR/results_L_K${K}_${A//:/-}_${J}.json"
elif [ "$T" -lt 80 ]; then
    I=$((T - 68)); WA=(NNBFJ NNBFJNF WIBJ); ARM=${WA[$((I % 3))]}; C=$((I / 3))
    case $C in
        0) NA=4; BL=4.4; K=20 ;;
        1) NA=4; BL=4.4; K=24 ;;
        2) NA=6; BL=3.6; K=12 ;;
        3) NA=6; BL=3.6; K=14 ;;
    esac
    $PY micro_test.py --part W --n-atoms $NA --bond-length $BL --n-active $K --w-arms $ARM \
        --hidden 64 --hj 32 --maxiter 6000 --n-samples 2000 \
        --out "$DIR/results_W_H${NA}_K${K}_${ARM}_${J}.json"
else
    BS=("2,1" "3,2" "4,3" "5,4"); B=${BS[$((T - 80))]}
    $PY micro_test.py --part S --n-atoms 8 --bond-length 2.0 --basis-series "$B" --n-samples 200 --n-wall 12 \
        --out "$DIR/results_S_b$((T - 80))_${J}.json"
fi
