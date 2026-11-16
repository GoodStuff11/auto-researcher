#!/bin/bash
#SBATCH --job-name=P4.2_cf_t2
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_closed_falsifier/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_closed_falsifier/%x_%A_%a.err
# P4.2 closed-leaf falsifier, T2 continuation (prepared 2026-11-16, NOT submitted by the agent).
# The T1 runs (480 s LM per family) ended INCONCLUSIVE: the LM was not faster than SR within the cap.
# This job asks the same pre-registered question at convergence on 4x3 OBC 5+5:
#   tasks 0-2  exact-sum LM (lm_retrain.py), 7000 s wall, families base / spin_jas2cs / c8h32
#   tasks 3-5  projected SR (critic_capacity.py --mode train), 3000 steps, same three families
# Rule (unchanged from the T1 pre-registration, r* now over all six runs):
#   FALSIFIED if r* <= 0.77% (reopens P4.2 with "the size-independent plateau is optimiser-limited");
#   CONFIRMED if r* > 1.02%; AMBIGUOUS otherwise. INCONCLUSIVE if neither base run ends below 1.536%.
# Read: python3 lm_retrain.py --aggregate results_t2 ; grep -h RESULT results_t2/sr_*.log
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_closed_falsifier
R5=$DIR/../P4.2_size_transfer_r5
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$DIR/results_t2"
T=${SLURM_ARRAY_TASK_ID}
declare -a V=(base spin_jas2cs c8h32)
declare -a OPTS=("" "--spin-feat --jas2 cs" "--channels 8 --bf-hidden 32")
if [ "$T" -lt 3 ]; then
  cd "$DIR"
  $PY lm_retrain.py --variant "${V[$T]}" --iters 100000 --wall 7000 --out "results_t2/lm_${V[$T]}.json"
else
  i=$((T - 3))
  cd "$R5"
  $PY critic_capacity.py --mode train --tgt 4x3:5 --steps 3000 ${OPTS[$i]} \
      --out "$DIR/results_t2/sr_${V[$i]}.json" > "$DIR/results_t2/sr_${V[$i]}.log" 2>&1
fi
