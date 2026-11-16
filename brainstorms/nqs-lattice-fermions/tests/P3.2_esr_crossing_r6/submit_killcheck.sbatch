#!/bin/bash
# P3.2 round 6 kill check of ESR-at-a-crossing (K1 / ESR0 Ritz with a trained penalty excited state).
# Array task i -> round-4 seed (1234, 2345, 3456)[i], width 4, targets d = -0.0005, -0.0002, +0.0005.
# Cost (smoke-measured 1.15 s/step at 8 threads, width 4): 3 targets x (2 warm x 800 + 800 phi) steps
# = 7200 steps ~ 2.3 h per task; 6 h limit. Read the verdict with:
#   python3 esr_killcheck.py --aggregate results
#SBATCH --job-name=P3.2r6_esrkc
#SBATCH --partition=kim
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P3.2_esr_crossing_r6/P3.2r6_esrkc_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P3.2_esr_crossing_r6/P3.2r6_esrkc_%A_%a.err
cd /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P3.2_esr_crossing_r6
SEEDS=(1234 2345 3456)
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
/share/apps/software/anaconda3/bin/python3 esr_killcheck.py --width 4 --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]} \
    --targets=-0.0005,-0.0002,0.0005
