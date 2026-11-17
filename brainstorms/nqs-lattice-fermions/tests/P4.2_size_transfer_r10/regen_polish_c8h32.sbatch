#!/bin/bash
#SBATCH --job-name=P4.2r10_c8h32
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r10/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r10/logs/%x_%j.err
# P4.2 r10, arm 1 of 3 (released one at a time): regenerate the 704844 SR-3000 c8h32 state with theta saved,
# then polish it with the truncated-metric exact-sum LM. Rule: RULE.md. Gated by compute_gate.py (voi_c8h32.md).
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r10
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$DIR"
if [ ! -f results/sr3000_c8h32.json ]; then
  $PY regen_sr.py --variant c8h32 --steps 3000
fi
$PY polish.py --variant c8h32 --theta-json results/sr3000_c8h32.json --steps 8 --thr 1e-5 --tau 0.05
