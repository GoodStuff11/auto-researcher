#!/bin/bash
#SBATCH --job-name=P3.1_aritz
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P3.1_order_parameter_error/P3.1_aritz_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P3.1_order_parameter_error/P3.1_aritz_%j.err
# P3.1 decisive test: A-Ritz order-parameter error estimator vs one-Lanczos-step
# variance extrapolation on ED-sized doped t-t'-U Hubbard clusters (CPU only).
# Primary pre-committed run: --subspace own3 --estimator vex, exact matrix elements.
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P3.1_order_parameter_error
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$DIR"
$PY micro_test.py \
  --clusters "4,3,8,cyl;4,3,8,pbc;4,3,10,pbc;6,2,8,cyl" \
  --tp "-0.35,-0.25,-0.15" --U 8.0 \
  --subspace own3 --estimator vex --mc_samples 0 \
  --n_seeds 2 --mf_maxiter 300 --seed 1234 \
  --out "$DIR/results_primary_${SLURM_JOB_ID}.json"
# Pre-committed AMBIGUOUS follow-up (run only if the primary verdict is AMBIGUOUS):
# $PY micro_test.py --clusters "4,3,8,cyl;4,3,8,pbc;4,3,10,pbc;6,2,8,cyl" --tp "-0.35,-0.25,-0.15" \
#   --subspace own4 --estimator vex --mc_samples 100000 --out "$DIR/results_ambig_${SLURM_JOB_ID}.json"
