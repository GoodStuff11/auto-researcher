#!/bin/bash
#SBATCH --job-name=P3.2_biased_ensemble
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P3.2_biased_ensemble/slurm_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P3.2_biased_ensemble/slurm_%j.err
# P3.2 decisive test (CPU only). Run (a) gives the pre-registered verdict;
# (b) and (c) are the pre-committed robustness runs (MC noise; better references).
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P3.2_biased_ensemble
PY=/share/apps/software/anaconda3/bin/python3
cd "$DIR"
echo "=== (a) primary: 6x2 po ladder, exact expectation values ==="
$PY micro_test.py --out "$DIR/results_a_exact.json"
echo "=== (b) robustness: Monte Carlo noise, N=10000 samples ==="
$PY micro_test.py --n-samples 10000 --out "$DIR/results_b_mc1e4.json"
echo "=== (c) robustness: better references (f0=0.3, q=0.05) ==="
$PY micro_test.py --f0 0.3 --q 0.05 --out "$DIR/results_c_goodref.json"
