#!/bin/bash
# P4.2 decisive test: zero-shot size transfer of a residual-geminal fermionic NQS,
# t-J 2-leg ladder 2x4 -> 2x8 at hole doping 1/4 (target ED dim 1,681,680).
# Runs the pre-committed default protocol (see micro_test.py docstring), then the
# pre-committed AMBIGUOUS follow-ups (antiperiodic BC; window 0) only if the first run is AMBIGUOUS.
#SBATCH --job-name=P4.2_size_transfer
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer/slurm_%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer/slurm_%x_%j.err

set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
cd "$DIR"

$PY micro_test.py --selftest

MAIN="$DIR/results_residual_periodic_L4to8.json"
$PY micro_test.py --out "$MAIN"

VERDICT=$($PY -c "import json;print(json.load(open('$MAIN'))['target']['verdict'])")
echo "main verdict: $VERDICT"
if [ "$VERDICT" = "AMBIGUOUS" ]; then
    $PY micro_test.py --bc antiperiodic --out "$DIR/results_residual_antiperiodic_L4to8.json"
    $PY micro_test.py --window 0 --out "$DIR/results_residual_periodic_w0_L4to8.json"
fi
