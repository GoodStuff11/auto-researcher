#!/bin/bash
# P4.1 round 4, step 1: tabular completeness of "one AGP per charge configuration".
# For the 4x3 OPEN cluster (U = 8, t' = -0.25) at N_e = 8, 10, 12 (4+4, 5+5, 6+6): ED psi_0 (cached in
# results/psi0_*.npz), then per charge configuration the best fidelity of the tabular families
# tocchio K = 1 (one symmetric pairing per c = Tocchio / SQ-RVB class), K = 2 (Route 4.2) and projected UHF
# (Che class). Output: results/agp_<tag>.json and logs/agp_<jobid>.out. Login-node smoke:
#   python3 r4_test.py --mode agp --Lx 4 --Ly 2 --Ne 6 --outdir results_smoke   (11 s)
#SBATCH --job-name=P4.1_agp_r4
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r4/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r4/logs/%x_%j.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r4
PY=/share/apps/software/anaconda3/bin/python3
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd "$DIR"
for NE in 8 10 12; do
  $PY -u r4_test.py --mode agp --Lx 4 --Ly 3 --Ne $NE --U 8 --tp -0.25 --bc-x obc --bc-y obc \
      --agp-restarts 3 --agp-maxiter 3000
done
