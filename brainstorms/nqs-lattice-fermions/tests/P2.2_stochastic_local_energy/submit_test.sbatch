#!/bin/bash
#SBATCH --job-name=p22_scv_eloc
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy/p22_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy/p22_%j.err
# CPU only. Decisive test for P2.2 (frozen-backflow control-variate local energy).
# Instances: H6 and H8 STO-3G chains at R = 1.8 and 3.0 bohr (3.0 = Hard-Core instance).
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P2.2_stochastic_local_energy
cd "$DIR"
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --sizes 6 8 --bond-lengths 1.8 3.0 --geometry chain \
    --hidden 32 --maxiter 3000 --snapshots 50 300 1000 \
    --kappa 1e-3 --bias-tol 1.6e-4 --mix-alpha 0.5 --n-eps 30 \
    --out "$DIR/results_${SLURM_JOB_ID}.json"
