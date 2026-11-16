#!/bin/bash
# P4.1 size-scaling: DMRG references (dmrg_ref.jl) for the open Lx x 3 ladder (amendment 2026-11-16, second).
# The VMC arms are projected onto (Mx, My) = (+,+), so the reference is the My = +1 SECTOR energy
# (--sector=1, penalty DMRG), with the My = -1 sector for the sector gap and 2S^z = 2 for the spin check.
# Array tasks: 0 = 4x3 Sz2=2 (12-site sectors were run interactively against ED);
#   1/2/3 = 8x3  (My=+1, My=-1, Sz2=2), 4/5/6 = 12x3 (same), 7/8/9 = 16x3 (same).
#SBATCH --job-name=P41scal_dmrg
#SBATCH --partition=kim
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_scaling/logs/dmrg_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_scaling/logs/dmrg_%A_%a.err
cd /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_scaling
LX=(4 8 8 8 12 12 12 16 16 16)
SZ=(2 0 0 2 0 0 2 0 0 2)
SE=(0 1 -1 0 1 -1 0 1 -1 0)
i=$SLURM_ARRAY_TASK_ID
case ${LX[$i]} in
  4)  MD="100,200,400,600,800,1000,1200"; SW=4 ;;
  8)  MD="200,400,800,1200,1600,2000,2500"; SW=8 ;;
  12) MD="200,400,800,1200,1600,2000,2500,3000"; SW=6 ;;
  16) MD="200,400,800,1200,1600,2000,2500,3000,3500"; SW=6 ;;
esac
export OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
~/.juliaup/bin/julia --project=dmrg_env -t 1 dmrg_ref.jl --Lx=${LX[$i]} --Sz2=${SZ[$i]} --sector=${SE[$i]} --maxdims=$MD --sweeps=$SW --noise=1e-4 --threads=16
