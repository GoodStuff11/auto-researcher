#!/bin/bash
# P4.1 size-scaling: DMRG references (dmrg_ref.jl) for the open Lx x 3 ladder.
# Array tasks: 0 = 4x3 Sz2=2 (spin check; Sz2=0 at 4x3 was run interactively and matches ED);
#   1/2 = 8x3 Sz2=0/2, 3/4 = 12x3 Sz2=0/2, 5/6 = 16x3 Sz2=0/2.
#SBATCH --job-name=P41scal_dmrg
#SBATCH --partition=kim
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-6
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_scaling/logs/dmrg_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_scaling/logs/dmrg_%A_%a.err
cd /home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_scaling
LX=(4 8 8 12 12 16 16)
SZ=(2 0 2 0 2 0 2)
i=$SLURM_ARRAY_TASK_ID
case ${LX[$i]} in
  4)  MD="100,200,400,600,800,1000,1200"; SW=4 ;;
  8)  MD="200,400,800,1200,1600,2000,2500"; SW=4 ;;
  12) MD="200,400,800,1200,1600,2000,2500,3000"; SW=5 ;;
  16) MD="200,400,800,1200,1600,2000,2500,3000,3500"; SW=5 ;;
esac
export OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
~/.juliaup/bin/julia --project=dmrg_env -t 1 dmrg_ref.jl --Lx=${LX[$i]} --Sz2=${SZ[$i]} --maxdims=$MD --sweeps=$SW --threads=16
