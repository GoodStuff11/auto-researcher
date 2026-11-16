#!/bin/bash
#SBATCH --job-name=P4.2_r5_critic
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r5/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r5/%x_%A_%a.err
# P4.2 round-5 per-leaf critic diagnostics (critic_capacity.py; see its docstring).
#   task 0  LM stationarity check, 4x4 6+6, base MFNG-O (426 par) theta at step 600 (job 452196)
#   task 1  LM stationarity check, 4x4 6+6, enlarged (818 par) theta at step 600 (job 604157)
#   task 2  component ablation on 4x3 OBC 5+5, 600 projected-SR steps each, 9 variants in parallel
#           (1 thread each): base, c8h32, shift 1e-3, free-F (oracle), spin-feat, jas2 c, jas2 cs,
#           spin-feat + jas2 cs, all three.
# NOTE (2026-11-15): array task 2 was cancelled while pending and the same 9 variants were run on
# the login node (1 thread each, ~10-16 min; logs in results_critic/train_*.log).
# Pre-registered readings (written before any result):
#   LM: gain fraction g = (E - E_LM)/(E - E0) at overlap cut 1e-8 (fixed after the trivial-target
#       smoke, before any 4x4 LM result) < 0.10 for both families -> the 4x4 plateau is a
#       tangent-space stationary point, not an lr / SR-shift / step-count artefact.
#   Ablation: a component is THE bottleneck if adding it lowers the 600-step relerr by >= 2x vs
#       base while c8h32 lowers it by < 25% and shift 1e-3 by < 25%.
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.2_size_transfer_r5
PY=/share/apps/software/anaconda3/bin/python3
cd "$DIR"
mkdir -p results_critic
T=${SLURM_ARRAY_TASK_ID}
if [ "$T" -eq 0 ] || [ "$T" -eq 1 ]; then
  export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
  if [ "$T" -eq 0 ]; then J=results_capacity/scr_s0.json; O=lm_c4h16.json; else J=results_capacity_c8h32/scr_s0.json; O=lm_c8h32.json; fi
  $PY critic_capacity.py --mode lm --json "$J" --out "results_critic/$O"
else
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  declare -a NAMES=(base c8h32 shift1e-3 freeF spinfeat jas2c jas2cs spin_jas2cs all)
  declare -a OPTS=("" "--channels 8 --bf-hidden 32" "--diag-shift 1e-3" "--free-f" "--spin-feat" "--jas2 c" "--jas2 cs" "--spin-feat --jas2 cs" "--spin-feat --jas2 cs --free-f")
  for i in "${!NAMES[@]}"; do
    $PY critic_capacity.py --mode train --tgt 4x3:5 --steps 600 ${OPTS[$i]} \
        --out "results_critic/train_${NAMES[$i]}.json" > "results_critic/train_${NAMES[$i]}.log" 2>&1 &
  done
  wait
  grep -h RESULT results_critic/train_*.log
fi
