#!/bin/bash
# P5.1 R51.8 `nscale`: second seeds (seed0 19000; seed 1 used 9001-9120, this uses 19001-19120) of the S10 (5x2)
# guided rungs, released by RULE.md's second-seed clause (both bracketing rungs within 0.3 ln-units of s = 0.05;
# round-06-scoring.md).  Order: 65536, 262144 (the RULE release), then 16384 (NOT in RULE's release: the upper
# bracketing rung of the 10% threshold PROPOSED in leaves/P5.1/log/round-06-next.md; the auditor may strip it).
# Prepared round 6, NOT submitted by the worker; submit only through compute_gate.py submit with voi_r06_s10_seed2.md.
# Wall time (interactive CPU-min/step, logs/preflight_S5x2_*.txt and RULE interpolation): 64k 0.042, 256k 0.055,
# 16k 0.032 -> 0.129 x 120 + 0.6 setup = 16.1 min; x 1.8 = 29; x 1.3 = 38 min -> 00:40:00.
# (In-job measured by 763968: 8.3 + 11.6 + 6.0 = 25.9 min, inside the limit.)  Memory 1.2 GB -> 4G.
# Resumability: as submit_r06_s12p_ladder.sbatch (state.npz every step; USR1/TERM forwarded, no later arm starts).
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:40:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
ST=starts/S5x2_N8_U8_tp-0.25_pg11_che_seed0_theta.npy
echo "P5.1 R51.8 S10 second seeds on $(hostname), job ${SLURM_JOB_ID:-local}"
STOP=0; CHILD=
trap 'STOP=1; if [ -n "$CHILD" ]; then kill -USR1 "$CHILD" 2>/dev/null || true; fi' USR1 TERM
run() {
  [ "$STOP" = 1 ] && return 0
  local log=$1; shift
  $PY -u ns_run.py --Lx 5 --Ly 2 --Ne 8 --start "$ST" --steps 120 --seed0 19000 --cpu-cap 30 "$@" >> "$log" 2>&1 &
  CHILD=$!
  local rc=0; wait "$CHILD" || rc=$?
  while kill -0 "$CHILD" 2>/dev/null; do rc=0; wait "$CHILD" || rc=$?; done
  CHILD=; return $rc
}
for N in 65536 262144 16384; do run logs/s10_guided_N${N}_s19000.txt --sampler guided --nsamp $N; done
echo "S10 second seeds done (STOP=$STOP)"
