#!/bin/bash
# P5.1 R51.8 `nscale`: second seed (seed0 19000, steps draw 19001-19120, as the S10 second seeds of job 769098) of
# the S12' (6x2) guided rungs, released by RULE.md's second-seed clause (round-11-scoring.md: the 65,536 rung's
# s = 0.1326 lies 0.28 ln-units from s = 0.10, inside the 0.3 ln-unit clause).  Order: 65536 (the rung the clause
# names), then 262144 (the other bracketing rung; s = 0.0686 is 0.38 ln-units from 0.10, so NOT named by the
# clause; added for a matched bracket as at S10; the auditor may strip it).  16384 is not a bracketing rung
# (s 0.251) and is not run.  The exact arm is reused from job 769376 (deterministic; one exact arm per size).
# Prepared round 11, NOT submitted by the worker; submit only through compute_gate.py submit with
# voi_r11_s12p_seed2.md.  The VOI note expects G2 to refuse it (predicted verdict unchanged, p 0.88).
# Wall time (in-job MEASURED, job 769376 task 1, results/S6x2_N10_guided_*_s9000/run.json cpu_min):
#   guided 65,536: 53.1 CPU-min / 120 steps; guided 262,144: 78.4 CPU-min / 120 steps; setup about 1 min each.
#   (53.1 + 78.4 + 2) = 133.5 min x 1.3 = 174 min -> --time 02:55:00.  Per-arm --cpu-cap = measured x 1.3.
# Memory: job 769376 task 1 MaxRSS 3.28 GB (incl. Born 262k) -> 6G.
# Resumability: as submit_r06_s12p_ladder.sbatch: ns_run.py saves theta + history atomically every step
# (state.npz) and resumes when the same command is rerun (same seeds); DONE arms are skipped.  Slurm sends USR1 to
# the batch shell (B:) 5 min before the limit; the trap forwards it to the running ns_run.py, which finishes its
# step, saves TRUNCATED and exits, and no later arm starts.  SIGTERM is handled the same way.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:55:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
ST=starts/S6x2_N10_U8_tp-0.25_pg11_che_seed0_theta.npy
echo "P5.1 R51.8 S12' second seeds on $(hostname), job ${SLURM_JOB_ID:-local}"
STOP=0; CHILD=
trap 'STOP=1; if [ -n "$CHILD" ]; then kill -USR1 "$CHILD" 2>/dev/null || true; fi' USR1 TERM
run() {
  [ "$STOP" = 1 ] && return 0
  local log=$1 cap=$2; shift 2
  $PY -u ns_run.py --Lx 6 --Ly 2 --Ne 10 --start "$ST" --steps 120 --seed0 19000 --cpu-cap "$cap" "$@" >> "$log" 2>&1 &
  CHILD=$!
  local rc=0; wait "$CHILD" || rc=$?
  while kill -0 "$CHILD" 2>/dev/null; do rc=0; wait "$CHILD" || rc=$?; done  # a trapped signal interrupts wait
  CHILD=; return $rc
}
run logs/s12p_guided_N65536_s19000.txt 70 --sampler guided --nsamp 65536
run logs/s12p_guided_N262144_s19000.txt 105 --sampler guided --nsamp 262144
echo "S12' second seeds done (STOP=$STOP)"
