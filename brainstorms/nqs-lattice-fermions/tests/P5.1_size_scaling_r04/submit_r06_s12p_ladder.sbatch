#!/bin/bash
# P5.1 R51.8 `nscale` unit 1c (RULE.md rev. 2 + per-leaf-critic addendum): the S12' (6x2, N_e 10) ladder.
# Revised round 7 per the auditor's CHANGES REQUIRED: guided 1k and born 1k/4k/16k dropped; rev 2 (s = 0.10 on
# steps 61-120) is the only primary statistic.  NOT submitted by the worker; submit only through compute_gate.py
# submit with voi_r06_s12p_ladder.md, after (c) was scored (done: round-07-scoring.md, S10 RESOLVABLE).
# Start: starts/S6x2_N10_U8_tp-0.25_pg11_che_seed0_theta.npy, trained by job 763902 (logs/start_6x2.txt).
# Seeds (RULE.md release order): every arm seed0 9000 (as S10's first seed); second seeds are released only after
# scoring, for a bracketing rung within 0.3 ln-units of s = 0.10, as a separate gated job.
# Two array tasks:
#   task 0: exact arm (N = infinity reference).
#   task 1: guided N = 65536, 16384, 262144, 4096, then born N = 262144, 65536 (comparator only, last).
# tau 0.005, 120 steps.
# Wall time.  Interactive CPU-min/step at 6x2: exact 1.688 measured (logs/preflight_S6x2_exact_probe.txt); guided
# 262k 0.425 measured (logs/preflight_S6x2_g262144_probe.txt); other rungs a + b x unique, a = 0.077, b = 1.06e-5
# (6x2 guided-262k and 4x3 born-262k pre-flights; unique counts in results/info_S6x2_N10.json): guided 4k 0.107,
# 16k 0.142, 64k 0.230, 256k 0.425; born 64k 0.179, 256k 0.269.  In-job rates = interactive x the in-job /
# interactive ratio measured by job 763968 at S10 (results/S5x2_*/run.json vs logs/preflight_S5x2_*): exact 1.60;
# sampled arms up to 1.75 (guided 262k 0.0964 in-job vs 0.055).
#   task 0: (1.688 x 120 + 1) = 204 min x 1.60 = 326 in-job x 1.3 = 424 min.
#   task 1: (1.352 x 120 + 6 x 0.3) = 164 min x 1.75 = 287 in-job x 1.3 = 373 min.
#   -> --time 07:10:00 (430 min) for both tasks.
# Memory: exact 11 GB measured (6x2 probe) -> 20G.
# Resumability: ns_run.py saves theta + history atomically every step (state.npz) and resumes when the same command
# is rerun (same seeds); DONE arms are skipped.  Slurm sends USR1 to the batch shell (B:) 10 min before the limit;
# the trap forwards it to the running ns_run.py, which finishes its step, saves TRUNCATED and exits, and no later
# arm starts (tested with a mock).  SIGTERM at the limit is handled the same way (at most one step lost).
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=07:10:00
#SBATCH --array=0-1
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
ST=starts/S6x2_N10_U8_tp-0.25_pg11_che_seed0_theta.npy
TASK=${SLURM_ARRAY_TASK_ID:-0}
echo "P5.1 R51.8 unit 1c (S12' 6x2, rev 2) task $TASK on $(hostname), job ${SLURM_JOB_ID:-local}"
STOP=0; CHILD=
trap 'STOP=1; if [ -n "$CHILD" ]; then kill -USR1 "$CHILD" 2>/dev/null || true; fi' USR1 TERM
run() {
  [ "$STOP" = 1 ] && return 0
  local log=$1 cap=$2; shift 2
  $PY -u ns_run.py --Lx 6 --Ly 2 --Ne 10 --start "$ST" --steps 120 --seed0 9000 --cpu-cap "$cap" "$@" >> "$log" 2>&1 &
  CHILD=$!
  local rc=0; wait "$CHILD" || rc=$?
  while kill -0 "$CHILD" 2>/dev/null; do rc=0; wait "$CHILD" || rc=$?; done  # a trapped signal interrupts wait
  CHILD=; return $rc
}
if [ "$TASK" = 0 ]; then
  run logs/s12p_exact.txt 420 --exact
else
  for N in 65536 16384 262144 4096; do run logs/s12p_guided_N$N.txt 120 --sampler guided --nsamp $N; done
  for N in 262144 65536; do run logs/s12p_born_N$N.txt 120 --sampler born --nsamp $N; done
fi
echo "unit 1c task $TASK done (STOP=$STOP)"
