#!/bin/bash
# P5.1 R51.8 `nscale` unit 1c (RULE.md rev. 1): the S12' (6x2, N_e 10) ladder.  Prepared round 6, NOT submitted by
# the worker; submit only through compute_gate.py submit with voi_r06_s12p_ladder.md.
# Start: starts/S6x2_N10_U8_tp-0.25_pg11_che_seed0_theta.npy, trained by job 763902 (logs/start_6x2.txt).
# Two array tasks, balanced:
#   task 0: exact arm (N = infinity reference).
#   task 1: guided N = 65536, 262144, 16384, 4096, 1024, then born N = 262144, 65536, 16384, 4096, 1024
#           (guided first: it is the primary statistic; born only feeds the comparator).
# tau 0.005, 120 steps, seed0 9000 (as S10).
# Wall time (CPU-min/step, interactive): exact 1.688 measured (logs/preflight_S6x2_exact_probe.txt); guided 262k
# 0.425 measured (logs/preflight_S6x2_g262144_probe.txt); other rungs from a + b x unique draws, a = 0.077,
# b = 1.06e-5 (fit through the 6x2 guided-262k and 4x3 born-262k pre-flights; unique counts from
# results/info_S6x2_N10.json): guided 1k 0.087, 4k 0.107, 16k 0.142, 64k 0.230, 256k 0.425 (sum 0.991);
# born 1k 0.084, 4k 0.098, 16k 0.127, 64k 0.179, 256k 0.269 (sum 0.757).
#   task 0: 1.688 x 120 + 1 = 204 min; x 1.8 = 367; x 1.3 = 477 min.
#   task 1: 1.748 x 120 + 3 = 213 min; x 1.8 = 384; x 1.3 = 499 min -> --time 08:20:00 for both tasks.
#   (In-job / interactive measured at S10 by job 763968: 1.54-1.60, so 1.8 is conservative.)
# Memory: exact 11 GB measured (6x2 probe) -> 20G.
# Resumability: ns_run.py saves theta + history atomically every step (state.npz) and resumes from it when the same
# command is rerun (same seeds); arms already DONE are skipped.  Slurm sends USR1 to the batch shell (B:) 10 min
# before the limit; the trap forwards it to the running ns_run.py, which finishes its step, saves TRUNCATED and
# exits, and no later arm starts.  A SIGTERM at the limit is handled the same way (at most one step lost).
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=08:20:00
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
echo "P5.1 R51.8 unit 1c (S12' 6x2) task $TASK on $(hostname), job ${SLURM_JOB_ID:-local}"
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
  run logs/s12p_exact.txt 480 --exact
else
  for N in 65536 262144 16384 4096 1024; do run logs/s12p_guided_N$N.txt 120 --sampler guided --nsamp $N; done
  for N in 262144 65536 16384 4096 1024; do run logs/s12p_born_N$N.txt 120 --sampler born --nsamp $N; done
fi
echo "unit 1c task $TASK done (STOP=$STOP)"
