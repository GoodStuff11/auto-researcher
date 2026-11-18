#!/bin/bash
# P5.1 R51.8 `nscale`, round 13: the S14 (7x2, N_e 12) minimal ladder + the S12' 65k second seed, under RULE.md
# rev. 2 + addenda, including "Addendum (2026-11-18, before any 7x2 data)" (step-2 reference S10, rungs, seed
# fold-in, STOP rule).  Prepared round 13, NOT submitted by the worker; submit only through compute_gate.py submit
# with voi_r13_s14_ladder.md.
#
# Work (in this order; each item resumable, DONE items skipped):
#   W0  S12' (6x2) guided 65,536, seed0 19000           (step-1 second seed, ~1% of cost)
#       -> score_r13_step1.py; if r' is no longer SUPPORT-LIKE: write results/STOP_r13 and run nothing at S14
#   W1  S14 exact arm (N = infinity reference)          (critical path, ~75% of cost)
#   W2  S14 guided 1,048,576, seed0 9000
#   W3  S14 guided   262,144, seed0 9000
# Structure.  Partition kim has MaxTime UNLIMITED (`scontrol show partition kim`), so no Slurm limit forces chunks;
# the chunking keeps the gate's cap (array x time x cpus) equal to the CPU actually reserved.  The array is 5
# identical 48-h workers run ONE AT A TIME (%1, so no two workers ever touch the same arm).  Each worker walks
# W0..W3, skips DONE items, and runs the first unfinished one with --cpu-cap = its remaining wall minus 20 min;
# ns_run.py stops before a step that would cross the cap, saves state.npz and returns, and the worker moves on to
# the next item if time remains, else exits; the next array task resumes from state.npz.  A worker that finds
# everything DONE (or STOP_r13) exits in seconds.  Exact arm needs about 4 workers, guided arms about 1.
# Wall time (in-job MEASURED rates x 1.3; logs/cost_r11_s14.txt):
#   W1 exact: 763903's in-job 7x2 kappa (154.3 CPU-min for two exact row passes + energies, ratio 0.89 exact-step /
#       kappa-pass measured at 6x2) -> 68.6 CPU-min/step (the cost fit gives 58.9; the larger is used) x 120
#       = 137.2 h; x 1.3 = 178.4 h.
#   W2 guided 1M: 8.99 CPU-min/step x 120 = 18.0 h; x 1.3 = 23.4 h.  W3 guided 262k: 6.42 -> 12.8 h; x 1.3 = 16.7 h.
#   W0: 53.1 CPU-min in-job (769376) + 1 setup = 54 min; x 1.3 = 1.2 h.
#   Per 7x2 invocation setup: 763903 elapsed 196.6 min vs 154.3 timed -> about 42 min (sector + Hr build) + 2
#   energy passes; ~7 invocations x 50 min x 1.3 = 7.6 h; tail waste (step that would cross a worker's cap)
#   <= 1 exact step per worker, about 5 x 1 h.
#   Total = 178.4 + 23.4 + 16.7 + 1.2 + 7.6 + 5 = 232 h <= 5 x 48 h = 240 h (cap).  Expected use 156-175 CPU-h.
# Memory: 763903 MaxRSS 12.8 GB (7x2 exact passes); 7x2 info probe 16.3 GB maxrss -> 32G for every worker.
# Signals: Slurm sends USR1 to the batch shell (B:) 20 min before each worker's limit, and SIGTERM on preemption
# (partition PreemptMode=REQUEUE; --requeue lets a preempted worker rerun the same script, which resumes).  The
# trap forwards the signal to ns_run.py, which finishes the current step, saves TRUNCATED and exits; no later item
# starts in that worker.  At most one step is lost if the step outruns the limit.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4%1
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%A_%a.err
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR" || exit 1
LIMIT_MIN=${LIMIT_MIN:-2880}          # keep equal to --time (48 h)
T0=$(date +%s)
ST6=starts/S6x2_N10_U8_tp-0.25_pg11_che_seed0_theta.npy
ST7=starts/S7x2_N12_U8_tp-0.25_pg11_che_seed0_theta.npy
echo "P5.1 R51.8 round 13 worker ${SLURM_ARRAY_TASK_ID:-local} on $(hostname), job ${SLURM_JOB_ID:-local}, $(date)"
STOP=0; CHILD=
trap 'STOP=1; if [ -n "$CHILD" ]; then kill -USR1 "$CHILD" 2>/dev/null || true; fi' USR1 TERM
remaining() { echo $(( LIMIT_MIN - ($(date +%s) - T0) / 60 - 20 )); }
status() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]+'/run.json')).get('status','NONE'))" "$1" 2>/dev/null || echo NONE; }
# item <results dir> ... returns 0 DONE, 1 stop this worker (cap, signal, too little time), 2 error
# go: run an item; exit the worker on 1 (exit 0) or 2 (exit 1)
go() { item "$@"; local r=$?; [ "$r" = 2 ] && exit 1; [ "$r" = 0 ] || exit 0; }
# item <results dir> <min minutes to be worth starting> <log> <ns_run args...>
item() {
  local res=$1 need=$2 log=$3; shift 3
  [ "$STOP" = 1 ] && return 1
  [ "$(status "$res")" = DONE ] && return 0
  local rem; rem=$(remaining)
  if [ "$rem" -lt "$need" ]; then echo "$(date) $res: $rem min left < $need; worker ends"; return 1; fi
  echo "$(date) $res: start/resume with --cpu-cap $rem"
  $PY -u ns_run.py --steps 120 --tau 0.005 --res "$res" --cpu-cap "$rem" "$@" >> "$log" 2>&1 &
  CHILD=$!
  local rc=0; wait "$CHILD" || rc=$?
  while kill -0 "$CHILD" 2>/dev/null; do rc=0; wait "$CHILD" || rc=$?; done  # a trapped signal interrupts wait
  CHILD=
  if [ "$rc" != 0 ] && [ "$STOP" = 0 ]; then echo "$(date) $res: ns_run exit $rc; worker fails"; return 2; fi
  [ "$(status "$res")" = DONE ] && return 0
  return 1                                   # TRUNCATED by the cap or a signal: this worker is done
}
[ -e results/STOP_r13 ] && { echo "STOP_r13 present: nothing to do"; exit 0; }
# W0: step-1 second seed (S12', seed0 19000), then the registered STOP rule
go results/S6x2_N10_guided_N65536_tau0.005_s19000 10 logs/s12p_guided_N65536_s19000.txt \
     --Lx 6 --Ly 2 --Ne 10 --start "$ST6" --seed0 19000 --sampler guided --nsamp 65536
if [ ! -e results/step1_r13.json ]; then
  $PY score_r13_step1.py >> logs/step1_r13.txt 2>&1; rc=$?
  if [ "$rc" = 3 ]; then touch results/STOP_r13; echo "step 1 no longer SUPPORT-LIKE: STOP_r13 written, no S14 arm"; exit 0; fi
  if [ "$rc" != 0 ]; then echo "score_r13_step1.py failed (rc $rc); stopping"; exit 1; fi
fi
grep -q '"verdict": "SUPPORT-LIKE"' results/step1_r13.json || { touch results/STOP_r13; echo "step1_r13: not SUPPORT-LIKE; STOP"; exit 0; }
# W1-W3: S14 ladder (seed0 9000)
go results/S7x2_N12_exact_tau0.005_s9000 140 logs/s14_exact.txt \
     --Lx 7 --Ly 2 --Ne 12 --start "$ST7" --seed0 9000 --exact
go results/S7x2_N12_guided_N1048576_tau0.005_s9000 60 logs/s14_guided_N1048576.txt \
     --Lx 7 --Ly 2 --Ne 12 --start "$ST7" --seed0 9000 --sampler guided --nsamp 1048576
go results/S7x2_N12_guided_N262144_tau0.005_s9000 55 logs/s14_guided_N262144.txt \
     --Lx 7 --Ly 2 --Ne 12 --start "$ST7" --seed0 9000 --sampler guided --nsamp 262144
echo "$(date) all round-13 items DONE (STOP=$STOP)"
