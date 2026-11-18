#!/bin/bash
# P5.1 R51.32 `regmatch`: regime-matched 10 -> 12 size comparison (RULE.md here, round 17). DRAFT, not submitted.
# Submit only through compute_gate.py submit with voi.md.
# Stage 1: 5x2 exact (noise-free) arms at tau in {0.00375, 0.0025, 0.00125}, 120 steps, via xsx_run.py (ns_run.py
#   unchanged + per-step sqrt(x^T S x) log). Selection: score_r17_regime.py select -> tau* or NOT-MATCHED (stop).
# Stage 2: 5x2 guided rungs 4096, 16384, 65536, 262144 at tau*, seed0 9000, 120 steps (ns_run.py).
# Stage 3: seed0 19000 at the rungs that define N_needed(5x2, tau*) (score_r17_regime.py owed; 1 or 2 rungs).
# 6x2 side: R51.8's stored tau 0.005 arms (no 6x2 compute).
# Wall time: interactive pre-flight round 17 (logs/preflight_*.txt): exact 0.069 CPU-min/step, guided 262,144
# 0.052; in-job rates of 763968/769098 (R51.8 5x2, CPU-min per 120 steps): exact 14.4, guided 4k ~4, 16k 6.0,
# 65k 8.4; 262k taken as 0.052 x 1.75 (in-job / interactive ratio of the exact arm) x 120 = 11.
# Stage 1 3 x 14.4 = 43; stage 2 4 + 6 + 8.4 + 11 = 29.4; stage 3 worst (65k + 262k) 19.4; setups ~3 -> 95 min
# x 1.3 = 124 min -> 02:30:00. Peak RSS at 5x2 well under 2 GB.
# Each item runs with --cpu-cap = remaining wall minus 8 min; DONE items are skipped, so resubmitting resumes.
#SBATCH --job-name=hpb-nqs-fermions-P5.1-R51.32-001
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r17_regime/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r17_regime/logs/%x_%j.err
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r17_regime
NS=$DIR/../P5.1_size_scaling_r04/ns_run.py
ST=$DIR/../P5.1_size_scaling_r04/starts/S5x2_N8_U8_tp-0.25_pg11_che_seed0_theta.npy
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR" || exit 1
LIMIT_MIN=${LIMIT_MIN:-150}           # keep equal to --time
T0=$(date +%s)
echo "P5.1 R51.32 regime-matched ladder on $(hostname), job ${SLURM_JOB_ID:-local}, $(date)"
STOP=0; CHILD=
trap 'STOP=1; if [ -n "$CHILD" ]; then kill -USR1 "$CHILD" 2>/dev/null || true; fi' USR1 TERM
remaining() { echo $(( LIMIT_MIN - ($(date +%s) - T0) / 60 - 8 )); }
status() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]+'/run.json')).get('status','NONE'))" "$1" 2>/dev/null || echo NONE; }
# item <result dir> <log name> <command...>: 0 DONE, 1 stop the job (signal / time / error)
item() {
  local res=$1 log=logs/$2.txt; shift 2
  [ "$STOP" = 1 ] && return 1
  [ "$(status "$res")" = DONE ] && return 0
  local rem; rem=$(remaining)
  if [ "$rem" -lt 3 ]; then echo "$(date) $res: $rem min left; job ends"; return 1; fi
  echo "$(date) $res: start/resume with --cpu-cap $rem"
  "$@" --res "$res" --cpu-cap "$rem" >> "$log" 2>&1 &
  CHILD=$!
  local rc=0; wait "$CHILD" || rc=$?
  while kill -0 "$CHILD" 2>/dev/null; do rc=0; wait "$CHILD" || rc=$?; done  # a trapped signal interrupts wait
  CHILD=
  if [ "$rc" != 0 ] && [ "$STOP" = 0 ]; then echo "$(date) $res: exit $rc"; return 1; fi
  [ "$(status "$res")" = DONE ] && return 0
  return 1
}
COMMON="--Lx 5 --Ly 2 --Ne 8 --start $ST --steps 120"
# Stage 1
for TAU in 0.00375 0.0025 0.00125; do
  R=results/S5x2_N8_exact_tau${TAU}_s9000
  # a resumed exact arm would duplicate the log line of an unsaved step; restart such an arm from scratch
  if [ "$(status "$R")" != DONE ] && [ -e "$R" ]; then rm -rf "$R" "results/xsx_exact_tau${TAU}.jsonl"; fi
  item "$R" "exact_tau${TAU}" \
    env XSX_LOG="$DIR/results/xsx_exact_tau${TAU}.jsonl" $PY -u xsx_run.py $COMMON --exact --tau "$TAU" --seed0 9000 || { echo "$(date) stage 1 stopped"; exit 0; }
done
TS=$($PY score_r17_regime.py select)
echo "$(date) selection: $TS"
case "$TS" in 0.*) ;; *) echo "$(date) no matched tau ($TS): job ends, verdict NOT-MATCHED/INCOMPLETE"; exit 0;; esac
# Stage 2
for N in 4096 16384 65536 262144; do
  item "results/S5x2_N8_guided_N${N}_tau${TS}_s9000" "guided_N${N}_tau${TS}_s9000" \
    $PY -u "$NS" $COMMON --nsamp "$N" --sampler guided --tau "$TS" --seed0 9000 || { echo "$(date) stage 2 stopped"; exit 0; }
done
# Stage 3
OWED=$($PY score_r17_regime.py owed)
echo "$(date) second seed owed at: $OWED"
case "$OWED" in INCOMPLETE|NOT-MATCHED) exit 0;; esac
for N in $OWED; do
  item "results/S5x2_N8_guided_N${N}_tau${TS}_s19000" "guided_N${N}_tau${TS}_s19000" \
    $PY -u "$NS" $COMMON --nsamp "$N" --sampler guided --tau "$TS" --seed0 19000 || { echo "$(date) stage 3 stopped"; exit 0; }
done
echo "$(date) all stages done; score with: python3 score_r17_regime.py score (orchestrator)"
