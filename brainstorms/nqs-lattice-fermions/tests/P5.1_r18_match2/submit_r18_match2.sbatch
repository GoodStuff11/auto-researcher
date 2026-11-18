#!/bin/bash
# P5.1 R51.33 `match2`, job 1 of 2: references and 5x2 matching curve (RULE.md here, round 18). DRAFT, not submitted.
# Submit only through compute_gate.py submit with voi.md. Job 2 (submit_r18_rungs6.sbatch) runs the two 6x2 guided
# rungs this job places; job 3 (second 6x2 seed, RULE.md clause) is gated after scoring.
# Stage A0: 5x2 exact arm at tau 0.000625 (xsx_run.py, 120 steps) + guided rungs 4k-262k seed 9000 (+1,048,576 iff
#   s(262,144) > 0.10). The tau 0.00375 / 0.00125 seed-9000 rungs exist (round-18 interactive pilots, disclosed in
#   RULE.md; same code and seeds, so item() skips them as DONE).
# Stage A: 6x2 exact (noise-free) arm at tau 0.0025 via ../P5.1_r17_regime/xsx_run.py. Release: score_r18_match2.py
#   release -> OK or STOP (job ends, INCOMPLETE).
# Stage A2: score_r18_match2.py bracket -> the two 5x2 grid points bracketing the 6x2 arm's m; seed0 19000 at their
#   defining rungs (or NOT-MATCHED: job ends).
# Stage A3: score_r18_match2.py place -> results/rungs6.json (the two 6x2 rungs for job 2).
# Wall time (CPU-min, in-job): 5x2 exact 14.2 (R51.32 job); 5x2 guided 4k 4.2, 16k 5.8, 65k 8.2, 262k 10.7 (R51.32
#   job), 1M 13.4 (R51.8 interactive pre-flight 0.064/step x 1.75 in-job factor x 120); stage A0 <= 57 (14.2 +
#   28.9 + one 1M; the pilots' s(262k) are 0.090 at 0.00125 and 0.047 at 0.00375, so no 1M there). 6x2 exact
#   2.90/step in-job (R51.8 job, 347.7 per 120 steps; round-18 probe at tau 0.0025 1.67/step interactive = the
#   tau 0.005 probe's 1.69) -> 348. Stage A2 worst 2 x (262k + 1M) = 48. Setups ~4. Total <= 457 min = 7.6 h;
#   x 1.25 -> 09:30:00.
# Each item runs with --cpu-cap = remaining wall minus 8 min; DONE items are skipped, so resubmitting resumes.
#SBATCH --job-name=hpb-nqs-fermions-P5.1-R51.33-001
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=09:30:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r18_match2/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r18_match2/logs/%x_%j.err
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r18_match2
NS=$DIR/../P5.1_size_scaling_r04/ns_run.py
XSX=$DIR/../P5.1_r17_regime/xsx_run.py
ST5=$DIR/../P5.1_size_scaling_r04/starts/S5x2_N8_U8_tp-0.25_pg11_che_seed0_theta.npy
ST6=$DIR/../P5.1_size_scaling_r04/starts/S6x2_N10_U8_tp-0.25_pg11_che_seed0_theta.npy
PY=~/.conda/envs/lit_env/bin/python
SC="$PY score_r18_match2.py"
cd "$DIR" || exit 1
LIMIT_MIN=${LIMIT_MIN:-570}           # keep equal to --time
T0=$(date +%s)
echo "P5.1 R51.33 job 1 (references, 5x2 curve) on $(hostname), job ${SLURM_JOB_ID:-local}, $(date)"
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
# trimx <result dir> <xsx log>: on resume keep only the log lines of the steps state.npz holds
trimx() {
  [ "$(status "$1")" = DONE ] && return 0
  $PY -c "
import json, os, sys
import numpy as np
r, x = sys.argv[1] + '/state.npz', sys.argv[2]
n = len(json.loads(str(np.load(r)['hist']))) if os.path.exists(r) else 0
if os.path.exists(x):
    L = open(x).read().splitlines()
    open(x, 'w').write(''.join(l + '\n' for l in L[:n]))
print('xsx log kept', n, 'lines')" "$1" "$2"
}
C5="--Lx 5 --Ly 2 --Ne 8 --start $ST5 --steps 120"
C6="--Lx 6 --Ly 2 --Ne 10 --start $ST6 --steps 120"
# Stage A0
T5=0.000625
R=results/S5x2_N8_exact_tau${T5}_s9000; XL=$DIR/results/xsx_exact_tau${T5}.jsonl
trimx "$R" "$XL"
item "$R" "exact5_tau${T5}" env XSX_LOG="$XL" $PY -u "$XSX" $C5 --exact --tau $T5 --seed0 9000 || { echo "$(date) stage A0 stopped"; exit 0; }
for TAU in 0.000625 0.00125 0.00375; do
  for N in 4096 16384 65536 262144; do
    item "results/S5x2_N8_guided_N${N}_tau${TAU}_s9000" "guided5_N${N}_tau${TAU}_s9000" \
      $PY -u "$NS" $C5 --nsamp "$N" --sampler guided --tau "$TAU" --seed0 9000 || { echo "$(date) stage A0 stopped"; exit 0; }
  done
  S262=$($SC s262 "$TAU")
  echo "$(date) 5x2 tau $TAU s(262144) = $S262"
  if $PY -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0.10 else 1)" "$S262" 2>/dev/null; then
    item "results/S5x2_N8_guided_N1048576_tau${TAU}_s9000" "guided5_N1048576_tau${TAU}_s9000" \
      $PY -u "$NS" $C5 --nsamp 1048576 --sampler guided --tau "$TAU" --seed0 9000 || { echo "$(date) stage A0 stopped"; exit 0; }
  fi
done
# Stage A
R=results/S6x2_N10_exact_tau0.0025_s9000; XL=$DIR/results/xsx_exact6_tau0.0025.jsonl
trimx "$R" "$XL"
item "$R" "exact6_tau0.0025" env XSX_LOG="$XL" $PY -u "$XSX" $C6 --exact --tau 0.0025 --seed0 9000 || { echo "$(date) stage A stopped"; exit 0; }
REL=$($SC release); echo "$(date) release: $REL"
case "$REL" in OK) ;; *) echo "$(date) 6x2 reference not usable: job ends (INCOMPLETE)"; exit 0;; esac
# Stage A2
OWED5=$($SC bracket); echo "$(date) 5x2 seed 19000 owed at: $OWED5"
case "$OWED5" in NOT-MATCHED*|STOP*) echo "$(date) job ends ($OWED5)"; exit 0;; esac
for P in $OWED5; do
  [ "$P" = NONE ] && continue
  TAU=${P%%:*}; N=${P##*:}
  item "results/S5x2_N8_guided_N${N}_tau${TAU}_s19000" "guided5_N${N}_tau${TAU}_s19000" \
    $PY -u "$NS" $C5 --nsamp "$N" --sampler guided --tau "$TAU" --seed0 19000 || { echo "$(date) stage A2 stopped"; exit 0; }
done
# Stage A3
PL=$($SC place); echo "$(date) 6x2 rungs placed: $PL (results/rungs6.json); job 2 = submit_r18_rungs6.sbatch"
