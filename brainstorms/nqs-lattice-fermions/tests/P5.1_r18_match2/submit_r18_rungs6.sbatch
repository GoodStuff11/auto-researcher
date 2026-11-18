#!/bin/bash
# P5.1 R51.33 `match2`, job 2 of 2: the two 6x2 guided rungs at tau 0.0025 (RULE.md here, round 18). DRAFT, not
# submitted; released only after job 1 (submit_r18_match2.sbatch) wrote results/rungs6.json, through
# compute_gate.py submit with voi_rungs6.md (its p is set from job 1's N5(m6) and 6x2 reference).
# Rungs: results/rungs6.json (RULE.md formula: N_P = 4096 floor(1.728 e^-0.14 N5*/4096), N_S = 4096 ceil(3.52 e^0.13
#   N5*/4096) capped at 2,097,152, N5* = 5x2 need interpolated at the 6x2 arm's m), seed0 9000, 120 steps.
# Wall time (CPU-min): 6x2 guided interactive round 18: 147,456 0.315/step, 393,216 0.469, 1,048,576 0.665;
#   in-job / interactive 1.54 (6x2 262,144: 0.653 / 0.425). Predicted rungs (N5* ~ 225k): ~335k and ~900k ->
#   0.68 and 0.99/step in-job -> 82 + 119 = 201 min. Worst case (N_S at the 2,097,152 cap, ~0.9/step interactive,
#   N_P ~ 800k): 115 + 166 = 281 min; x 1.25 -> 06:00:00.
#SBATCH --job-name=hpb-nqs-fermions-P5.1-R51.33-002
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r18_match2/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r18_match2/logs/%x_%j.err
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r18_match2
NS=$DIR/../P5.1_size_scaling_r04/ns_run.py
ST6=$DIR/../P5.1_size_scaling_r04/starts/S6x2_N10_U8_tp-0.25_pg11_che_seed0_theta.npy
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR" || exit 1
LIMIT_MIN=${LIMIT_MIN:-360}           # keep equal to --time
T0=$(date +%s)
echo "P5.1 R51.33 job 2 (6x2 rungs) on $(hostname), job ${SLURM_JOB_ID:-local}, $(date)"
RUNGS=$($PY -c "import json; print(' '.join(str(n) for n in json.load(open('results/rungs6.json'))['rungs']))") \
  || { echo "results/rungs6.json missing: job 1 not finished"; exit 0; }
echo "$(date) rungs: $RUNGS"
STOP=0; CHILD=
trap 'STOP=1; if [ -n "$CHILD" ]; then kill -USR1 "$CHILD" 2>/dev/null || true; fi' USR1 TERM
remaining() { echo $(( LIMIT_MIN - ($(date +%s) - T0) / 60 - 8 )); }
status() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]+'/run.json')).get('status','NONE'))" "$1" 2>/dev/null || echo NONE; }
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
  while kill -0 "$CHILD" 2>/dev/null; do rc=0; wait "$CHILD" || rc=$?; done
  CHILD=
  if [ "$rc" != 0 ] && [ "$STOP" = 0 ]; then echo "$(date) $res: exit $rc"; return 1; fi
  [ "$(status "$res")" = DONE ] && return 0
  return 1
}
for N in $RUNGS; do
  item "results/S6x2_N10_guided_N${N}_tau0.0025_s9000" "guided6_N${N}_tau0.0025_s9000" \
    $PY -u "$NS" --Lx 6 --Ly 2 --Ne 10 --start "$ST6" --steps 120 --tau 0.0025 --seed0 9000 --nsamp "$N" \
    --sampler guided || { echo "$(date) stopped"; exit 0; }
done
echo "$(date) done; score with: python3 score_r18_match2.py score; owed seeds: score_r18_match2.py owed"
