#!/bin/bash
# P5.1 R51.7 `rmavg` unshelve test at 10 and 12 sites (RULE_r14_decay.md, rule round 14; driver ns_decay.py round 15).
# Submit only through compute_gate.py submit with voi_r14_decay.md.
# 18 continuations of 60 steps (CONT fixed 0.005; DEC decaying + Polyak-Ruppert average) from the step-120 states of
# R51.8's arms: S10 exact, S10 guided 4k s9000, 16k and 65k s9000/s19000; S12' guided 16k, 65k, 262k s9000.
# Driver identity test (round 15, leaves/P5.1/log/round-15.md): ns_decay.py --mode cont reproduces ns_run.py resumed
# with --steps 123 bit for bit (|dE| = 0) at S10 exact and S10 guided 16k s19000; USR1 + resume reproduces an
# uninterrupted DEC run bit for bit.
# Wall time from in-job rates of 763968/769098/769376 (CPU-min per 120 steps, halved for 60): S10 exact 7.2,
# guided 4k ~2, 16k 3.0, 65k 4.2; S12' guided 16k 16.0, 65k 26.6, 262k 39.2; DEC adds one exact energy per step
# (S12' 0.06 CPU-min). Sum: S10 2x(7.2+2+2x3+2x4.2) = 47 min; S12' 2x81.8 + 11 = 175 min; setups ~15 min
# -> 237 min x 1.3 = 308 min -> 06:00:00. (Interactive round-15 rates were lower: S12' 16k DEC 0.18 CPU-min/step
# including the extra energy; the in-job rates are kept.) Peak RSS at S12' 1.7 GB.
# Each item runs with --cpu-cap = remaining wall minus 12 min; DONE items are skipped, so resubmitting resumes.
# Signals: Slurm sends USR1 to this shell (B:) 10 min before the limit; the trap forwards it (and SIGTERM) to
# ns_decay.py, which finishes the current step, saves TRUNCATED and exits 0; no later item starts.
#SBATCH --job-name=hpb-nqs-fermions-P5.1-R51.7-001
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.err
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR" || exit 1
LIMIT_MIN=${LIMIT_MIN:-360}           # keep equal to --time (6 h)
T0=$(date +%s)
echo "P5.1 R51.7 decay test on $(hostname), job ${SLURM_JOB_ID:-local}, $(date)"
STOP=0; CHILD=
trap 'STOP=1; if [ -n "$CHILD" ]; then kill -USR1 "$CHILD" 2>/dev/null || true; fi' USR1 TERM
remaining() { echo $(( LIMIT_MIN - ($(date +%s) - T0) / 60 - 12 )); }
status() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]+'/run.json')).get('status','NONE'))" "$1" 2>/dev/null || echo NONE; }
FAIL=0
# item <R51.8 arm> <mode>: 0 DONE, 1 stop the job (signal / time), 2 error (logged; the job goes on)
item() {
  local src=results/$1 mode=$2 res=results/r14_decay/$1_$2 log=logs/r14_decay_$1_$2.txt
  [ "$STOP" = 1 ] && return 1
  [ "$(status "$res")" = DONE ] && return 0
  local rem; rem=$(remaining)
  if [ "$rem" -lt 5 ]; then echo "$(date) $res: $rem min left; job ends"; return 1; fi
  echo "$(date) $res: start/resume with --cpu-cap $rem"
  $PY -u ns_decay.py --src "$src" --mode "$mode" --extra 60 --tau0 0.005 --res "$res" --cpu-cap "$rem" >> "$log" 2>&1 &
  CHILD=$!
  local rc=0; wait "$CHILD" || rc=$?
  while kill -0 "$CHILD" 2>/dev/null; do rc=0; wait "$CHILD" || rc=$?; done  # a trapped signal interrupts wait
  CHILD=
  if [ "$rc" != 0 ] && [ "$STOP" = 0 ]; then echo "$(date) $res: ns_decay exit $rc"; FAIL=1; return 2; fi
  [ "$(status "$res")" = DONE ] && return 0
  return 1
}
ARMS="S5x2_N8_exact_tau0.005_s9000 S5x2_N8_guided_N4096_tau0.005_s9000
S5x2_N8_guided_N16384_tau0.005_s9000 S5x2_N8_guided_N16384_tau0.005_s19000
S5x2_N8_guided_N65536_tau0.005_s9000 S5x2_N8_guided_N65536_tau0.005_s19000
S6x2_N10_guided_N16384_tau0.005_s9000 S6x2_N10_guided_N65536_tau0.005_s9000 S6x2_N10_guided_N262144_tau0.005_s9000"
for A in $ARMS; do
  for M in cont dec; do
    item "$A" "$M"; r=$?
    if [ "$r" = 1 ]; then echo "$(date) stopping (STOP=$STOP)"; exit 0; fi
  done
done
echo "$(date) all 18 items processed (FAIL=$FAIL, STOP=$STOP)"
exit $FAIL
