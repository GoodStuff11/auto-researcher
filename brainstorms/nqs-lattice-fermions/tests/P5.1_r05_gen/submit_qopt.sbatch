#!/bin/bash
# P5.1 R51.11 `qopt` T2 (RULE_qopt_T2.md, revision 1): SR with draws from q ~ |psi| at 4x3 against guided SR in
# total samples.  Prepared round 5, NOT submitted by the worker; submit only through compute_gate.py submit with
# voi_qopt.md.  Scoring needs T_guided from submit_cmp.sbatch task 0 (guided 16k continuation) and R51.1 arm B.
# Arms (ns_q.py --qmix 1.0 = ns_run.py's step with q ~ |psi|): N 16,384, 362 steps, then N 65,536, 91 steps
# (5.93M draws each = 0.8 x guided B's sustained T, the NO-GAIN boundary; nothing beyond it can change the verdict).
# Wall time: no in-job q ~ |psi| rate exists; interactive pre-flights 0.20 (16k) and 0.39 (65k) CPU-min/step
# (logs/preflight_q1_*.txt) x 2.4 (R51.1 arm A in-job 0.36 over guided 16k interactive 0.15) = 0.48 and 0.94.
# 362 x 0.48 + 91 x 0.94 = 260 min x 1.3 = 338 min -> --time 05:45:00; per-run --cpu-cap 226 + 112 = 338 min.
# Memory 2 GB measured -> 4G.  Signal handling as submit_cmp.sbatch (USR1 forwarded to the running python).
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=05:45:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r05_gen/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r05_gen/logs/%x_%j.err
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r05_gen
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
ST12=../P4.1_cheap_su2_scaling/logs/pilot/pg_che/S4x3_N10_U8_tp-0.25_pg11_che_seed0_theta.npy
pid=""; stop=0
trap 'stop=1; [ -n "$pid" ] && kill -USR1 "$pid" 2>/dev/null' USR1
run() {
  local log=$1 cap=$2; shift 2
  [ "$stop" = 1 ] && { echo "skipped (stop signal): $*"; return; }
  $PY -u ns_q.py --qmix 1.0 --Lx 4 --Ly 3 --Ne 10 --cpu-cap "$cap" "$@" >> "$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do wait "$pid"; done
  pid=""
}
echo "P5.1 R51.11 qopt T2 on $(hostname), job ${SLURM_JOB_ID:-local}"
run logs/q1_16k.txt 226 --start $ST12 --sampler guided --nsamp 16384 --tau 0.005 --steps 362 --seed0 9000 --res results/q1_N16384_tau0.005
run logs/q1_65k.txt 112 --start $ST12 --sampler guided --nsamp 65536 --tau 0.005 --steps 91 --seed0 9000 --res results/q1_N65536_tau0.005
echo "done (stop=$stop)"
