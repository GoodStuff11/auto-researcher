#!/bin/bash
# P5.1 R51.16 `cmp` (RULE_cmp.md, revision 1): plain SR (Born draws; MinSR gives the same step at N >= P = 2637) against
# guided fixed-step SR at equal total samples, 4x3 only.  Prepared round 5, NOT submitted by the worker; submit only
# through compute_gate.py submit with voi_cmp.md.  R51.11's T2 is a separate job (submit_qopt.sbatch).
# Task 0 (first, since it supplies T_guided for both R51.16 and R51.11): guided N 16,384 continued from R51.1 arm A's
#   step-120 theta, 332 steps (A + continuation = 452 steps = 7.41M draws), then born N 65,536, tau 0.010, 226 steps.
# Task 1: born N 16,384, tau 0.005, 904 steps (14.8M draws = 2 x guided B's sustained T).
# Wall time from R51.1's in-job 4x3 rates (CPU-min/step, wall ~ CPU): 16k 0.36, 65k 0.27 (Born draws have fewer
# unique configurations than guided at equal N, so these are upper bounds).  Task 0: 332 x 0.36 + 226 x 0.27 = 181
# min; task 1: 904 x 0.36 = 325 min; x 1.3 margin -> 235 and 423 min.  Each run's --cpu-cap is its own estimate x 1.3
# (task 0: 156 + 79 = 235 min; task 1: 423 min), so a task's sequential total is bounded by the sum, inside --time.
# Memory 2 GB measured (4x3) -> 4G.  Resumable: ns_run.py saves state every step; on USR1 (sent to the batch shell
# 10 min before the limit, forwarded to the running python) it stops after the current step; later runs are skipped.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=07:10:00
#SBATCH --array=0-1
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r05_gen/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r05_gen/logs/%x_%A_%a.err
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r05_gen
PY=~/.conda/envs/lit_env/bin/python
NS=../P5.1_size_scaling_r04/ns_run.py
cd "$DIR"
ST12=../P4.1_cheap_su2_scaling/logs/pilot/pg_che/S4x3_N10_U8_tp-0.25_pg11_che_seed0_theta.npy
STA=../P5.1_noise_floor_r01/results/N16384_tau0.005_s9000/theta.npy
pid=""; stop=0
trap 'stop=1; [ -n "$pid" ] && kill -USR1 "$pid" 2>/dev/null' USR1
run() {  # run LOG CAP args...: python in the background so the trap can forward USR1; wait until it has saved
  local log=$1 cap=$2; shift 2
  [ "$stop" = 1 ] && { echo "skipped (stop signal): $*"; return; }
  $PY -u $NS --Lx 4 --Ly 3 --Ne 10 --cpu-cap "$cap" "$@" >> "$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do wait "$pid"; done
  pid=""
}
echo "P5.1 R51.16 cmp task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname), job ${SLURM_JOB_ID:-local}"
if [ "${SLURM_ARRAY_TASK_ID:-0}" = "0" ]; then
  run logs/guided16k_cont.txt 156 --start $STA --sampler guided --nsamp 16384 --tau 0.005 --steps 332 --seed0 9120 --res results/guided_N16384_tau0.005_cont
  run logs/born65k_t01.txt 79 --start $ST12 --sampler born --nsamp 65536 --tau 0.01 --steps 226 --seed0 9000 --res results/born_N65536_tau0.01
else
  run logs/born16k.txt 423 --start $ST12 --sampler born --nsamp 16384 --tau 0.005 --steps 904 --seed0 9000 --res results/born_N16384_tau0.005
fi
echo "task done (stop=$stop)"
