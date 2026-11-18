#!/bin/bash
# P5.1 R51.16 `cmp` (RULE_cmp.md): plain SR / MinSR (Born draws) against guided fixed-step SR at equal total samples,
# 4x3 only.  Prepared round 5, NOT submitted by the worker; submit only through compute_gate.py submit with voi_cmp.md.
# Task 0: born N 16,384, tau 0.005, 824 steps.  Task 1: born 65,536 (tau 0.005 and 0.01, 206 steps), born 262,144
# (51 steps), guided 16,384 continued from R51.1 arm A's step-120 theta (292 steps).  Task 2 (R51.11 T2,
# RULE_qopt_T2.md): q ~ |psi| (ns_q.py --qmix 1.0) N 16,384 (412 steps) and N 65,536 (103 steps).
# Task 2 wall: pre-flights 0.20 and 0.39 CPU-min/step (logs/preflight_q1_*.txt): 82 + 40 = 122 min x 2.34 = 286 min.
# Wall time: interactive pre-flight born 16k 0.135 CPU-min/step (logs/preflight_born16k.txt); R51.8 pre-flight born
# 262k 0.28; born 65k 0.18 and guided 16k 0.15 interpolated by unique draws.  Task 0: 824 x 0.135 = 111 min;
# task 1: 2 x 206 x 0.18 + 51 x 0.28 + 292 x 0.15 = 132 min.  x 1.8 (in-job / interactive, jobs 759974, 761860)
# x 1.3 margin: task 0 260 min, task 1 309 min -> --time 05:20:00 per task.  Memory 2 GB measured (4x3) -> 4G.
# Resumable: ns_run.py saves state every step; USR1 10 min before the limit stops cleanly; rerun resumes.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=05:20:00
#SBATCH --array=0-2
#SBATCH --signal=USR1@600
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r05_gen/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r05_gen/logs/%x_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r05_gen
PY=~/.conda/envs/lit_env/bin/python
NS=../P5.1_size_scaling_r04/ns_run.py
cd "$DIR"
ST12=../P4.1_cheap_su2_scaling/logs/pilot/pg_che/S4x3_N10_U8_tp-0.25_pg11_che_seed0_theta.npy
STA=../P5.1_noise_floor_r01/results/N16384_tau0.005_s9000/theta.npy
run() { local log=$1; shift; $PY -u $NS --Lx 4 --Ly 3 --Ne 10 --cpu-cap 290 "$@" >> "$log" 2>&1; }
echo "P5.1 R51.16 cmp task ${SLURM_ARRAY_TASK_ID:-0} on $(hostname), job ${SLURM_JOB_ID:-local}"
if [ "${SLURM_ARRAY_TASK_ID:-0}" = "2" ]; then
  $PY -u ns_q.py --qmix 1.0 --Lx 4 --Ly 3 --Ne 10 --cpu-cap 290 --start $ST12 --sampler guided --nsamp 16384 --tau 0.005 --steps 412 --seed0 9000 --res results/q1_N16384_tau0.005 >> logs/q1_16k.txt 2>&1
  $PY -u ns_q.py --qmix 1.0 --Lx 4 --Ly 3 --Ne 10 --cpu-cap 290 --start $ST12 --sampler guided --nsamp 65536 --tau 0.005 --steps 103 --seed0 9000 --res results/q1_N65536_tau0.005 >> logs/q1_65k.txt 2>&1
elif [ "${SLURM_ARRAY_TASK_ID:-0}" = "0" ]; then
  run logs/born16k.txt --start $ST12 --sampler born --nsamp 16384 --tau 0.005 --steps 824 --seed0 9000 --res results/born_N16384_tau0.005
else
  run logs/born65k.txt --start $ST12 --sampler born --nsamp 65536 --tau 0.005 --steps 206 --seed0 9000 --res results/born_N65536_tau0.005
  run logs/born65k_t01.txt --start $ST12 --sampler born --nsamp 65536 --tau 0.01 --steps 206 --seed0 9000 --res results/born_N65536_tau0.01
  run logs/born262k.txt --start $ST12 --sampler born --nsamp 262144 --tau 0.005 --steps 51 --seed0 9000 --res results/born_N262144_tau0.005
  run logs/guided16k_cont.txt --start $STA --sampler guided --nsamp 16384 --tau 0.005 --steps 292 --seed0 9120 --res results/guided_N16384_tau0.005_cont
fi
echo "task done"
