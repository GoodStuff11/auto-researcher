#!/bin/bash
# P5.1 R51.5 `mom`, premise arm (T1; rule pre-registered on card R51.5, 2026-11-17, before any data): fixed-step
# guided SR, open 4x3 `che`, R51.1's start, N = 4096 per step, tau 0.005, seed0 9000 (step n draws with 9000 + n),
# 240 steps = 0.98M draws.  Scored by score_premise.py.  Prepared, NOT submitted by the worker; submit only through
# compute_gate.py submit with voi.md.  The T2 median-of-means arm is released only by a SURVIVES verdict.
# Wall time: login-node preflight (logs/preflight_N4096.txt) 0.102 CPU-min/step at 4096 against 0.142 at 16,384
# (logs/preflight_N16384_repro.txt, which reproduces R51.1 arm A's step 1 exactly); R51.1's in-job 16k rate is 0.36,
# 2.54x the login rate, so in-job 4096 ~ 0.102 x 2.54 = 0.26 CPU-min/step (the step is not linear in N: exact
# scoring each step is a fixed cost).  240 x 0.26 = 62 min, x 1.3 = 81 min -> --cpu-cap 81; + 10 min signal lead and
# start-up -> --time 01:35:00.  Memory: 16k arms ran in 4G; 4096 needs less -> 4G.
# Resumable: ns_run.py saves state every step; on USR1 (10 min before the limit, forwarded to python by the trap) it
# finishes the current step, saves and exits; rerunning the same command resumes with the same seeds.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:35:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r51_5/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r51_5/logs/%x_%j.err
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r51_5
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
echo "P5.1 R51.5 premise arm on $(hostname), job ${SLURM_JOB_ID:-local}"
pid=""
trap '[ -n "$pid" ] && kill -USR1 "$pid" 2>/dev/null' USR1
$PY -u ../P5.1_size_scaling_r04/ns_run.py --Lx 4 --Ly 3 --Ne 10 --sampler guided --nsamp 4096 --tau 0.005 \
    --steps 240 --seed0 9000 --cpu-cap 81 --res results/guided_N4096_tau0.005_s9000 >> logs/guided_N4096.txt 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do wait "$pid"; done
$PY score_premise.py >> logs/guided_N4096.txt 2>&1
echo "done"
