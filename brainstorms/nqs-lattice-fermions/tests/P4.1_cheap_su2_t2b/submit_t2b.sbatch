#!/bin/bash
# P4.1 R2 `pcg-scal` T2b (leaf section "R2 T2b design (2026-11-16)"): does the 12-site gain over che survive at 24 sites?
# Task i runs line i+1 of tasks_t2b.txt: "<Lx> <arm> <seed> <iters> <exact|-> <n_eval> <max_hours> <ckpts> <futility|->".
#   0-2: open 8x3, N_e = 20, (+,+): pcg (R2), che (H = 16), r1 (foil); 3000 SR steps, final energy from 262,144 samples.
#   3-5: open 4x3, N_e = 10, (+,+): pcg, che, r1; same 3000-step protocol, exact final and checkpoint energies.
# Protocol = job 686990 (1024 chains, lr 0.05 constant then cosine to 10%, Polyak average of the last 750 steps, P_G (1,1)).
# pcg uses the compiled Pfaffian (pf_fast.py). Futility (task 0 only): rho_ck - 2 sigma > 8 at step 750 or > 5 at step
# 1500, against the W-24 che (698224 task 0) and the 687302 (+,+) reference; disarmed while either is missing.
# Written budget exception (Stage 3.5.1): about 40 CPU-hours for task 0, 16 for task 1, 8 for task 2, 3-5 for tasks 3-5
# (sum about 76 CPU-hours expected, hard caps sum about 120). A capped task writes capped = true and resumes on resubmit.
# AMBIGUOUS verdict: sbatch --export=ALL,SEED=1 --array=0-2 submit_t2b.sbatch
# Afterwards (login node): python3 t2b_report.py --mode report
#SBATCH --job-name=P41t2b
#SBATCH --partition=kim
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_t2b/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_t2b/logs/%x_%A_%a.err
set -euo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_t2b
cd "$DIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
read LX ARM SEED0 ITERS EXACT NEVAL MAXH CKPTS FUT <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" tasks_t2b.txt)"
SEED=${SEED:-$SEED0}
CMP=$DIR/../P4.1_cheap_su2_t2/results/w24/S8x3_N20_U8_tp-0.25_pg11_che_H16_seed0.json
ARGS="--mode train --arm $ARM --Lx $LX --Ly 3 --pg=1,1 --seed $SEED --iters $ITERS --n-chains 1024 --threads 4"
ARGS="$ARGS --max-hours $MAXH --ckpts $CKPTS --ckpt-eval 32768 --outdir $DIR/results/t2b"
if [ "$EXACT" = "exact" ]; then ARGS="$ARGS --exact-eval"; else ARGS="$ARGS --n-eval $NEVAL"; fi
if [ "$FUT" != "-" ]; then ARGS="$ARGS --futility $FUT --fut-che $CMP"; fi
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): t2b.py $ARGS"
~/.conda/envs/lit_env/bin/python -u t2b.py $ARGS
