#!/bin/bash
# P5.1 R51.8 `nscale` unit 1, part a (RULE.md): every S10 arm (5x2, N_e 8), run in sequence in one task.
# Prepared round 4, NOT submitted by the worker; submit only through compute_gate.py submit with voi_unit1.md.
# Arms: exact; guided N = 1024, 4096, 16384, 65536, 262144, 1048576; born N = 1024, 4096, 16384, 65536, 262144;
# tau 0.005, 120 steps, seed0 9000.  The start (starts/S5x2_..._theta.npy) was trained interactively (9.9 CPU-min).
# Wall time from 3-step interactive pre-flights (logs/preflight_S5x2_*.txt, CPU-min/step): exact 0.075, guided 1M
# 0.066, guided 16k 0.032, born 1k 0.020; other rungs interpolated (256k 0.055, 64k 0.042, 4k 0.025, 1k 0.022;
# born 256k 0.045, 64k 0.035, 16k 0.028, 4k 0.022).  Sum over the 12 arms 0.50 CPU-min per step x 120 = 60 min,
# + 12 x 0.2 min setup = 62 min interactive; x 1.8 (in-job / interactive, jobs 759974 and 761860) = 112 min;
# x 1.3 = 146 min -> --time 02:30:00.  Memory 1.2 GB measured -> 4G.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:30:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
ST=starts/S5x2_N8_U8_tp-0.25_pg11_che_seed0_theta.npy
echo "P5.1 R51.8 nscale unit 1a (S10) on $(hostname), job ${SLURM_JOB_ID:-local}"
run() { $PY -u ns_run.py --Lx 5 --Ly 2 --Ne 8 --start "$ST" --steps 120 --seed0 9000 --cpu-cap 30 "$@"; }
run --exact | tee -a logs/s10_exact.txt
for N in 1024 4096 16384 65536 262144 1048576; do run --sampler guided --nsamp $N | tee -a logs/s10_guided_N$N.txt; done
for N in 1024 4096 16384 65536 262144; do run --sampler born --nsamp $N | tee -a logs/s10_born_N$N.txt; done
echo "unit 1a done"
