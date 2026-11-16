#!/bin/bash
# Pre-flight dry run (Stage 3.5.4) of the T2b equal-wall-time amendment on the 4x2 toy (N_e = 6, (+,+)), local,
# 4 threads, a few CPU-minutes. Output: logs/wall_toy_output.txt, then t2b_wall_report.py --mode preflight.
#   identity control: t2b.py and t2b_wall.py with the same arguments give identical energies (exact and sampled),
#                     so the wrapper changes only the file tag;
#   toy che_wall: 40 steps (toy ratio 2 against the 20-step T2b toy runs in results_preflight/), exact and sampled;
#   resume control: rerunning a finished wall task is skipped.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
PY=~/.conda/envs/lit_env/bin/python
O=$PWD/results_preflight_wall
B="--mode train --arm che --Lx 4 --Ly 2 --Ne 6 --pg=1,1 --n-chains 128 --therm-sweeps 10 --threads 4 --ckpt-every 5"
S24="--force-sampled --ckpt-eval 2048 --n-eval 4096"
echo "== identity control: plain t2b.py vs t2b_wall.py, same args (20 steps)"
$PY t2b.py $B --iters 20 --ckpts 5,10 --exact-eval --outdir $O/plain_s12
$PY t2b_wall.py $B --iters 20 --ckpts 5,10 --exact-eval --outdir $O/id_s12
$PY t2b.py $B --iters 20 --ckpts 5,10 $S24 --outdir $O/plain_s24
$PY t2b_wall.py $B --iters 20 --ckpts 5,10 $S24 --outdir $O/id_s24
echo "== toy che_wall (40 steps = 20 x toy ratio 2, checkpoints at 25 and 50%)"
$PY t2b_wall.py $B --iters 40 --ckpts 10,20 --exact-eval --outdir $O/s12
$PY t2b_wall.py $B --iters 40 --ckpts 10,20 $S24 --outdir $O/s24
echo "== resume control: rerunning a finished wall task is skipped"
$PY t2b_wall.py $B --iters 40 --ckpts 10,20 $S24 --outdir $O/s24
