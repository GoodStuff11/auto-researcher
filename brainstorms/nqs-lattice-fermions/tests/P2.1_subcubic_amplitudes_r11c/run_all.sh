#!/bin/bash
# P2.1 Round 11c scored cost runs: four single-threaded arms in parallel, CPU-capped per process.
# rmax per route arm from scan_output.txt (smallest t_sw at L=16): full 32, half 16, quarter 16.
# Usage: bash run_all.sh   (from this directory). Writes results/<arm>.json and run_<arm>.log.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/jek354_numba_r11c
PY=~/.conda/envs/lit_env/bin/python
for spec in "dense 0" "full 32" "half 16" "quarter 16"; do
  set -- $spec
  ( ulimit -t 260; /usr/bin/time -f "CPU user=%U sys=%S wall=%e" $PY delayed.py --mode run --arm $1 --rmax $2 \
      --sizes 8,12,16,24 --budgets 10,20,90,40 --tsw-only 24 ) > run_$1.log 2>&1 &
done
wait
