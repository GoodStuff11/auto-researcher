#!/bin/bash
# P2.1 Round 12c scored cost runs: two waves of three single-threaded arms, CPU-capped per process.
# Wave A: dense2, half_F4, half_F16; wave B: dense2, half_F8, half_F12. Each half_F is scored against
# the dense2 of its own wave. Usage: bash run_all.sh (from this directory). Writes results/<wave>_<arm>.json
# and logs/run_<wave>_<arm>.log. Then: python larger_f.py --mode aggregate > aggregate_output.txt
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/jek354_numba_r12c
PY=~/.conda/envs/lit_env/bin/python
mkdir -p logs results
for wave in "A dense2 half_F4 half_F16" "B dense2 half_F8 half_F12"; do
  set -- $wave
  w=$1; shift
  for arm in "$@"; do
    ( ulimit -t 150; /usr/bin/time -f "CPU user=%U sys=%S wall=%e" $PY larger_f.py --mode run --wave $w --arm $arm \
        --sizes 12,16 --budgets 20,90 ) > logs/run_${w}_${arm}.log 2>&1 &
  done
  wait
done
