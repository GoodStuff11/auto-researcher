#!/bin/bash
# Round 12 T1 amendment A2: rerun full and quarter on toy 23 (SR cap 400 CPU-s each), single-threaded.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
( ulimit -t 600; python3 t1_sub.py --mode train --arm full --instance 23 --max-cpu 400 > logs/t1_full_inst23.log 2>&1 ) &
( ulimit -t 600; python3 t1_sub.py --mode train --arm quarter --instance 23 --max-cpu 400 > logs/t1_quarter_inst23.log 2>&1 ) &
wait
