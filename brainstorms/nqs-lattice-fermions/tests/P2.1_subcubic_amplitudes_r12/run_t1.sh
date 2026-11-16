#!/bin/bash
# Round 12 T1 driver: scored arms first (<= 4 single-threaded processes at a time), dense last.
# Usage: bash run_t1.sh scored          (full, half, quarter on toys 20-23)
#        bash run_t1.sh dense "23 21" 90  (dense on the listed toys, SR cap in CPU-s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
mkdir -p logs
run() {  # arm inst cap
  ( ulimit -t 600; python3 t1_sub.py --mode train --arm "$1" --instance "$2" --max-cpu "$3" \
      > "logs/t1_$1_inst$2.log" 2>&1 )
}
if [ "$1" = scored ]; then
  run full 23 200 & run half 23 200 & run quarter 23 200 & run full 20 90 & wait
  run half 20 90 & run quarter 20 90 & run full 21 90 & run half 21 90 & wait
  run quarter 21 90 & run full 22 90 & run half 22 90 & run quarter 22 90 & wait
elif [ "$1" = dense ]; then
  for i in $2; do run dense "$i" "${3:-90}" & done; wait
fi
