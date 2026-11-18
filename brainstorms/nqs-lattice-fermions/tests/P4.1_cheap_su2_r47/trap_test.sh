#!/bin/bash
# Local test of submit_r47_t1.sbatch's USR1 trap forwarding (same run() and trap; smoke results dir).
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
DIR=$(cd "$(dirname "$0")" && pwd)
cd "$DIR"
PY=~/.conda/envs/lit_env/bin/python
CHILD=""
GOT=0
trap 'GOT=1; [ -n "$CHILD" ] && kill -USR1 "$CHILD" 2>/dev/null' USR1
run() {
    "$@" &
    CHILD=$!
    wait "$CHILD"; rc=$?
    while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD"; rc=$?; done
    CHILD=""
    return $rc
}
for spec in nnbf_u:5 che:5; do
    [ "$GOT" = "1" ] && echo "shell: stopping after USR1" && exit 4
    run $PY -u r47_t1.py --mode train --arm "${spec%%:*}" --maxfun 300 --cpu-min "${spec##*:}" --res "$DIR/results_trap" || { rc=$?; echo "child rc $rc"; exit $rc; }
done
echo "all done"
