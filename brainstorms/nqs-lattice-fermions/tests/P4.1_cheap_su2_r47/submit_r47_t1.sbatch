#!/bin/bash
# P4.1 R47 `distill-24` T1 at 12 sites (RULE_T1.md + amendment 2026-11-18). Card: leaves/P4.1/cards/R47.md.
# Prepared round 34, NOT submitted; submit only through compute_gate.py submit with voi.md.
# task 0: nnbf_p (exactly projected nnbf, sampled fit).  task 1: che, nnbf_u, nnbf_even in turn, then the DMRG
# rehearsal (r47_dmrg.jl) and its check.  Each task runs verdict at its end (the later one writes the full verdict).
# Wall time (measured 2026-11-18 interactively, 1 thread, logs/check_local.txt): CPU-min per evaluation che 0.00554,
# nnbf_u 0.00309, nnbf_even 0.00434, nnbf_p 0.0204; exact-metrics log 0.026-0.039 per call (80 calls); setup 0.3.
# 4000 evaluations -> che 25.6, nnbf_u 14.8, nnbf_even 19.7, nnbf_p 84.1 CPU-min interactive; x in-job slowdown
# (brief's R45 in-job rates 0.19/0.117 = 1.62 for che, 0.12/0.065 = 1.85 for nnbf-type) x 1.3 margin
# -> CPU caps che 54, nnbf_u 36, nnbf_even 47, nnbf_p 202.  DMRG rehearsal: smoke (maxdim 200) 3.4 min incl. Julia
# compile; full schedule (maxdim 1000) estimated 20 min at 1 thread -> 30.  Task 1 total 168 min; time limit 03:40:00.
# Cap 2 x 3.67 = 7.3 CPU-h; expected about 4.7 CPU-h.  Memory: peak RSS 2.8 GB (check, all arms), 3.0 GB (Julia smoke).
# SIGUSR1 600 s before the limit: the batch shell traps it and forwards it to the running child (python checkpoints,
# exits 4; resubmitting resumes from the best theta; finished arms and an existing DMRG output are skipped).
#SBATCH --job-name=P41r47t1
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:50:00
#SBATCH --array=0-1
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r47/logs/%x_%A_%a.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r47/logs/%x_%A_%a.err
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r47
cd "$DIR"
PY=~/.conda/envs/lit_env/bin/python
T=${SLURM_ARRAY_TASK_ID:-0}
CHILD=""
GOT=0
trap 'GOT=1; [ -n "$CHILD" ] && kill -USR1 "$CHILD" 2>/dev/null' USR1
run() {  # child in the background so the trap can forward USR1; wait returns early on the signal, so loop
    "$@" &
    CHILD=$!
    wait "$CHILD"; rc=$?
    while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD"; rc=$?; done
    CHILD=""
    return $rc
}
echo "r47 T1 task $T on $(hostname), job ${SLURM_JOB_ID:-local}"
if [ "$T" = "0" ]; then
    run $PY -u r47_t1.py --mode train --arm nnbf_p --maxfun 4000 --cpu-min 208 --res "$DIR/results" || exit $?
else
    for spec in che:54 nnbf_u:36 nnbf_even:47; do
        [ "$GOT" = "1" ] && exit 4
        run $PY -u r47_t1.py --mode train --arm "${spec%%:*}" --maxfun 4000 --cpu-min "${spec##*:}" --res "$DIR/results" || exit $?
    done
    [ "$GOT" = "1" ] && exit 4
    run ~/.juliaup/bin/julia --project=../P4.1_cheap_su2_scaling/dmrg_env -t 1 r47_dmrg.jl --threads=1 --outdir="$DIR/dmrg_out" || exit $?
    run $PY -u r47_t1.py --mode dmrgcheck --dmrg-dir "$DIR/dmrg_out" --res "$DIR/results" || exit $?
fi
run $PY -u r47_t1.py --mode verdict --res "$DIR/results"
