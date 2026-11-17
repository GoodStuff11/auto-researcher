#!/bin/bash
# P4.1 rung 1 (Round 11 VOI note): exact truncated-metric LM polish of pcg (and che steps 5-8) at 12 sites,
# then the same polish with sampled S and H (8192 samples per step). Rule: RULE.md (pre-registered).
# Prepared, NOT submitted by the worker; submit only through compute_gate.py submit with voi.md.
# One job, not two: the sampled che arm is needed on every exact verdict; the sampled pcg arm runs only if the
# exact verdict keeps R2 alive AND sampled che did not already fix SAMPLED-LOSES (RULE.md amendment 1;
# polish.py --mode need-pcg-sampled). The local smoke test predicts it will be skipped (sampled che step 1: R = -4.9).
# Wall time (measured, 1 thread): exact step 3.0 min (logs/probe_pcg_1step_1thread.txt, 9.1 GB peak RSS); sampled
# che step 1 incl. the exact S/H check 2.3 min (logs/probe_sampled_che_1step_1thread.txt). 8 pcg + 4 che exact steps
# = 36 min, sampled che 4 x 2.3 = 9.2 min, set-up and verdicts ~4 min: ~49 min, x1.3 = 64 min -> 1:05:00.
# If sampled pcg is unexpectedly needed (+~9 min) it may be cut by the limit; it is then INCOMPLETE and resumable.
# Afterwards: logs/verdict_output.txt and results/verdict.json.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:05:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r19/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r19/logs/%x_%j.err
set -uo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r19
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
echo "rung 1 on $(hostname), job ${SLURM_JOB_ID:-local}"

# Do not overwrite a finished run (rerunning-jobs rule); partial results are resumed by polish.py.
if [ -f results/verdict.json ] && grep -q '"sampled_verdict": "SAMPLED-' results/verdict.json; then
    echo "results/verdict.json already holds a sampled verdict; nothing to do"; exit 0
fi

set -e
# (a) exact polish, up to 8 steps per arm (che resumes from Round 11 step 4)
$PY -u polish.py --mode seq --arm pcg --steps 8 --threads 1 --cpu-cap 40 | tee logs/seq_pcg_output.txt
$PY -u polish.py --mode seq --arm che --steps 8 --threads 1 --cpu-cap 25 | tee logs/seq_che_output.txt
$PY -u polish.py --mode verdict | tee logs/verdict_exact_output.txt

# (b) sampled polish, 4 steps, 8192 samples per step; pcg only when the exact verdict leaves R2 alive
[ -f results/sampled_che_N8192.json ] || \
    $PY -u polish.py --mode sampled --arm che --steps 4 --threads 1 --cpu-cap 20 | tee logs/sampled_che_output.txt
set +e
$PY -u polish.py --mode need-pcg-sampled
NEED=$?
set -e
if [ "$NEED" -eq 0 ] && [ ! -f results/sampled_pcg_N8192.json ]; then
    $PY -u polish.py --mode sampled --arm pcg --steps 4 --threads 1 --cpu-cap 20 | tee logs/sampled_pcg_output.txt
fi
$PY -u polish.py --mode verdict | tee logs/verdict_output.txt
