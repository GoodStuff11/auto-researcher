#!/bin/bash
# P4.1 r20: sampled truncated-metric LM viability T1 at 12 sites (che only). Rule: RULE.md (pre-registered + amendment 1).
# Prepared, NOT submitted by the worker; submit only through compute_gate.py submit with voi.md.
# Arms: tu64k (Toulouse-Umrigar: non-symmetric H, a_diag chosen on held-out samples, 65536/step), tu8k (same, 8192/step),
# naive64k (r19 protocol at 65536/step).
# Wall time (measured, 1 thread): tu64k step 2.4 CPU-min + 1.6 min set-up (logs/probe_tu64k_1step_1thread.txt);
# tu8k step <= 2.4 (+1.5 min exact solver check); naive step 1.9 at 8k (r19 log) + ~0.6 for 64k sampled S/H ~ 2.5.
# 4 x (2.4 + 2.4 + 2.5) = 29 min + 3 x 1.6 set-up + 2.5 checks ~ 37 min; x1.3 = 48 -> 0:50:00.
# Memory: 11.8 GB peak RSS at 64k (probe); naive64k holds Y, HY and 64k-row copies -> --mem=20G.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00:50:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r20/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r20/logs/%x_%j.err
set -uo pipefail
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P4.1_cheap_su2_r20
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
echo "r20 on $(hostname), job ${SLURM_JOB_ID:-local}"

# Do not overwrite a finished run (rerunning-jobs rule); partial arms resume from results/*.json.
if [ -f results/verdict.json ] && grep -qE '"verdict": "(VIABLE|MARGINAL|NOT-VIABLE)"' results/verdict.json; then
    echo "results/verdict.json already holds a verdict; nothing to do"; exit 0
fi
set -e
# most decisive arm first
$PY -u lm_viability.py --mode tu --nsamp 65536 --steps 4 --threads 1 --cpu-cap 16 | tee -a logs/tu64k_output.txt
$PY -u lm_viability.py --mode tu --nsamp 8192 --steps 4 --threads 1 --cpu-cap 18 | tee -a logs/tu8k_output.txt
[ -f results/sampled_che_N65536.json ] || \
    $PY -u lm_viability.py --mode naive --nsamp 65536 --steps 4 --threads 1 --cpu-cap 18 | tee logs/naive64k_output.txt
$PY -u lm_viability.py --mode verdict | tee logs/verdict_output.txt
