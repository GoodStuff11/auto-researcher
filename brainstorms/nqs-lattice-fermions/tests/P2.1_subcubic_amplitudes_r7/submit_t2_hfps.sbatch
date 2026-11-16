#!/bin/bash
#SBATCH --job-name=P2.1r8_T2hfps
#SBATCH --partition=kim
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P2.1 round 8, T2 pilot of Route 7.1 (LH-HFPS, pairing frame + density Jastrow) on instance 4
# (5x3 OBC 6+6 t-t'-J), seed 11, 75 projected SR steps; the protocol, thresholds and early stop of
# the Route 6.1' T2 (job 698229). Rule: leaf artifact, Round 8 "T2 rule, Route 7.1".
#   task 0 hfps_plus (route, plus-shaped 5-site patch, F = 4)
#   task 1 hfps_site (foil: one-site features = pairing lattice backflow with long-range K)
#   task 2 agpj      (reported only: projected BCS x tanh Jastrow, the round-5 Foil arm)
# Reference: nnbf from job 698229 (results/t2_nnbf_inst4_s11.json), not rerun.
# Verdict (after this job AND 698229): python3 t2_routes.py --mode aggregate --route hfps
# Budget: see the leaf (pre-flight step timing at 4 threads). Existing JSONs are skipped.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/share/apps/software/anaconda3/bin/python3
${PY} t2_routes.py --mode train --route hfps --task "${SLURM_ARRAY_TASK_ID}" --instance 4 --seed 11 --F 4 \
    --sr-iters 75 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
    --lbfgs-iters 4000 --fn-iters 10 --fn-floor 1e-3
