#!/bin/bash
#SBATCH --job-name=P3.1_r3
#SBATCH --partition=kim
#SBATCH --array=0-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.1 round-3 decisive test: order-parameter error estimators on SYMMETRY-PROJECTED states.
# Routes: ECC (3.1, ensemble-calibrated covariance) and ZR (3.2, Hylleraas-trained response
# network); controls SD (seed spread), ZV (zero-variance extrapolation), L1 (one Lanczos
# step); ablations SMA, KR2; oracle ZRO. CPU only.
# One array task = one (instance, seed): instance = t // 6, seed = (101..106)[t % 6]. Instances
# (t'=-0.25): 0 4x3 open 4+4 U=8; 1 4x3 cylinder 4+4 U=8; 2 6x2 ladder 5+5 U=8 (lower symmetry,
# P1.2 r3 probe); 3 4x3 torus 5+5 U=8; 4 4x3 torus 4+4 U=8; 5 4x3 torus 5+5 U=4. Instances whose
# projected states are already exact are reported NO-TARGET by --aggregate (premise gate).
# Populations per task: projected NNBF w2, w8 (P_chi) and w8 (P_S P_chi) with snapshots at SR steps 30 and 100 plus the final
# state; projected Slater-Jastrow foil (P_chi, P_S P_chi); reported only: plain NNBF w8 and its
# post-hoc projection. 400 exact-force SR steps each.
# Submit from this directory so the logs land next to the script:
#   cd <this dir> && sbatch submit_test.sbatch
# Cost (measured on the login node, 16 threads, D=627264, timing_output.txt): one projected SR
# step 4.3 s (P) / 4.8 s (PS); one response-network Adam step about 2.5 s; one A application
# 0.46 s, one H application 0.05 s. Per task on the D=627264 instances: 6 x 400 SR steps
# = about 3.0 h; 13 analysed states x (300 ZR steps = 12.5 min + MINRES and estimators,
# at most about 7 min) = at most about 4.2 h; total at most about 7.5 h. The D=245025
# instances (0, 1, 4) cost about 40% of that. 24 h leaves a > 3x margin. Memory: P1.1-P peaked at 3.4 GB on the
# same clusters with the same arrays; 16G gives > 4x margin.
# Afterwards (seconds, any node): python3 micro_test.py --aggregate results
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --sr-iters 400 --snaps 30,100 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 \
    --populations P_w2,P_w8,PS_w8,F_P,F_PS,U_w8 --r-min 1.5 \
    --mc-samples 100000 --n-boot 200 --zr-width 8 --zr-steps 300 --zr-lr 0.02
# Pre-committed AMBIGUOUS follow-up (only if a route's verdict is AMBIGUOUS): rerun with
# --mc-samples 1000000 and seeds 107..112 (edit SEEDS) into results_ambig/.
