#!/bin/bash
#SBATCH --job-name=P3.1r7_excK6
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=32:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# P3.1 round 7, step 1.5b (2026-11-14): extends the GO'd K=3 excited-state pilot (job 456094) to K=6 so the
# step-2 premise (eps_k/omega_k <= 0.4 for every k <= max K_1/2 over scored observables) is checked for spin
# (K_1/2 = 6). phi_1..3 are reloaded from results_excpilot/params (resume, --overwrite 0); phi_4..6 are trained
# with the same settings. The K=3 summary is preserved as results_excpilot/excited_pilot_K3_456094.json.
# Pre-registered rule: see .subproblems/P3.1_order_parameter_error.md, "Step 1.5b".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 excited_pilot.py --mode run --instance main --K 6 --beta 2.0 \
    --seed 11 --width 6 --sr-iters 1200 --n-low 30 --results results_excpilot
