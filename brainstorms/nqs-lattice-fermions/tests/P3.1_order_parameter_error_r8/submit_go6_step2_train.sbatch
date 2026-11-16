#!/bin/bash
#SBATCH --job-name=P3.1r8_go6s2
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.1 ESR0 step 2, PRE-PLANNED for the GO-6 branch of job 618646 (NOT submitted; written 2026-11-16).
# Trains phi_1..6 for the P2.1 width-6 ground states of seeds 22 and 33 (seed 11's six states come from
# 456094 + 618646), exactly as the pilot (penalty SR, beta = 2, 1200 iterations), and prints the exact-matrix
# ESR0(K=6) ratios. About 40 h per task (6 states x ~6.7 h at 16 CPUs), i.e. ~640 CPU-hours per task: far above
# the 24 CPU-hour T3 cap, so it needs the orchestrator's written exception. Round 8 Gate 4 FAIL (pair, charge)
# means a PASS here would be an ED-size diagnostic only. The sampled scorer (Metropolis mixture, fix (c)) is
# the remaining code to write while these run.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/../P3.1_order_parameter_error_r7"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
SEEDS=(22 33)
/share/apps/software/anaconda3/bin/python3 excited_pilot.py --mode run --instance main --K 6 --beta 2.0 \
    --seed "${SEEDS[$SLURM_ARRAY_TASK_ID]}" --width 6 --sr-iters 1200 --n-low 30 \
    --p21-json "../P2.1_subcubic_amplitudes_r3/results/inst0_nnbf_s${SEEDS[$SLURM_ARRAY_TASK_ID]}.json" \
    --results "results_step2_s${SEEDS[$SLURM_ARRAY_TASK_ID]}"
