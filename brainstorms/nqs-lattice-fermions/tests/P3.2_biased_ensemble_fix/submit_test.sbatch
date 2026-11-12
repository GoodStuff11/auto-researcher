#!/bin/bash
#SBATCH --job-name=P3.2_fix
#SBATCH --partition=kim
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.2 round-4 fix: corrected decisive array (CPU only). One task per (width, seed):
#   width = (4, 8)[t // 3], seed = (1234, 2345, 3456)[t % 3].
# Per task: 6 snapshot points x 2 branches (12 trainings) + 6 targets x (cold A/B + warm A/B)
#   (24) + 2 sf targets x sampled-force warm A/B (4) = 40 trainings x 800 projected-SR steps;
#   ED with a global-E0 check at 12 points; sampled EC (20 replicas x 20000 draws).
# Cost: pilot 410917 measured 1.54 s/step at width 8 on 16 threads (r3 timing_output.txt:
#   2.04-2.19 s/step on 8 threads at widths 2-8). 40 x 800 x 1.6 s = 14.2 h (+ < 0.5 h ED and
#   sampling); worst case at 2.2 s/step 19.6 h; 24 h limit. Total about 6 x 15 h = 90 task-hours.
# Submit from this directory:  cd <this dir> && sbatch submit_test.sbatch
# Afterwards (seconds):        python3 micro_test.py --aggregate results
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/share/apps/software/anaconda3/bin/python3 micro_test.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --targets=-0.002,-0.0005,-0.0002,0.0002,0.0005,0.002 \
    --snaps=-0.08,-0.05,-0.03,0.03,0.05,0.08 --sf-targets=-0.0002,0.0002 \
    --sr-iters 800 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 --window 100 --z 2 \
    --n-samples 20000 --n-rep 20 --tau 0.02
