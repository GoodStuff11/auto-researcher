#!/bin/bash
#SBATCH --job-name=P3.2r5
#SBATCH --partition=kim
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# P3.2 round 5: FRESH (out-of-sample) runs for the LCC verdict (CPU only). The training code is the
# round-4 fix script ../P3.2_biased_ensemble_fix/micro_test.py, unchanged (same instance, snapshots,
# cold and warm A/B branches, SO-EC and oracle EC), without the sampled-force arm.
#   tasks 0-2: width 6, seeds 1234/2345/3456          -> results_w6/w6_s<seed>.json
#   tasks 3-5: width 8, FRESH seeds 4567/5678/6789     -> results_w8fresh/w8_s<seed>.json
# Widths 4 and 8 with seeds 1234/2345/3456 (job 451852) are reused, in-sample, from ../P3.2_biased_ensemble_fix/results.
# Per task: 12 snapshot + 24 target trainings x 800 projected-SR steps. 451852 measured 1.6-2.1 s/step at
# width 8 on 16 threads (50-63 ks per task with 40 trainings) -> about 36/40 of that: 12.5-16 h; width 6 is cheaper.
# Submit from this directory:  cd <this dir> && sbatch submit_test.sbatch
# Afterwards (seconds):        python3 lcc.py --runs ../P3.2_biased_ensemble_fix/results --fresh results_w6 results_w8fresh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
T=${SLURM_ARRAY_TASK_ID}
if [ "$T" -lt 3 ]; then
    W=6; SEEDS=(1234 2345 3456); S=${SEEDS[$T]}; DIR=results_w6
else
    W=8; SEEDS=(4567 5678 6789); S=${SEEDS[$((T - 3))]}; DIR=results_w8fresh
fi
mkdir -p "${DIR}"
/share/apps/software/anaconda3/bin/python3 ../P3.2_biased_ensemble_fix/micro_test.py \
    --width "${W}" --seed "${S}" \
    --targets=-0.002,-0.0005,-0.0002,0.0002,0.0005,0.002 \
    --snaps=-0.08,-0.05,-0.03,0.03,0.05,0.08 --sf-targets= \
    --sr-iters 800 --sr-lr 0.05 --sr-shift 1e-3 --sr-samples 2000 --window 100 --z 2 \
    --n-samples 20000 --n-rep 20 --tau 0.02 \
    --out "${SLURM_SUBMIT_DIR}/${DIR}/w${W}_s${S}.json"
