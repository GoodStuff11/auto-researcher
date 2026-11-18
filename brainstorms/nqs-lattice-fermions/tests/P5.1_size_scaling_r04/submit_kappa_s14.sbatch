#!/bin/bash
# P5.1 R51.10 / R51.8 T1 at S14 (RULE_kappa.md revision 1): train the S14 start (7x2, N_e 12, same protocol, no exact
# eval inside vmc.py) and compute its exact kappa (chunked, two passes) and --info.  Prepared round 4, NOT submitted
# by the worker; submit only through compute_gate.py submit with voi_kappa_s14.md.
# Wall time (interactive probes, logs/): training 20 iterations at 7x2 = 48 CPU-s incl. about 14 s setup and a
# 4096-sample eval -> 1.7 s/iteration -> 1000 iterations 28 min + 1 min eval; --info at S14 (probe start,
# logs/preflight_S7x2_info_probe.txt): setup 0.23, one exact energy 1.24 CPU-min, rows 1.02e-5 CPU-min each ->
# 2,255,421 rows = 23.0 CPU-min per pass; kappa = 2 passes 46 + 2 energies 2.5 + eigh/products 2 = 51 min; info
# 1.5 min.  Sum 82 min interactive x 1.8 = 148 min x 1.3 = 192 min -> --time 03:15:00.
# Memory: --info at S14 peaked at 16.3 GB (sector, Hr with 7.6e7 nonzeros, configurations); kappa adds chunks of
# 32768 x 3590 doubles (0.9 GB) and P x P accumulators (0.1 GB each) -> 32G.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:15:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR/../P4.1_cheap_su2_scaling"
echo "P5.1 T1 S14 start + kappa on $(hostname), job ${SLURM_JOB_ID:-local}"
$PY -u vmc.py --mode train --Lx 7 --Ly 2 --Ne 12 --pg 1,1 --arm che --width 16 --seed 0 --iters 1000 --lr 0.05 \
  --lr-decay 0.1 --avg-frac 0.25 --diag-shift 1e-3 --max-step 0.5 --n-chains 1024 --sweeps-per-step 0.5 \
  --n-eval 65536 --threads 1 --outdir "$DIR/starts" | tee -a "$DIR/logs/start_7x2.txt"
cd "$DIR"
ST=starts/S7x2_N12_U8_tp-0.25_pg11_che_seed0_theta.npy
$PY -u ns_run.py --Lx 7 --Ly 2 --Ne 12 --start "$ST" --info | tee -a logs/info_S7x2.txt
$PY -u kappa.py --Lx 7 --Ly 2 --Ne 12 --start "$ST" --tag S7x2_start | tee -a logs/kappa_S7x2_start.txt
