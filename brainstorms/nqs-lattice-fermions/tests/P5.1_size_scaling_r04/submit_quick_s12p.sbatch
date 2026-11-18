#!/bin/bash
# P5.1 R51.8 / R51.10 quick T1 (RULE.md revision 1, RULE_kappa.md revision 1): train the S12' start (6x2, N_e 10,
# same protocol as the S10 and 4x3 starts) and compute its exact kappa and --info.  Prepared round 4, NOT submitted
# by the worker; submit only through compute_gate.py submit with voi_quick_s12p.md.
# Wall time (interactive probes, logs/): vmc.py training 20 iterations at 6x2 = 48 CPU-s incl. about 24 s setup and
# exact eval -> 1.2 s/iteration -> 1000 iterations 20 min + 0.5 min eval; kappa (two passes of 156,816 rows at
# 6.9e-6 CPU-min/row = 2.2 min + energy/eigh 0.5) 2.7 min; info 0.3 min.  Sum 24 min interactive x 1.8 (in-job /
# interactive, jobs 759974, 761860) = 43 min x 1.3 = 56 min -> --time 01:00:00.  Memory: kappa 14.5 GB (4x3,
# unchunked) -> chunked now; 6x2 exact pre-flight 11 GB -> 16G.
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR/../P4.1_cheap_su2_scaling"
echo "P5.1 quick T1 S12' on $(hostname), job ${SLURM_JOB_ID:-local}"
$PY -u vmc.py --mode train --Lx 6 --Ly 2 --Ne 10 --pg 1,1 --arm che --width 16 --seed 0 --iters 1000 --lr 0.05 \
  --lr-decay 0.1 --avg-frac 0.25 --diag-shift 1e-3 --max-step 0.5 --n-chains 1024 --sweeps-per-step 0.5 --exact-eval \
  --threads 1 --outdir "$DIR/starts" | tee -a "$DIR/logs/start_6x2.txt"
cd "$DIR"
ST=starts/S6x2_N10_U8_tp-0.25_pg11_che_seed0_theta.npy
$PY -u ns_run.py --Lx 6 --Ly 2 --Ne 10 --start "$ST" --info | tee -a logs/info_S6x2.txt
$PY -u kappa.py --Lx 6 --Ly 2 --Ne 10 --start "$ST" --tag S6x2_start | tee -a logs/kappa_S6x2_start.txt
