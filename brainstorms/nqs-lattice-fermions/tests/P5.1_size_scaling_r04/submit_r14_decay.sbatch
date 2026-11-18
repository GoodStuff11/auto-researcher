#!/bin/bash
# DRAFT, DO NOT SUBMIT: ns_decay.py does not exist yet (RULE_r14_decay.md lists its identity test).
# P5.1 R51.7 `rmavg` unshelve test at 10 and 12 sites (RULE_r14_decay.md, round 14). Submit only through
# compute_gate.py submit with voi_r14_decay.md.
# 18 continuations of 60 steps (CONT fixed 0.005; DEC decaying + Polyak-Ruppert average) from the step-120 states of
# R51.8's arms: S10 exact, S10 guided 4k s9000, 16k and 65k s9000/s19000; S12' guided 16k, 65k, 262k s9000.
# Wall time from in-job rates of 763968/769098/769376 (CPU-min per 120 steps, halved for 60): S10 exact 7.2,
# guided 4k ~2, 16k 3.0, 65k 4.2; S12' guided 16k 16.0, 65k 26.6, 262k 39.2; DEC adds one exact energy per step
# (S12' 0.06 CPU-min). Sum: S10 2x(7.2+2+2x3+2x4.2) = 47 min; S12' 2x81.8 + 11 = 175 min; setups ~15 min
# -> 237 min x 1.3 = 308 min -> 06:00:00.
#SBATCH --job-name=hpb-nqs-fermions-P5.1-R51.7-001
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_size_scaling_r04/logs/%x_%j.err
set -euo pipefail
echo "DRAFT: ns_decay.py not written; refusing to run" >&2
exit 1
