#!/bin/bash
# P5.1 R51.11 `qopt` size step 10 -> 12 sites under R51.8's protocol with q ~ |psi| (RULE_r14_qladder.md, round 14).
# NOT submitted by the worker; submit only through compute_gate.py submit with voi_r14_qladder.md, after the
# orchestrator releases RULE_R51.11_vol3.md's ladder hold.
# Arms (ns_q.py --qmix 1.0 = ns_run.py's fixed-length step with q ~ |psi|), tau 0.005, 120 steps, seed0 as listed:
#   S10 (5x2, N_e 8): 16384 s9000, 16384 s19000, 4096 s19000, 65536 s9000 (4096 s9000 reused: results/storm_N4096_a1.0_q1.0)
#   S12' (6x2, N_e 10): 16384, 65536, 4096, seed 9000.
# References: R51.8's exact arms (../P5.1_size_scaling_r04/results/S{5x2_N8,6x2_N10}_exact_tau0.005_s9000).
# Wall time: in-job CPU-min/step with q ~ |psi| at the 4x3 sector (157,216 representatives, as 6x2's 156,816; job
# 769211): 16k 0.325, 65k 0.631; 4k about 0.15 (a + b x unique). S10 (11,100 representatives): <= the exact arm's
# 0.12. S12': 39 + 76 + 18 = 133 min; S10: about 26 min; setups about 7 min -> 166 min x 1.3 = 216 min -> 04:00:00.
# Resumability: ns_run.py saves state.npz every step and resumes on rerun; USR1 10 min before the limit is forwarded.
#SBATCH --job-name=hpb-nqs-fermions-P5.1-R51.11-002
#SBATCH --partition=kim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r05_gen/logs/%x_%j.out
#SBATCH --error=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r05_gen/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=/home/jek354/research/.claude/worktrees/nqs-brainstorm/brainstorms/nqs-lattice-fermions/tests/P5.1_r05_gen
PY=~/.conda/envs/lit_env/bin/python
cd "$DIR"
SD=../P5.1_size_scaling_r04/starts
ST10=$SD/S5x2_N8_U8_tp-0.25_pg11_che_seed0_theta.npy
ST12=$SD/S6x2_N10_U8_tp-0.25_pg11_che_seed0_theta.npy
STOP=0; CHILD=
trap 'STOP=1; if [ -n "$CHILD" ]; then kill -USR1 "$CHILD" 2>/dev/null || true; fi' USR1 TERM
run() {  # run <tag> <cap CPU-min> <Lx> <Ly> <Ne> <start> <N> <seed0>
  [ "$STOP" = 1 ] && return 0
  local tag=$1 cap=$2 lx=$3 ly=$4 ne=$5 st=$6 n=$7 s=$8
  $PY -u ns_q.py --qmix 1.0 --Lx $lx --Ly $ly --Ne $ne --start "$st" --sampler guided --nsamp $n --tau 0.005 \
    --steps 120 --seed0 $s --cpu-cap "$cap" --res results/q1L_$tag >> logs/q1L_$tag.txt 2>&1 &
  CHILD=$!
  local rc=0; wait "$CHILD" || rc=$?
  while kill -0 "$CHILD" 2>/dev/null; do rc=0; wait "$CHILD" || rc=$?; done
  CHILD=; return $rc
}
run S5x2_N16384_s9000  20 5 2 8  $ST10 16384 9000
run S5x2_N16384_s19000 20 5 2 8  $ST10 16384 19000
run S6x2_N16384_s9000  55 6 2 10 $ST12 16384 9000
run S6x2_N65536_s9000 100 6 2 10 $ST12 65536 9000
run S6x2_N4096_s9000   30 6 2 10 $ST12 4096 9000
run S5x2_N4096_s19000  15 5 2 8  $ST10 4096 19000
run S5x2_N65536_s9000  20 5 2 8  $ST10 65536 9000
echo "R51.11 r14 ladder done (STOP=$STOP)"
