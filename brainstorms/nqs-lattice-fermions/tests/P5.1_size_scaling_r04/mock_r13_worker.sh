#!/bin/bash
# Mock test of submit_r13_s14_ladder.sbatch's worker logic (no physics, no Slurm): a fake ns_run.py that charges
# 10 "min" per exact step and 1 per guided step against --cpu-cap, and a fake step-1 scorer.  Checks (1) chunked
# resume across workers until all items are DONE and later workers exit at once; (2) the STOP_r13 path; (3) USR1
# forwarding (TRUNCATED, worker exits 0, no later item started).  Usage: bash mock_r13_worker.sh
SRC=$(cd "$(dirname "$0")" && pwd)
mk() {
  local M; M=$(mktemp -d /tmp/r13mock.XXXX); mkdir -p "$M/results" "$M/logs" "$M/starts"
  sed "s#^DIR=.*#DIR=$M#; s#^PY=.*#PY=python3#" "$SRC/submit_r13_s14_ladder.sbatch" > "$M/w.sh"
  cat > "$M/ns_run.py" <<'EOF'
import argparse, json, os, signal, time
p = argparse.ArgumentParser()
for x in ['--Lx', '--Ly', '--Ne', '--start', '--seed0', '--sampler', '--nsamp', '--res', '--tau']: p.add_argument(x)
p.add_argument('--steps', type=int); p.add_argument('--cpu-cap', type=float); p.add_argument('--exact', action='store_true')
a = p.parse_args(); per = 10.0 if a.exact else 1.0
os.makedirs(a.res, exist_ok=True); fn = a.res + '/run.json'
n = json.load(open(fn))['n'] if os.path.exists(fn) else 0
stop = [False]
signal.signal(signal.SIGUSR1, lambda s, f: stop.__setitem__(0, True)); signal.signal(signal.SIGTERM, lambda s, f: stop.__setitem__(0, True))
used = 0; st = 'RUNNING'
while n < a.steps:
    if used + per > a.cpu_cap or stop[0]: st = 'TRUNCATED'; break
    time.sleep(0.2 if os.environ.get('SLOW') else 0.001); n += 1; used += per
if n >= a.steps: st = 'DONE'
json.dump(dict(n=n, status=st), open(fn, 'w')); print(a.res, n, st)
EOF
  cat > "$M/score_r13_step1.py" <<'EOF'
import json, os, sys
v = os.environ.get('V', 'SUPPORT-LIKE'); json.dump({"verdict": v}, open('results/step1_r13.json', 'w'), indent=1)
sys.exit(0 if v == 'SUPPORT-LIKE' else 3)
EOF
  echo "$M"
}
echo "=== (1) chunked resume, LIMIT_MIN 420 (400 'min' per worker)"
M=$(mk); cd "$M" || exit 1
for i in 0 1 2 3 4 5; do LIMIT_MIN=420 bash w.sh | grep -v '^P5.1'; echo "-- worker $i rc ${PIPESTATUS[0]}"; done
for f in results/*/run.json; do echo "$f $(cat "$f")"; done
echo "=== (2) STOP path"
M=$(mk); cd "$M" || exit 1
V=AMBIGUOUS LIMIT_MIN=420 bash w.sh; echo "rc $?"; ls results; LIMIT_MIN=420 bash w.sh; echo "rc $?"
echo "=== (3) USR1 forwarding"
M=$(mk); cd "$M" || exit 1
SLOW=1 LIMIT_MIN=420 bash w.sh & P=$!; sleep 3; kill -USR1 $P; wait $P; echo "rc $?"
for f in results/*/run.json; do echo "$f $(cat "$f")"; done
