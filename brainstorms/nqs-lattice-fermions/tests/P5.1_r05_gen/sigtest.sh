#!/bin/bash
# Test of the USR1-forwarding pattern used in submit_cmp.sbatch / submit_qopt.sbatch (run: bash sigtest.sh).
inner() {
  pid=""; stop=0
  trap 'stop=1; [ -n "$pid" ] && kill -USR1 "$pid" 2>/dev/null' USR1
  run() { [ "$stop" = 1 ] && { echo "skipped $1"; return; }
    python3 -c "
import signal, time, sys
s = {'x': 0}
signal.signal(signal.SIGUSR1, lambda *a: s.update(x=1))
for i in range(30):
    time.sleep(0.2)
    if s['x']:
        print('child $1 got USR1 at', i); time.sleep(0.5); print('child $1 saved'); sys.exit(0)
print('child $1 finished')" &
    pid=$!; while kill -0 "$pid" 2>/dev/null; do wait "$pid"; done; pid=""; }
  run A; run B; echo "end stop=$stop"
}
inner &
p=$!
sleep 1
kill -USR1 $p
wait $p
