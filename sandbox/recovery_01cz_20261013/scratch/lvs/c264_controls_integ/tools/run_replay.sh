#!/usr/bin/env bash
# Full replay from the committed tree, exactly as the task names it; a /proc sampler records the peak VmHWM of any
# lean/leanchecker process per Lean step (read-only), in a separate file.
set -u
export PATH="$HOME/.elan/bin:$PATH"
R=<SCRATCH>/repos/math_c264ctl
I=<SCRATCH>/lvs/c264_controls_integ
while pgrep -x lean >/dev/null || pgrep -x leanchecker >/dev/null; do sleep 10; done
cd "$R"
rm -rf "$R/.lake/torus-filtration-evidence"
unset LEAN_NUM_THREADS PYTHONOPTIMIZE
START=$(date -u +%Y-%m-%dT%H:%M:%SZ); S0=$(date +%s)
( peak=0; while :; do for p in $(pgrep -x leanchecker; pgrep -x lean); do h=$(awk '/^VmHWM:/ {print $2}' /proc/$p/status 2>/dev/null); c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-160); [ -n "${h:-}" ] && echo "$(date +%s) $p $h $c"; done; sleep 1; done ) > "$I/replay_mem_samples.txt" 2>/dev/null &
SAMPLER=$!
code=0
bash companions/torus_filtration_20261011/replay.sh > "$I/replay.out" 2>&1 || code=$?
kill $SAMPLER 2>/dev/null
END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "replay exit=$code start=$START end=$END wall=$(( $(date +%s) - S0 ))s head=$(git rev-parse HEAD)" > "$I/replay_exit.txt"
cat "$I/replay_exit.txt"
