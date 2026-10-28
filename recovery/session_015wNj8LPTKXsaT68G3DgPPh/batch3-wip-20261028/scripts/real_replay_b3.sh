#!/bin/bash
# The real theorem-A planar and torus checkers through tools/check_mode_replay.py,
# run as the workflow steps run them: `python -B -S -` from the repository root,
# so the import resolves through the working tree. Theorem-A runs as its one job;
# torus runs as its two matrix jobs, one after the other (FLAGS per job). Each
# accepted record is timestamped; at the end every stdout and stderr digest is
# compared with run 1's.
#
# Usage: real_replay_b3.sh WORKTREE RUN1DIR [JOB...]   (JOB: theorem-a, torus-normal, torus-optimized)
set -uo pipefail
C=/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/C14
WT=$1; RUN1=$2; shift 2
JOBS=${*:-theorem-a torus-normal torus-optimized}
cd "$WT" || exit 2
export TMPDIR="$C/tmp" PYTHONDONTWRITEBYTECODE=1
echo "start $(date -u +%T) head $(git rev-parse HEAD) contracts $(sha256sum tools/check_mode_contracts.json | cut -c1-64) helper $(git hash-object tools/check_mode_replay.py)"
status=0
for job in $JOBS; do
  case $job in
    theorem-a) family=c8-theorem-a-constants-planar; dir=frontiers/c8_theorem_a_constants_planar_20260930; flags= ;;
    torus-normal) family=c8-torus-vanishing-remainder; dir=frontiers/c8_torus_vanishing_remainder_20261001; flags='-B -S' ;;
    torus-optimized) family=c8-torus-vanishing-remainder; dir=frontiers/c8_torus_vanishing_remainder_20261001; flags='-B -O -S' ;;
    *) echo "unknown job $job"; exit 2 ;;
  esac
  echo "job $job $(date -u +%T)"
  FLAGS="$flags" python3 -B -S - "$family" "$dir" "$RUN1/$family.json" <<'PY' 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date -u +%T)" "$line"; done
import json, os, pathlib, sys, time
from tools.check_mode_replay import replay
family, directory, reference_path = sys.argv[1:4]
root = pathlib.Path(directory).resolve()
flags = os.environ['FLAGS'].split() or None
start = time.time()
records = replay(family, root, flags=flags) if flags else replay(family, root)
seconds = round(time.time() - start, 1)
reference = {(r['mode'], r['label']): (r['stdout_sha256'], r['stderr_sha256'])
             for r in json.loads(pathlib.Path(reference_path).read_text())}
expected = {key for key in reference if flags is None or key[0] == ('optimized' if '-O' in flags else 'normal')}
match = ({(r['mode'], r['label']) for r in records} == expected and len(records) == len(expected) and all(
    reference[(r['mode'], r['label'])] == (r['stdout_sha256'], r['stderr_sha256']) for r in records))
print('REAL_REPLAY_OK', family, ' '.join(flags or ['both-modes']), len(records), match, seconds, flush=True)
PY
  rc=${PIPESTATUS[0]}
  echo "job $job exit=$rc $(date -u +%T)"
  [ "$rc" = 0 ] || status=1
done
echo "exit=$status end $(date -u +%T)"
exit $status
