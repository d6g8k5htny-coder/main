#!/bin/bash
# The real lifetime-difference and d = 3 Theorem A checkers through tools/check_mode_replay.py,
# run as the workflow steps run them: `python -B -S -` from the repository root, so the
# import resolves through the working tree; each family is one job in both interpreter
# modes. Each accepted record is timestamped; at the end every stdout and stderr digest is
# compared with the characterization run's.
#
# Usage: real_replay_b4.sh WORKTREE RUN1DIR [JOB...]   (JOB: lifetime, d3)
set -uo pipefail
C=/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/C14
WT=$1; RUN1=$2; shift 2
JOBS=${*:-lifetime d3}
cd "$WT" || exit 2
export TMPDIR="$C/tmp" PYTHONDONTWRITEBYTECODE=1
echo "start $(date -u +%T) head $(git rev-parse HEAD) contracts $(sha256sum tools/check_mode_contracts.json | cut -c1-64) helper $(git hash-object tools/check_mode_replay.py) cpu_count $(nproc) python $(python3 -c 'import platform; print(platform.python_version())')"
status=0
for job in $JOBS; do
  case $job in
    lifetime) family=c8-lifetime-difference-constant; dir=frontiers/c8_lifetime_difference_constant_20261001 ;;
    d3) family=c8-theorem-a-constants-d3; dir=frontiers/c8_theorem_a_constants_d3_20261001 ;;
    *) echo "unknown job $job"; exit 2 ;;
  esac
  echo "job $job $(date -u +%T)"
  python3 -B -S - "$family" "$dir" "$RUN1/$family.json" <<'PY' 2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date -u +%T)" "$line"; done
import json, pathlib, sys, time
from tools.check_mode_replay import replay
family, directory, reference_path = sys.argv[1:4]
root = pathlib.Path(directory).resolve()
start = time.time()
records = replay(family, root)
seconds = round(time.time() - start, 1)
reference = {(r['mode'], r['label']): (r['stdout_sha256'], r['stderr_sha256'])
             for r in json.loads(pathlib.Path(reference_path).read_text())}
match = ({(r['mode'], r['label']) for r in records} == set(reference) and len(records) == len(reference) and all(
    reference[(r['mode'], r['label'])] == (r['stdout_sha256'], r['stderr_sha256']) for r in records))
print('REAL_REPLAY_OK', family, 'both-modes', len(records), match, seconds, flush=True)
PY
  rc=${PIPESTATUS[0]}
  echo "job $job exit=$rc $(date -u +%T)"
  [ "$rc" = 0 ] || status=1
done
echo "exit=$status end $(date -u +%T)"
exit $status
