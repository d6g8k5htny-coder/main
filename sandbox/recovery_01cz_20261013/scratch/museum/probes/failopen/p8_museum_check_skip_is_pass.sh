#!/bin/sh
# museum_check.py treats `node --test` exit 0 as "actual display integration verified" without asserting that the
# pinned integration test executed. Scratch copy; the only change is the env var name museum_check exports
# (simulating drift between the checker and the test's skip predicate). Offline: Math- from the verified fixture.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); W="$HERE/work/p8"; rm -rf "$W"; cp -r "$HERE/../../cd32303" "$W"; find "$W" -name __pycache__ -prune -exec rm -rf {} +
python3 -I - "$W/tools/museum_check.py" <<'PY'
import sys; p=sys.argv[1]; s=open(p).read(); o='MUSEUM_FIXTURE=str(manifest)'; assert s.count(o)==1; open(p,'w').write(s.replace(o,'MUSEUM_FIXTURE_PATH=str(manifest)'))
PY
cd "$W" && PATH=/opt/node22/bin:$PATH python3 -B tools/museum_check.py --math-root "$HERE/fixture/Math-" > "$HERE/work/p8.log" 2>&1; rc=$?
grep -E "^# (tests|pass|fail|skipped) |^ok [0-9]+ - actual pinned|integration verified" "$HERE/work/p8.log"; echo "museum_check exit=$rc"; rm -f "$HERE/work/p8.log"
rm -rf "$W"
