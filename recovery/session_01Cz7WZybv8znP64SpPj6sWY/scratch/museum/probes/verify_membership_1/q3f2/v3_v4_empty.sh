#!/bin/sh
# V3: the cd32303 browser validator with the REAL cd32303 manifest, only packets set to [] (snapshot 71400b94 holds a packet).
# V4: the generator made to emit packets=[] (packet_descriptors() -> []): does its summary line change; does CI notice?
H=$(cd "$(dirname "$0")" && pwd); CD="$H/../../../cd32303"; export PATH=/opt/node22/bin:$PATH GIT_OPTIONAL_LOCKS=0
unset GIT_DIR GIT_WORK_TREE; mkdir -p "$H/tmp"; export TMPDIR="$H/tmp"
echo "=== V3: git ls-tree -d 71400b94 incoming/: $(GIT_DIR="$H/objstore.git" git ls-tree -d --name-only 71400b94f6cb354a8cf7aba73ffede2138a64efa incoming/)"
python3 -I -c "import json,sys;d=json.load(open(sys.argv[1]));d['packets']=[];open(sys.argv[2],'w').write(json.dumps(d,indent=2,ensure_ascii=False,sort_keys=True)+'\n')" "$CD/docs/site/museum.json" "$H/v3_packets_empty.json"
node "$H/render.mjs" "$CD" "$H/v3_packets_empty.json"
echo "=== V4: generator with packet_descriptors() returning []"
N="$H/wt_empty"; rm -rf "$N"; cp -a "$CD" "$N"; find "$N" -name __pycache__ -prune -exec rm -rf {} +
python3 -I -c "
import sys;p=sys.argv[1];s=open(p).read();old='    prefix = \"incoming/side24-identity-replay-20260926/\"\n    return [dict('
assert s.count(old)==1;open(p,'w').write(s.replace(old,'    prefix = \"incoming/side24-identity-replay-20260926/\"\n    return [] and [dict(',1))" "$N/tools/museum_data.py"
cd "$N"; GIT_DIR="$H/objstore.git" python3 -B -E -s tools/museum_data.py --main-root "$N" --math-root "$H/fixture/Math-"; echo "generator exit=$?"
python3 -I -c "import json,sys;print('packets in regenerated museum.json:',json.load(open(sys.argv[1]))['packets'])" docs/site/museum.json
python3 -B -m unittest discover -s tests -p test_museum_data.py > "$H/v4_unittest.log" 2>&1; echo "unittest exit=$? ($(grep -E '^(FAIL|ERROR):' "$H/v4_unittest.log" | tr '\n' ' '))"
GIT_DIR="$H/objstore.git" python3 -B tools/museum_check.py --math-root "$H/fixture/Math-" > "$H/v4_check.log" 2>&1; echo "museum_check exit=$?"
grep -E '^# (pass|fail) |^not ok|did not match' "$H/v4_check.log" | head -4
