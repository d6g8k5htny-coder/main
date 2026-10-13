#!/bin/sh
# P2c (calibration for P2): if the generator emitted NO packet, does the CI gate notice?
# Copy cd32303, make packet_descriptors() return [], regenerate, run museum_check.
H=$(cd "$(dirname "$0")" && pwd); N="$H/wt_p2c"
rm -rf "$N"; cp -a "<SCRATCH>/museum/cd32303" "$N"; rm -rf "$N/tools/__pycache__"
python3 -I -c "
import sys;p=sys.argv[1];s=open(p).read()
s=s.replace('    prefix = \"incoming/side24-identity-replay-20260926/\"\n    return [dict(','    prefix = \"incoming/side24-identity-replay-20260926/\"\n    return [] and [dict(',1);open(p,'w').write(s)" "$N/tools/museum_data.py"
export GIT_DIR=$H/objstore.git GIT_OPTIONAL_LOCKS=0 PATH=/opt/node22/bin:$PATH
python3 -B -E -s "$N/tools/museum_data.py" --main-root "$N" --math-root "$H/fixture/Math-"
python3 -I -c "import json,sys;print('packets in regenerated museum.json:',json.load(open(sys.argv[1]))['packets'])" "$N/docs/site/museum.json"
cd "$N" && python3 -B -E -s tools/museum_check.py --math-root "$H/fixture/Math-" > "$H/out_p2c_check.txt" 2>&1; echo "museum_check exit=$?"
grep -E '^# (pass|fail)|^not ok|packet' "$H/out_p2c_check.txt" | head -6
