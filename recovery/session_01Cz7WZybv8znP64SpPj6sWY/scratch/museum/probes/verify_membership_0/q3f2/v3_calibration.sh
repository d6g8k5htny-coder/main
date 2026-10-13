#!/bin/sh
# v3_calibration.sh: what DOES the CI gate catch?
# (D1) committed museum.json hand-edited to packets:[] with the generator unchanged -> museum_check?
# (D2) generator changed so packet_descriptors() returns [] -> generator summary line + museum_check?
H=$(cd "$(dirname "$0")" && pwd); unset GIT_DIR; export GIT_OPTIONAL_LOCKS=0 PATH=/opt/node22/bin:$PATH
"$H/mkwt.sh" wt_d1 >/dev/null; cp "$H/cd32303_packets_empty.json" "$H/wt_d1/docs/site/museum.json"
cd "$H/wt_d1" && python3 -B tools/museum_check.py --math-root "$H/fixture/Math-" > "$H/wt_d1.check.log" 2>&1; echo "(D1) museum_check exit=$? :: $(grep -oE 'ValueError: .*' "$H/wt_d1.check.log")"
"$H/mkwt.sh" wt_d2 >/dev/null
python3 -I -c "
import sys;p=sys.argv[1];s=open(p).read();old='    return [dict(id=\"side24-identity-replay-20260926\"'
assert s.count(old)==1;open(p,'w').write(s.replace(old,'    return [] and [dict(id=\"side24-identity-replay-20260926\"'))" "$H/wt_d2/tools/museum_data.py"
cd "$H/wt_d2" && echo "(D2) generator stdout: $(python3 -B tools/museum_data.py --math-root "$H/fixture/Math-")"
python3 -I -c "import json,sys;print('     packets in regenerated museum.json:',json.load(open(sys.argv[1]))['packets'])" docs/site/museum.json
python3 -B tools/museum_check.py --math-root "$H/fixture/Math-" > "$H/wt_d2.check.log" 2>&1; echo "     museum_check exit=$? :: $(grep -E '^not ok' "$H/wt_d2.check.log" | head -1) :: $(grep -m1 -oE 'The input did not match the regular expression .*' "$H/wt_d2.check.log")"
