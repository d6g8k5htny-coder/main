#!/bin/sh
# V2b: a complete "reviewed engineering refresh" to bd9fc9d6: replace the main pin everywhere the PR
# names it (tools/museum_data.py, the test literal at tests/test_museum_data.py:91, docs/site/README.md,
# IMPLEMENTATION.md), regenerate museum.json, then run the WHOLE museum step of public-shop.yml.
H=$(cd "$(dirname "$0")" && pwd); C=bd9fc9d603fd70946de364afcc3c902f09a5bf29; N="$H/wt_bd9_full"
unset GIT_DIR; export GIT_OPTIONAL_LOCKS=0 PATH=/opt/node22/bin:$PATH
rm -rf "$N"; cp -a "$H/wt_base" "$N"; cd "$N"
grep -rl 71400b94f6cb354a8cf7aba73ffede2138a64efa tools tests docs | sort > ../pinfiles.txt; cat ../pinfiles.txt; xargs sed -i "s/71400b94f6cb354a8cf7aba73ffede2138a64efa/$C/g" < ../pinfiles.txt
python3 -B tools/museum_data.py --math-root "$H/fixture/Math-"
echo "--- step 'Museum source projection and refusal controls':"
python3 -B -m unittest discover -s tests -p test_museum_data.py 2>&1 | tail -1; echo "unittest exit=$?"
python3 -B tools/museum_check.py --math-root "$H/fixture/Math-" > "$H/v2b_check.log" 2>&1; rc=$?
grep -E '^# (tests|pass|fail)|^Museum' "$H/v2b_check.log"; echo "museum_check exit=$rc"
node --test tests/test_museum_geometry.mjs 2>&1 | grep -E '^# (tests|pass|fail)'
echo "--- membership of every main source path at $C (what raw.githubusercontent.com/<repo>/$C/<path> serves):"
python3 -I -c "
import json,subprocess,sys
d=json.load(open('docs/site/museum.json'))
seen=set()
def walk(o):
    if isinstance(o,dict):
        if o.get('repository')=='d6g8k5htny-coder/main' and 'path' in o and 'commit' in o:
            k=(o['commit'],o['path'])
            if k not in seen:
                seen.add(k); ok=subprocess.run(['git','cat-file','-e',o['commit']+':'+o['path']],capture_output=True).returncode==0
                print(('MEMBER    ' if ok else 'NOT-MEMBER'),o['commit'][:8],o['path'])
        for v in o.values(): walk(v)
    elif isinstance(o,list):
        for v in o: walk(v)
walk(d)"
