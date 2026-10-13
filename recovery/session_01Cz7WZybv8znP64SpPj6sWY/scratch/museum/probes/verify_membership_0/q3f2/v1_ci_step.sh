#!/bin/sh
# v1_ci_step.sh <name>: in ./<name>, regenerate museum.json with the PR's own generator, then run the exact
# commands of public-shop.yml "Museum source projection and refusal controls" (museum_check.py gets
# --math-root because probes have no network; every Math- byte is still verified against its pin).
H=$(cd "$(dirname "$0")" && pwd); N="$H/$1"; unset GIT_DIR; export GIT_OPTIONAL_LOCKS=0 PATH=/opt/node22/bin:$PATH
cd "$N"
python3 -B tools/museum_data.py --math-root "$H/fixture/Math-"; echo "  generator exit=$?"
python3 -I -c "import json,sys;d=json.load(open(sys.argv[1]));print('  museum.json status_source.commit =',d['status_source']['commit'][:8],'| packets =',[p['id']+'@'+p['result']['commit'][:8] for p in d['packets']])" docs/site/museum.json
C=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['status_source']['commit'])" docs/site/museum.json)
echo "  git ls-tree -d $(echo $C|cut -c1-8) incoming/ = $(git ls-tree -d --name-only $C incoming/ | paste -sd' ')"
python3 -B -m unittest discover -s tests -p test_museum_data.py > "$H/$1.unittest.log" 2>&1; echo "  [step 1] unittest test_museum_data.py exit=$? :: $(tail -1 "$H/$1.unittest.log")"
grep -E '^(FAIL|ERROR):|AssertionError' "$H/$1.unittest.log" | sed 's/^/      /'
python3 -B tools/museum_check.py --math-root "$H/fixture/Math-" > "$H/$1.check.log" 2>&1; echo "  [step 2] museum_check.py exit=$? :: $(grep -E '^# (pass|fail) ' "$H/$1.check.log" | paste -sd' ') :: $(grep '^Museum' "$H/$1.check.log")"
node --test tests/test_museum_geometry.mjs > "$H/$1.geom.log" 2>&1; echo "  [step 3] node --test test_museum_geometry.mjs exit=$? :: $(grep -E '^# (pass|fail) ' "$H/$1.geom.log" | paste -sd' ')"
