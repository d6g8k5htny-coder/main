#!/bin/sh
# V2: a complete "reviewed engineering refresh" of the main snapshot 71400b94 -> a12c178c (the real main merge of
# #150, two commits after #151's merge d77a85a7). Replace the pin in EVERY file under tools/ tests/ docs/ that names it
# (generator, the unit-test literal, the docs), regenerate museum.json with the PR's own generator reading the main
# objects at the new commit, run the WHOLE museum step of .github/workflows/public-shop.yml, then compare the emitted
# packet list with the incoming/ tree of the commit it now declares, and render strictly at that commit.
H=$(cd "$(dirname "$0")" && pwd); CD="$H/../../../cd32303"; N="$H/wt_refresh"; C=a12c178c0f857a130cf434e9efd44233a038195b
OLD=71400b94f6cb354a8cf7aba73ffede2138a64efa
export PATH=/opt/node22/bin:$PATH GIT_OPTIONAL_LOCKS=0; unset GIT_DIR GIT_WORK_TREE
mkdir -p "$H/tmp"; export TMPDIR="$H/tmp"
rm -rf "$N"; cp -a "$CD" "$N"; find "$N" -name __pycache__ -prune -exec rm -rf {} +
cd "$N"
echo "--- files naming the old pin (all rewritten to $C):"; grep -rl $OLD tools tests docs | sort | tee "$H/v2_pinfiles.txt"
xargs sed -i "s/$OLD/$C/g" < "$H/v2_pinfiles.txt"
echo "--- remaining old-pin occurrences under tools/ tests/ docs/: $(grep -rl $OLD tools tests docs | wc -l)"
echo "--- main-repo pinned blobs at old vs new commit (git rev-parse <c>:<path>):"
python3 -I -B -c "
import importlib.util,subprocess,sys
s=importlib.util.spec_from_file_location('m',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
for p,i in m.SOURCES.items():
  if i['repository'].endswith('/main'):
    r=lambda c:subprocess.run(['git','--git-dir='+sys.argv[2],'rev-parse',c+':'+i['path']],capture_output=True,text=True).stdout.strip()
    print('   ',p,'same' if r(sys.argv[3])==r(sys.argv[4])==i['blob'] else 'DIFFERENT')" "$N/tools/museum_data.py" "$H/objstore.git" $OLD $C
echo "--- generator (main objects read with git show $C:<path>):"
GIT_DIR="$H/objstore.git" python3 -B -E -s tools/museum_data.py --main-root "$N" --math-root "$H/fixture/Math-"; echo "generator exit=$?"
echo "--- public-shop.yml step 'Museum source projection and refusal controls':"
python3 -B -m unittest discover -s tests -p test_museum_data.py > "$H/v2_unittest.log" 2>&1; rc=$?; tail -1 "$H/v2_unittest.log"; echo "  unittest exit=$rc"
GIT_DIR="$H/objstore.git" python3 -B tools/museum_check.py --math-root "$H/fixture/Math-" > "$H/v2_check.log" 2>&1; rc=$?
grep -E '^# (tests|pass|fail) |^Museum' "$H/v2_check.log" | sed 's/^/  /'; echo "  museum_check exit=$rc"
node --test tests/test_museum_geometry.mjs 2>&1 | grep -E '^# (tests|pass|fail) ' | sed 's/^/  /'
echo "--- membership at the declared commit:"
echo "  git ls-tree -d $C incoming/: $(GIT_DIR="$H/objstore.git" git ls-tree -d --name-only $C incoming/ | tr '\n' ' ')"
python3 -I -c "import json,sys;d=json.load(open(sys.argv[1]));print('  museum.json status_source.commit:',d['status_source']['commit'][:8],'| packets:',[p['id']+'@'+p['result']['commit'][:8] for p in d['packets']])" docs/site/museum.json
echo "--- strict raw-at-commit render of the refreshed manifest with the refreshed display code:"
node "$H/render.mjs" "$N" "$N/docs/site/museum.json"
