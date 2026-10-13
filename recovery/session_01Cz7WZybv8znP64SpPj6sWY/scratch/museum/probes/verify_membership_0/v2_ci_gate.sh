#!/bin/sh
# V2: one-line "snapshot refresh" to a commit lacking the packet: change ONLY MAIN_COMMIT/main pins
# 71400b94 -> bd9fc9d6 in tools/museum_data.py, regenerate museum.json with the PR's own generator,
# then run the PR's CI gate commands from public-shop.yml (museum_check.py with --math-root because
# probes have no network; Math- pins are untouched). git is discovered from cwd (no GIT_DIR).
H=$(cd "$(dirname "$0")" && pwd); C=bd9fc9d603fd70946de364afcc3c902f09a5bf29; N="$H/wt_bd9"
unset GIT_DIR; export GIT_OPTIONAL_LOCKS=0 PATH=/opt/node22/bin:$PATH
rm -rf "$N"; cp -a "$H/wt_base" "$N"
sed -i "s/71400b94f6cb354a8cf7aba73ffede2138a64efa/$C/g" "$N/tools/museum_data.py"
echo "--- diff of tools/museum_data.py vs cd32303 (lines changed):"; diff "$H/wt_base/tools/museum_data.py" "$N/tools/museum_data.py" | grep -c '^>'
echo "--- incoming/ entries in tree of $C: $(git -C "$N" ls-tree -r --name-only $C incoming/ | wc -l); in 71400b94: $(git -C "$N" ls-tree -r --name-only 71400b94 incoming/ | wc -l)"
cd "$N" && python3 -B tools/museum_data.py --math-root "$H/fixture/Math-"; echo "generator exit=$?"
echo "--- museum.json changes vs cd32303 (unique changed tokens):"
diff <(tr ',' '\n' < "$H/wt_base/docs/site/museum.json") <(tr ',' '\n' < "$N/docs/site/museum.json") | grep '^[<>]' | sed -E 's/[0-9a-f]{40}/<SHA:&>/g' | grep -oE '^[<>]|<SHA:[0-9a-f]{40}>' | paste -sd' ' | tr ' ' '\n' | sort | uniq -c
python3 -I -c "import json,sys;d=json.load(open(sys.argv[1]));[print('packet',p['id'],'result.commit=',p['result']['commit'],'url=',p['result']['url']) for p in d['packets']]" "$N/docs/site/museum.json"
echo "--- python3 -B -m unittest discover -s tests -p test_museum_data.py"
python3 -B -m unittest discover -s tests -p test_museum_data.py 2>&1 | tail -3
echo "--- python3 -B tools/museum_check.py --math-root <fixture>"
python3 -B tools/museum_check.py --math-root "$H/fixture/Math-" > "$H/v2_check.log" 2>&1; rc=$?
grep -E '^# (tests|pass|fail|skipped)|^Museum|^not ok|actual pinned' "$H/v2_check.log"; echo "museum_check exit=$rc"
