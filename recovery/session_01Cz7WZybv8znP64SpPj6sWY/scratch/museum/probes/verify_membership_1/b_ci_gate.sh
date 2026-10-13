#!/bin/sh
# B: run the PR's own CI museum steps (public-shop.yml lines 25-26: unittest test_museum_data.py and
# tools/museum_check.py) in a full-history checkout of cd32303 where ONLY the main pin is changed
# 71400b94 -> bd9fc9d6 and museum.json is regenerated with the PR's generator. Math- bytes come from the
# verified plain fixture (--math-root) so no network is used; the main-repository path is the CI path.
# Controls: (0) unchanged pin passes; (2) moved pin + working-tree packet file removed fails.
V=$(cd "$(dirname "$0")" && pwd); C=bd9fc9d603fd70946de364afcc3c902f09a5bf29
export PATH=/opt/node22/bin:$PATH PYTHONDONTWRITEBYTECODE=1
run_gate() { (cd "$1" && python3 -B tools/museum_check.py --math-root "$V/mathfix" >"$1.check.txt" 2>&1; echo "museum_check exit=$?"); \
  grep -E '^# (tests|pass|fail)|^Museum source|^(FileNotFound|Value)Error' "$1.check.txt"; }
echo "=== (0) control: unchanged pin 71400b94"; "$V/setup.sh" mkrepo "$V/ck0" >/dev/null; run_gate "$V/ck0"
echo "=== (1) main pin moved to bd9fc9d6"; "$V/setup.sh" mkrepo "$V/ck1" >/dev/null
sed -i "s/71400b94f6cb354a8cf7aba73ffede2138a64efa/$C/g" "$V/ck1/tools/museum_data.py"
echo "git ls-tree -r $C -- incoming/ (lines): $(git -C "$V/ck1" ls-tree -r --name-only $C -- incoming/ | wc -l)"
echo "git ls-tree -r 71400b94 -- incoming/side24-identity-replay-20260926/ (lines): $(git -C "$V/ck1" ls-tree -r --name-only 71400b94 -- incoming/side24-identity-replay-20260926/ | wc -l)"
(cd "$V/ck1" && python3 -B tools/museum_data.py --main-root . --math-root "$V/mathfix")
python3 -I -c "import json,sys;d=json.load(open(sys.argv[1]));p=d['packets'][0];print('museum.json packet',p['id'],'result.commit',p['result']['commit'],'url',p['result']['url'])" "$V/ck1/docs/site/museum.json"
echo "tracked files changed vs cd32303: $(git -C "$V/ck1" diff --name-only | tr '\n' ' ')"
run_gate "$V/ck1"
echo "--- CI step: python3 -B -m unittest discover -s tests -p test_museum_data.py"
(cd "$V/ck1" && python3 -B -m unittest discover -s tests -p test_museum_data.py 2>&1 | tail -3)
echo "=== (2) negative control: moved pin, working-tree RESULT.md removed"
rm "$V/ck1/incoming/side24-identity-replay-20260926/RESULT.md"; run_gate "$V/ck1"
git -C "$V/ck1" checkout -q -- incoming/side24-identity-replay-20260926/RESULT.md
