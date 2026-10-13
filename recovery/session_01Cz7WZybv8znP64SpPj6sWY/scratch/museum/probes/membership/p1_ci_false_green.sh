#!/bin/sh
# P1 (CI): with the main pin moved to bd9fc9d6 -- a real main commit whose tree has STATUS.md, the SARD
# review and the notebook byte-identical to the pins but NO incoming/ directory -- the PR's own CI gate
# tools/museum_check.py (incl. its node "actual pinned museum renders all cards" test) exits 0 and the
# generated museum.json lists the packet "at" bd9fc9d6.
H=$(cd "$(dirname "$0")" && pwd); C=bd9fc9d603fd70946de364afcc3c902f09a5bf29
"$H/mkcopy.sh" wt_p1 $C
export GIT_DIR=$H/objstore.git GIT_OPTIONAL_LOCKS=0 PATH=/opt/node22/bin:$PATH
echo "--- git ls-tree $C incoming/ (empty = no packet at the declared snapshot):"
git ls-tree -r --name-only $C incoming/; echo "[end ls-tree]"
echo "--- generated museum.json packets:"
python3 -I -c "import json,sys;d=json.load(open(sys.argv[1]));[print(p['id'],'result.commit=',p['result']['commit'],'status_source.commit=',d['status_source']['commit']) for p in d['packets']]" "$H/wt_p1/docs/site/museum.json"
echo "--- PR CI gate: python3 -B tools/museum_check.py --math-root fixture/Math-"
cd "$H/wt_p1" && python3 -B -E -s tools/museum_check.py --math-root "$H/fixture/Math-" > "$H/out_p1_check.txt" 2>&1; rc=$?
grep -E '^# (tests|pass|fail|skipped)|^Museum|not ok|Error' "$H/out_p1_check.txt"; echo "museum_check exit=$rc"
echo "--- PR unit test that claims to read the landed packet from the pinned checkout:"
python3 -B -E -s -m unittest discover -s tests -p test_museum_data.py -k test_landed_packet_can_be_read_from_pinned_sparse_checkout 2>&1 | tail -3
