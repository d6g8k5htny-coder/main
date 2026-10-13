#!/bin/sh
# P2: packet membership is a hard-coded list, never enumerated from the declared commit.
# (a) Snapshot refresh to a12c178c -- the real main merge of #150, whose tree holds TWO packets and whose
#     STATUS.md / SARD review / notebook are byte-identical to the pins. The PR's generator, its CI gate
#     and a strict raw-GitHub render all pass while listing ONE packet.
# (b) The cd32303 browser validator accepts packets:[] for the real 71400b94 manifest and prints
#     "No packet appears in this pinned projection." although the snapshot tree contains a packet.
H=$(cd "$(dirname "$0")" && pwd); C=a12c178c0f857a130cf434e9efd44233a038195b
"$H/mkcopy.sh" wt_p2 $C
export GIT_DIR=$H/objstore.git GIT_OPTIONAL_LOCKS=0 PATH=/opt/node22/bin:$PATH
echo "--- (a) git ls-tree -d $C incoming/:"; git ls-tree -d --name-only $C incoming/
cd "$H/wt_p2" && python3 -B -E -s tools/museum_check.py --math-root "$H/fixture/Math-" > "$H/out_p2_check.txt" 2>&1; rc=$?
grep -E '^# (pass|fail)|^Museum' "$H/out_p2_check.txt"; echo "museum_check exit=$rc"
cd "$H"; node render.mjs wt_p2/docs/site/museum.json strict | grep -v 'packet-cards'
echo "--- (b) real cd32303 manifest with packets:[] (git ls-tree -d 71400b94 incoming/ -> $(git ls-tree -d --name-only 71400b94 incoming/))"
python3 -I -c "import json,sys;d=json.load(open(sys.argv[1]));d['packets']=[];open(sys.argv[2],'w').write(json.dumps(d))" <SCRATCH>/museum/cd32303/docs/site/museum.json "$H/p2b_empty_packets.json"
node render.mjs "$H/p2b_empty_packets.json" strict | grep -v 'packet requests'
echo "--- membership code paths in the three files (no ls-tree / incoming enumeration / MAIN_COMMIT use):"
grep -nE "ls-tree|iterdir|glob|listdir|incoming/|MAIN_COMMIT" <SCRATCH>/museum/cd32303/tools/museum_data.py <SCRATCH>/museum/cd32303/tools/museum_check.py <SCRATCH>/museum/cd32303/docs/site/museum.mjs
