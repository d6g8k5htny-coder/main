#!/usr/bin/env bash
# Re-runs every check in this folder from git objects only and prints the results.
# Usage: reproduce.sh MAIN_GIT MATH_GIT OUT_DIR   (relative paths are fine)
#   MAIN_GIT  a full clone of d6g8k5htny-coder/main (plain `git clone`) that has commits
#             6fce9b8c, 3edacfe5 and the hardening head a01c72f1. Shallow clones lack
#             the old commits; --filter=blob:none clones work but fetch each missing
#             blob one at a time during `git grep`, which can take about an hour.
#   MATH_GIT  a clone of d6g8k5htny-coder/Math- that has commits 4e722922 and b6fc0e64.
#   OUT_DIR   a new, empty directory for inputs and outputs.
set -euo pipefail
mkdir -p "$3"
MAIN_GIT=$(cd "$1" && pwd); MATH_GIT=$(cd "$2" && pwd); OUT=$(cd "$3" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
H=a01c72f1937875b94bfd78e183a6dbbdb2b04da4
B=drive/deltas/2026-09-19/DG-MIGRATION-20260919
PACK=55e00c9634b5771d95075e67445d6f5850ab215e2fdca38c7e5a0d834c6d14be.zip
# The scans ignore git grep's exit status, so stop here if any commit is missing.
for c in 6fce9b8c 3edacfe5 "$H"; do git -C "$MAIN_GIT" cat-file -e "$c^{commit}"; done
for c in 4e722922 b6fc0e64; do git -C "$MATH_GIT" cat-file -e "$c^{commit}"; done
# The scan rows store git grep's path text; pin git's default quoting of non-ASCII paths,
# appending to (not replacing) any GIT_CONFIG_KEY_n/VALUE_n entries the caller already has.
n=${GIT_CONFIG_COUNT:-0}
export "GIT_CONFIG_KEY_$n=core.quotePath" "GIT_CONFIG_VALUE_$n=true" GIT_CONFIG_COUNT=$((n + 1))
mkdir -p "$OUT/g90" "$OUT/rung" "$OUT/h3scan"
cd "$OUT"

git -C "$MAIN_GIT" show "$H:$B/objects.json" > g90/objects.json
git -C "$MAIN_GIT" show "$H:$B/packs/$PACK" > g90/pack_55e00c96.zip
cp g90/objects.json rung/objects.json
sha256sum g90/objects.json g90/pack_55e00c96.zip

echo "== g90/check.py"; python3 -I "$HERE/g90/check.py" "$OUT/g90"
echo "== g90/idx.py";   python3 -I "$HERE/g90/idx.py" "$OUT/g90"
echo "== rung/rebuild.py"; python3 -I "$HERE/rung/rebuild.py" "$OUT/rung" "$MAIN_GIT" "$H"

echo "== h3scan/scan.py (inventory 6243792389 cuts)"
python3 -I "$HERE/h3scan/scan.py" "$MAIN_GIT" 6fce9b8c "$MATH_GIT" 4e722922 "$MAIN_GIT" a01c72f1 > h3scan/scan1.json
python3 -I "$HERE/h3scan/summarize.py" h3scan/scan1.json
echo "== h3scan/scan2.py (addendum 6245774604 cuts)"
python3 -I "$HERE/h3scan/scan2.py" "$MAIN_GIT" 3edacfe5 "$MATH_GIT" b6fc0e64 "$MAIN_GIT" a01c72f1 > h3scan/scan2.json
python3 -I "$HERE/h3scan/summarize.py" h3scan/scan2.json
echo "== h3scan/filter2.py on scan2 output"
python3 -I "$HERE/h3scan/filter2.py" h3scan/scan2.json > h3scan/filtered2.json
python3 -I "$HERE/h3scan/summarize.py" h3scan/filtered2.json
