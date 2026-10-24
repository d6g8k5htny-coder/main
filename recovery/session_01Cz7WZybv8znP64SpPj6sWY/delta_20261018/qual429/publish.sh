#!/bin/bash
# QUAL429-CURMAIN-01CZ-R2-20261018 publication (run with DRY=1 for checks only)
set -euo pipefail
S=<SCRATCH>/q429
REF=refs/heads/claude/qual429-current-main-20261018
R=eae6aa3bd07e6db9178c23904a354e0d7cdfa0e8
P=experiments/universality/fold_isolation_diagonal
PKT=c634cf52d3e0a47717d8291b8dc08c37abe62fe8
REVIEWED_WF=9e7bacb8f04a1abba1fef0a77a7e0e666b064c76
cd /home/user/main
git fetch -q origin main
MAIN=$(git rev-parse origin/main)
echo "main=$MAIN"
# 1. ref must be absent
if [ -n "$(git ls-remote origin "$REF")" ]; then echo "STOP: $REF exists"; exit 2; fi
# 2. original #429 head unchanged
H=$(gh api repos/d6g8k5htny-coder/main/pulls/429 --jq .head.sha)
[ "$H" = "$R" ] || { echo "STOP: #429 head moved to $H"; exit 2; }
# 3. reviewed workflows unchanged (additions only)
WD=$(git diff --name-status "$REVIEWED_WF" "$MAIN" -- .github/workflows | grep -v '^A' || true)
[ -z "$WD" ] || { echo "STOP: reviewed workflow changed/removed:"; echo "$WD"; exit 2; }
echo "reviewed workflows unchanged; added: $(git diff --name-status "$REVIEWED_WF" "$MAIN" -- .github/workflows | grep -c '^A' || true)"
# 4. compose: main tree + exactly the R packet subtree
export GIT_INDEX_FILE=$S/index_pub
git read-tree "$MAIN"
if git ls-files --error-unmatch "$P" >/dev/null 2>&1; then echo "STOP: packet path exists on main"; exit 2; fi
git read-tree --prefix="$P/" "$R:$P"
T=$(git write-tree); unset GIT_INDEX_FILE
[ "$(git rev-parse "$T:$P")" = "$PKT" ] || { echo "STOP: packet subtree mismatch"; exit 2; }
NS=$(git diff --name-status "$MAIN" "$T")
NA=$(printf '%s\n' "$NS" | grep -c '^A' || true); NO=$(printf '%s\n' "$NS" | grep -vc "^A	$P/" || true)
[ "$NA" = 83 ] && [ "$NO" = 0 ] || { echo "STOP: diff not exactly 83 A under packet (A=$NA other=$NO)"; exit 2; }
echo "composed tree $T (83 A, packet $PKT)"
TRIG=$(python3 -I $S/triggers.py "$MAIN" "$T")
EXP="public-intake.yml:pull_request_target public-shop.yml:pull_request universality-controls.yml:pull_request workspace-landing.yml:pull_request"
[ "$TRIG" = "$EXP" ] || { echo "STOP: triggered set differs: $TRIG"; exit 2; }
echo "triggered set = reviewed 4 workflows"
# 5. contenders: no other open PR touches the packet path
for n in $(gh api "repos/d6g8k5htny-coder/main/pulls?state=open&per_page=100" --jq '.[].number'); do
  [ "$n" = 429 ] && continue
  cf=$(gh api repos/d6g8k5htny-coder/main/pulls/$n --jq .changed_files); pages=$(( (cf+99)/100 )); hit=0
  for ((p=1;p<=pages;p++)); do c=$(gh api "repos/d6g8k5htny-coder/main/pulls/$n/files?per_page=100&page=$p" --jq "[.[].filename|select(startswith(\"$P/\"))]|length"); hit=$((hit+c)); done
  [ "$hit" = 0 ] || { echo "STOP: PR #$n touches packet path ($hit files)"; exit 2; }
done
echo "contenders: none on packet path"
if [ "${DRY:-0}" = 1 ]; then echo "DRY RUN OK"; exit 0; fi
# 6. signed commit, plain push, readback
C=$(git commit-tree -S "$T" -p "$MAIN" -F $S/commitmsg2.txt)
echo "commit $C"
git cat-file commit "$C" | sed '/^$/q' | grep -q '^gpgsig' || { echo "STOP: unsigned"; exit 2; }
git push origin "$C:$REF"
RB=$(git ls-remote origin "$REF" | cut -f1)
[ "$RB" = "$C" ] || { echo "STOP: readback mismatch $RB"; exit 2; }
echo "PUSHED $REF = $C (parent $MAIN, tree $T)"
echo "$C $MAIN $T" > $S/published.txt
