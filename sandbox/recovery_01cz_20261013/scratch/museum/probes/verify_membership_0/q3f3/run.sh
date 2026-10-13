#!/bin/sh
# Verifier 1 / Q3-F3: an unlanded (NEVER-MERGE) main pin at the cd32303 bytes.
# Variant A: change only the pin in tools/museum_data.py (what the original probe did).
# Variant B: a complete refresh, which also updates the hard-coded literal in tests/test_museum_data.py:91.
# Each variant runs the full CI step "Museum source projection and refusal controls" from
# .github/workflows/public-shop.yml@cd32303 plus a strict render. Read-only object store
# (alternates -> /home/user/main/.git/objects), no network, no changes to the repository.
set -u
H=$(cd "$(dirname "$0")" && pwd); CD="$H/../../../cd32303"; MATH="$H/../../membership/fixture/Math-"
OLD=71400b94f6cb354a8cf7aba73ffede2138a64efa; NEW=3b8c45452c01f669a7245c472acdb8cc4af5b2f6
export PATH=/opt/node22/bin:$PATH GIT_OPTIONAL_LOCKS=0
if [ ! -d "$H/objs.git" ]; then git init -q --bare "$H/objs.git"; echo /home/user/main/.git/objects > "$H/objs.git/objects/info/alternates"; fi
export GIT_DIR="$H/objs.git"
echo "== facts"
git log -1 --format='  %h %s' $NEW
git merge-base --is-ancestor $NEW 513261bb && echo "  3b8c4545 ancestor of main 513261bb: yes" || echo "  3b8c4545 ancestor of main 513261bb: NO"
git merge-base --is-ancestor $NEW d77a85a7 && echo "  3b8c4545 ancestor of #151 merge d77a85a7: yes" || echo "  3b8c4545 ancestor of #151 merge d77a85a7: NO"
git branch -r --contains $NEW 2>/dev/null | sed 's/^ */  contained only in: /' 
git merge-base --is-ancestor $OLD d77a85a7 && echo "  control: real pin 71400b94 ancestor of #151 merge d77a85a7: yes"
for p in STATUS.md incoming/side24-identity-replay-20260926/RESULT.md; do
  [ "$(git rev-parse $NEW:$p)" = "$(git rev-parse $OLD:$p)" ] && echo "  $p identical at both commits (git show from object store, not working tree)"; done
for V in A B; do
  W="$H/wt_$V"; rm -rf "$W"; cp -a "$CD" "$W"; find "$W" -name __pycache__ -prune -exec rm -rf {} +
  sed -i "s/$OLD/$NEW/g" "$W/tools/museum_data.py"
  [ $V = B ] && sed -i "s/$OLD/$NEW/g" "$W/tests/test_museum_data.py"
  python3 -B -E -s "$W/tools/museum_data.py" --main-root "$W" --math-root "$MATH" >/dev/null || { echo "generator failed"; continue; }
  echo "== variant $V (pin changed in: tools/museum_data.py$([ $V = B ] && echo ', tests/test_museum_data.py'))"
  cd "$W"
  python3 -B -m unittest discover -s tests -p test_museum_data.py > "$H/unit_$V.txt" 2>&1; echo "  unittest test_museum_data.py exit=$? :: $(tail -1 "$H/unit_$V.txt")"
  grep -E "^FAIL:|AssertionError" "$H/unit_$V.txt" | sed 's/^/    /' | head -3
  python3 -B tools/museum_check.py --math-root "$MATH" > "$H/check_$V.txt" 2>&1; echo "  museum_check exit=$? :: $(tail -1 "$H/check_$V.txt")"
  node --test tests/test_museum_geometry.mjs > "$H/geom_$V.txt" 2>&1; echo "  geometry exit=$?"
  grep -o '"commit": "[0-9a-f]*"' docs/site/museum.json | sort | uniq -c | sed 's/^/  museum.json /'
  cd "$H"; node "$H/strict.mjs" "$W/docs/site/museum.json" "$CD" "$H/objs.git" | sed 's/^/  /'
done
echo "== ancestry/default-branch logic anywhere in museum code, tests or workflows at cd32303:"
grep -rcE "merge-base|is-ancestor|rev-list|first-parent|refs/heads|origin/main|--contains" "$CD/tools/museum_data.py" "$CD/tools/museum_check.py" "$CD/docs/site/museum.mjs" "$CD/docs/site/core.mjs" "$CD/tests/test_museum_data.py" "$CD/tests/test_museum_frontend.mjs" "$CD/.github/workflows/public-shop.yml" | sed "s#$CD/#  #"
