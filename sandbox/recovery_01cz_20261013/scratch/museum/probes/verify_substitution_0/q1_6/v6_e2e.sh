#!/bin/sh
# Verifier probe for Q1-6, end to end through the copy's OWN builder, museum_check (which runs
# tests/test_museum_frontend.mjs incl. the "actual pinned museum" test) and the CI unittest step
# (public-shop.yml:25). Two variants: REPIN_ONLY (control: same STATUS bytes, new main commit) and
# SUCCESSOR (D5 pin-neighborhood row replaced by a different open obligation, re-pinned).
# No network: proxies point at a closed port; every source is read from local bytes that verify_bytes checks.
set -u
Q=<SCRATCH>/museum/probes/verify_substitution_0/q1_6
T=<SCRATCH>/museum/cd32303
export PATH=/opt/node22/bin:$PATH HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 PYTHONDONTWRITEBYTECODE=1
rm -rf "$Q/math" "$Q/runs"; mkdir -p "$Q/runs"
cp -r "$Q/fixture/Math-" "$Q/math"; cp "$Q/readme_10e1f191.md" "$Q/math/README.md"
for V in REPIN_ONLY SUCCESSOR; do
  R="$Q/runs/$V"; cp -r "$T" "$R"; rm -rf "$R/tools/__pycache__" "$R/tests/__pycache__"
  python3 -B -I "$Q/v6_patch.py" "$R" "$V" "$Q/math/README.md" "$Q/p6_STATUS_successor.md" >/dev/null || { echo "$V: PATCH FAILED"; continue; }
  if ! out=$(cd "$R" && python3 -B -E -s tools/museum_data.py --math-root "$Q/math" 2>&1); then
    echo "$V: BUILDER REFUSED -> $(echo "$out" | tail -1)"; continue; fi
  echo "$V: builder -> $out"
  if out=$(cd "$R" && python3 -B -E -s tools/museum_check.py --math-root "$Q/math" 2>&1); then
    echo "$V: museum_check PASSED -> $(echo "$out" | grep -E '^# (pass|fail)' | tr '\n' ' ')| $(echo "$out" | tail -1)"
  else
    echo "$V: museum_check FAILED -> $(echo "$out" | grep -E '^# (pass|fail)|not ok' | tr '\n' ' ')"; fi
  python3 -B -E -s -c "
import json;m=json.load(open('$R/docs/site/museum.json'));c=m['claims'][12]
print('$V: claims[12]', c['id'], '|', c['title'], '|', c['proof'].get('availability'), '|', c['review']['review_url'][-60:])"
  out=$(cd "$R" && python3 -B -E -s -m unittest discover -s tests -p test_museum_data.py 2>&1)
  echo "$V: unittest test_museum_data.py -> $(echo "$out" | tail -1) ; failing: $(echo "$out" | grep -E '^(FAIL|ERROR):' | sed 's/ (.*//' | tr '\n' ' ')"
  echo "$out" | grep -E "AssertionError" | sed "s/^/    $V /" | cut -c1-200
done
