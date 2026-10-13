#!/bin/sh
# End-to-end: for each variant, copy the cd32303 tree, patch the COPY's museum_data.py, regenerate
# museum.json with the copy's own builder, then run the copy's own tools/museum_check.py (which
# rebuilds, compares the export, and runs tests/test_museum_frontend.mjs including the
# "actual pinned museum" test through startMuseum/validateBoundManifest). No network: proxies are
# pointed at a closed port and every source is read from local bytes that verify_bytes checks.
set -u
P=<SCRATCH>/museum/probes/verify_substitution_0
T=<SCRATCH>/museum/cd32303
SUB=<SCRATCH>/museum/probes/substitution
export PATH=/opt/node22/bin:$PATH HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 PYTHONDONTWRITEBYTECODE=1
rm -rf "$P/math" "$P/runs"; mkdir -p "$P/runs"
cp -r "$SUB/fixture/Math-" "$P/math"; cp "$SUB/readme_10e1f191.md" "$P/math/README.md"
for V in baseline A_two_scale_review_is_its_proof B_d2_review_is_its_proof C_d5open_review_is_index_no_pointer CONTROL_two_scale_review_is_other_claims_review; do
  R="$P/runs/$V"; cp -r "$T" "$R"; rm -rf "$R/tools/__pycache__" "$R/tests/__pycache__"
  python3 -B -I "$P/patch.py" "$R" "$V" "$P/math/README.md" >/dev/null || { echo "$V: PATCH FAILED"; continue; }
  if ! out=$(cd "$R" && python3 -B -E -s tools/museum_data.py --math-root "$P/math" 2>&1); then
    echo "$V: BUILDER REFUSED -> $(echo "$out" | tail -1)"; continue; fi
  if out=$(cd "$R" && python3 -B -E -s tools/museum_check.py --math-root "$P/math" 2>&1); then
    echo "$V: museum_check PASSED -> $(echo "$out" | grep -E '^# (pass|fail)' | tr '\n' ' ')| $(echo "$out" | tail -1)"
  else
    echo "$V: museum_check FAILED -> $(echo "$out" | grep -E '^# (pass|fail)|not ok' | tr '\n' ' ')"; fi
done
