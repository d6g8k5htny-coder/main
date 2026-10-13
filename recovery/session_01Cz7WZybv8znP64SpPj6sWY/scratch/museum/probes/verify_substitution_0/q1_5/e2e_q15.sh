#!/bin/sh
# Verifier probe Q1-5, end to end. For each variant: copy the cd32303 tree, patch the COPY's builder
# (patch_q15.py), regenerate museum.json with the copy's own builder, then run the copy's own
# tools/museum_check.py (rebuild + check_export + node --test tests/test_museum_frontend.mjs, including
# the "actual pinned museum renders all cards and rejects cross-claim ... replay substitutions" test).
# No network: proxies point at a closed port; every source is read from local bytes that verify_bytes checks.
set -u
P=<SCRATCH>/museum/probes/verify_substitution_0/q1_5
T=<SCRATCH>/museum/cd32303
export PATH=/opt/node22/bin:$PATH HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 PYTHONDONTWRITEBYTECODE=1
rm -rf "$P/math" "$P/runs"; mkdir -p "$P/runs"
cp -r "$P/fixture/Math-" "$P/math"; cp "$P/readme_10e1f191.md" "$P/math/README.md"
for V in baseline SWAP_D2_D6__D4_gets_D0_hard_gate D2_D4_D6_all_get_full_price CONTROL_D2_invented_command; do
  R="$P/runs/$V"; cp -r "$T" "$R"; rm -rf "$R/tools/__pycache__" "$R/tests/__pycache__"
  python3 -B -I "$P/patch_q15.py" "$R" "$V" "$P/math/README.md" >/dev/null || { echo "$V: PATCH FAILED"; continue; }
  if ! out=$(cd "$R" && python3 -B -E -s tools/museum_data.py --math-root "$P/math" 2>&1); then
    echo "$V: BUILDER REFUSED -> $(echo "$out" | tail -1)"; continue; fi
  cmds=$(python3 -I "$P/show_cmds.py" "$R/docs/site/museum.json")
  echo "$V: built museum.json replay commands: $cmds"
  if out=$(cd "$R" && python3 -B -E -s tools/museum_check.py --math-root "$P/math" 2>&1); then
    echo "  museum_check PASSED -> $(echo "$out" | grep -E '^# (pass|fail)' | tr '\n' ' ')| $(echo "$out" | tail -1)"
  else
    echo "  museum_check FAILED -> $(echo "$out" | grep -E '^# (pass|fail)|^not ok' | tr '\n' ' ')"; fi
done
