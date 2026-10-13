#!/usr/bin/env bash
# Commands run on 2026-10-13 for main#392 (C160-PATH-01, head 322c8291); results are in review 5509417996.
# Usage: reproduce_and_mutate.sh <checkout-of-main-or-322c8291>. Needs Python 3.11 and git.
set -euo pipefail
W="${1:?checkout path}"; B="$(mktemp -d)"; mkdir "$B/real"; ln -s "$B/real" "$B/alias"
cd "$W"
for root in real alias; do for mode in "-B -S" "-B -O -S"; do
  echo "root=$root mode=$mode"; TMPDIR="$B/$root" python3 $mode -m unittest tests.test_site_asset_release 2>&1 | grep -E '^(FAIL|ERROR):|^Ran|^OK|^FAILED' || true
done; done
# Production mutant P1: inventory() drops site.resolve(); run with a canonical TMPDIR (what Linux CI sees).
sed -i '0,/^    site = site.resolve()$/s//    site = site/' tools/site_asset_release.py
TMPDIR="$B/real" python3 -B -S -m unittest tests.test_site_asset_release 2>&1 | grep -E '^(FAIL|ERROR):|^OK|^FAILED' || true
git checkout -- tools/site_asset_release.py
