#!/bin/sh
# C: the whole public-shop.yml "Museum source projection and refusal controls" step (run under bash -e,
# GitHub's default shell): unittest test_museum_data.py; tools/museum_check.py; node --test geometry.
# (3) pin -> bd9fc9d6 (packet NOT in tree) AND the one hard-coded pin literal in
#     tests/test_museum_data.py:91 refreshed, as any snapshot refresh must do.
# (4) pin -> e495837b (packet IS in tree, all six main blobs identical), literal NOT refreshed:
#     shows the literal assertion is a pin-change detector, not a membership check.
V=$(cd "$(dirname "$0")" && pwd); P=71400b94f6cb354a8cf7aba73ffede2138a64efa
export PATH=/opt/node22/bin:$PATH PYTHONDONTWRITEBYTECODE=1
step() { (cd "$1" && bash -e -c '
  python3 -B -m unittest discover -s tests -p test_museum_data.py 2>&1 | tail -1
  test "${PIPESTATUS:-0}" = 0
  python3 -B -m unittest discover -s tests -p test_museum_data.py >/dev/null 2>&1
  python3 -B tools/museum_check.py --math-root "$0" 2>&1 | grep -E "^# (pass|fail)|^Museum source"
  node --test tests/test_museum_geometry.mjs 2>&1 | grep -E "^# (pass|fail)"' "$V/mathfix"; echo "CI step exit=$?"); }
case_() { name=$1 c=$2 lit=$3; echo "=== $name: pin -> ${c%????????????????????????????????} member=$(git -C /home/user/main cat-file -e $c:incoming/side24-identity-replay-20260926/RESULT.md 2>/dev/null && echo yes || echo no) literal-refreshed=$lit"
  "$V/setup.sh" mkrepo "$V/$name" >/dev/null
  sed -i "s/$P/$c/g" "$V/$name/tools/museum_data.py"
  [ "$lit" = yes ] && sed -i "s/$P/$c/" "$V/$name/tests/test_museum_data.py"
  (cd "$V/$name" && python3 -B tools/museum_data.py --main-root . --math-root "$V/mathfix" >/dev/null)
  echo "changed vs cd32303: $(git -C "$V/$name" diff --stat | tail -1)"; step "$V/$name"; }
case_ ck3 bd9fc9d603fd70946de364afcc3c902f09a5bf29 yes
case_ ck4 e495837bcd2d6d5014508e9991c5b3089977ce20 no
