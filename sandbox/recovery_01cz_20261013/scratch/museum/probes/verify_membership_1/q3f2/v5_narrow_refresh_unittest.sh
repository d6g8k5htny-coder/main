#!/bin/sh
# V5 calibration of the original probe: rewrite the pin ONLY in tools/museum_data.py (as membership/mkcopy.sh does),
# then run the unit-test half of the public-shop.yml museum step that the original probe did not run.
H=$(cd "$(dirname "$0")" && pwd); CD="$H/../../../cd32303"; N="$H/wt_narrow"; C=a12c178c0f857a130cf434e9efd44233a038195b
unset GIT_DIR GIT_WORK_TREE; mkdir -p "$H/tmp"; export TMPDIR="$H/tmp" GIT_OPTIONAL_LOCKS=0
rm -rf "$N"; cp -a "$CD" "$N"; find "$N" -name __pycache__ -prune -exec rm -rf {} +
sed -i "s/71400b94f6cb354a8cf7aba73ffede2138a64efa/$C/g" "$N/tools/museum_data.py"; cd "$N"
python3 -B -m unittest discover -s tests -p test_museum_data.py > "$H/v5_unittest.log" 2>&1; echo "unittest exit=$?"
grep -E '^(FAIL|ERROR):|AssertionError' "$H/v5_unittest.log" | cut -c1-200
