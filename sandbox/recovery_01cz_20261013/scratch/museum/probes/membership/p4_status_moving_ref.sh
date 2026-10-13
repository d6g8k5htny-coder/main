#!/bin/sh
# P4: the PR's unit tests read live working-tree STATUS.md (a moving ref) and require it to equal the
# 71400b94 snapshot bytes. A later, unrelated STATUS.md edit on main (snapshot unchanged) breaks the
# projection tests -- including the one named for immunity to working-tree appends -- while the CI gate
# (git show 71400b94:STATUS.md) still passes. Fail-closed robustness defect.
# (Current main 513261bb replaced the read with tests/fixtures/status_f2e432e.md.)
H=$(cd "$(dirname "$0")" && pwd); N="$H/wt_p4"
rm -rf "$N"; cp -a "$H/../../cd32303" "$N"; rm -rf "$N/tools/__pycache__"
cd "$N"
echo "--- control, unmodified STATUS.md:"
python3 -B -E -s -m unittest discover -s tests -p test_museum_data.py 2>&1 | tail -1
printf '\nUnrelated later status note.\n' >> "$N/STATUS.md"
echo "--- after appending one unrelated line to working-tree STATUS.md:"
python3 -B -E -s -m unittest discover -s tests -p test_museum_data.py > "$H/out_p4_unit.txt" 2>&1; echo "unit exit=$?"
grep -E '^(ERROR|FAIL):|^FAILED|identity mismatch: STATUS|does not match' "$H/out_p4_unit.txt" | sed 's/ (test_museum_data[^)]*)//' | sort | uniq -c | sort -rn
echo "--- CI gate on the same tree (full-clone objects via GIT_DIR):"
GIT_DIR=$H/objstore.git GIT_OPTIONAL_LOCKS=0 PATH=/opt/node22/bin:$PATH python3 -B -E -s tools/museum_check.py --math-root "$H/fixture/Math-" > "$H/out_p4_check.txt" 2>&1; echo "museum_check exit=$?"; grep '^Museum' "$H/out_p4_check.txt"
