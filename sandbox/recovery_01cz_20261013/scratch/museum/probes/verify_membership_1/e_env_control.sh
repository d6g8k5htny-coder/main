#!/bin/sh
# E: cd32303 tests/test_museum_data.py::test_plain_fixture_directory_does_not_bypass_byte_verification writes a
# CHANGED STATUS.md into a tempfile.TemporaryDirectory and expects read_source to refuse it. read_source picks
# "Git object vs plain file" only by whether `git show` succeeds from cwd=<that dir>, so the control's outcome
# depends on where TMPDIR is. (i) TMPDIR outside any repository; (ii) TMPDIR inside a full-history checkout.
V=$(cd "$(dirname "$0")" && pwd); T=tests.test_museum_data.MuseumDataTests.test_plain_fixture_directory_does_not_bypass_byte_verification
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$V/outside_tmp" "$V/ck0/.tmp"
echo "(i) TMPDIR=$V/outside_tmp  (git rev-parse there: $(cd "$V/outside_tmp" && git rev-parse --show-toplevel 2>&1 | head -c 40))"
(cd "$V/ck0" && TMPDIR="$V/outside_tmp" python3 -B -m unittest -q $T 2>&1 | grep -E '^(OK|FAILED)|AssertionError')
echo "(ii) TMPDIR=$V/ck0/.tmp  (inside the full-history checkout)"
(cd "$V/ck0" && TMPDIR="$V/ck0/.tmp" python3 -B -m unittest -q $T 2>&1 | grep -E '^(OK|FAILED)|AssertionError')
rmdir "$V/ck0/.tmp" 2>/dev/null
