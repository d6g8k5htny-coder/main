#!/bin/sh
# P1c: read_source decides "Git object vs plain file" only by whether `git show` succeeds in whatever
# repository git discovers from cwd. The PR's own negative control for plain fixtures therefore depends
# on the environment: with a discoverable repository holding 71400b94 (GIT_DIR here; equally a TMPDIR
# inside a full clone) the changed plain fixture is silently replaced by object bytes and the control fails.
H=$(cd "$(dirname "$0")" && pwd); N="$H/wt_p1c"
rm -rf "$N"; cp -a "$H/../../cd32303" "$N"; rm -rf "$N/tools/__pycache__"; cd "$N"
T=test_museum_data.MuseumDataTests.test_plain_fixture_directory_does_not_bypass_byte_verification
echo "--- no discoverable repository:"; python3 -B -E -s -m unittest -q tests.test_museum_data.MuseumDataTests.test_plain_fixture_directory_does_not_bypass_byte_verification 2>&1 | tail -1
echo "--- GIT_DIR=$H/objstore.git (read-only):"; GIT_DIR=$H/objstore.git GIT_OPTIONAL_LOCKS=0 python3 -B -E -s -m unittest -q tests.test_museum_data.MuseumDataTests.test_plain_fixture_directory_does_not_bypass_byte_verification 2>&1 | grep -E 'AssertionError|^FAILED|^OK'
