#!/bin/sh
# V4: read_source's "Git object vs plain file" choice depends only on whether `git show` succeeds in the
# repository git DISCOVERS from cwd. Run the PR's own plain-fixture negative control with TMPDIR (a) outside
# any repository, (b) inside the CI-like worktree (whose discovered .git holds 71400b94). No GIT_DIR.
H=$(cd "$(dirname "$0")" && pwd); unset GIT_DIR; export GIT_OPTIONAL_LOCKS=0
T=tests.test_museum_data.MuseumDataTests.test_plain_fixture_directory_does_not_bypass_byte_verification
cd "$H/wt_base"
echo "--- (a) TMPDIR unset (/tmp, no repository):"; python3 -B -m unittest -q $T 2>&1 | tail -1
mkdir -p "$H/wt_base/.probe_tmp"
echo "--- (b) TMPDIR inside the worktree:"; TMPDIR="$H/wt_base/.probe_tmp" python3 -B -m unittest -q $T 2>&1 | grep -E 'AssertionError|^FAILED|^OK'
rmdir "$H/wt_base/.probe_tmp"
