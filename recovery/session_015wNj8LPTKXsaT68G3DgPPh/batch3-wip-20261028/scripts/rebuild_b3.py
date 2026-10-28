#!/usr/bin/env python3
"""Rebuild the batch-3 work tree from the session transcript's own records.

Usage: rebuild_b3.py CALLS_JSON WORKTREE SCRIPTDIR

WORKTREE is a fresh checkout whose batch-2 files equal 6fbdbba3's. In transcript
order this:
  1. applies the printed `git diff` of the two workflows (call 0), of the helper
     (call 3) and of the test file (call 57) with `git apply`;
  2. replays every later Edit of the helper and the test file, each old string
     required to occur exactly once, and the custody-digest fill of call 91;
  3. rewrites SCOPE.md from its Write (call 71), then replays its python fills
     (calls 73 and 104) and its Edit (call 76);
  4. rebuilds the contract merge script from its printed text (call 15) and
     replays its python edits (calls 40 and 41) into SCRIPTDIR.
Nothing else in the transcript is executed. Every step fails loudly on a mismatch.
"""
import json
import pathlib
import re
import subprocess
import sys

OLD_ROOT = '/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/C13/c8wt3/'
TAIL = 'Shell cwd was reset to /home/user'


def strip_tail(text):
    text = text.rstrip('\n')
    if text.endswith(TAIL):
        text = text[:-len(TAIL)]
    return text.rstrip('\n') + '\n'


def diff_part(result):
    """The unified diff in a tool result, from its first 'diff --git' line."""
    start = result.index('diff --git')
    return strip_tail(result[start:])


def heredoc(command, marker="<<'PY'"):
    start = command.index(marker) + len(marker)
    body = command[start:].lstrip('\n')
    end = body.index('\nPY')
    return body[:end] + '\n'


def main():
    calls = {c['n']: c for c in json.loads(pathlib.Path(sys.argv[1]).read_text())}
    tree, scripts = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    for n in (0, 3, 57):
        patch = diff_part(calls[n]['result'])
        path = scripts / ('call%d.patch' % n)
        path.write_text(patch)
        subprocess.run(['git', 'apply', '--check', str(path)], cwd=tree, check=True)
        subprocess.run(['git', 'apply', str(path)], cwd=tree, check=True)
        print('applied diff of call', n)
    # The helper diff (call 3) predates the helper Edits 26-31 and 33. The test
    # diff (call 57) was printed after the test Edits 34-39 and 48, so it holds them.
    for n in (26, 27, 28, 29, 30, 31, 33):
        c = calls[n]
        assert c['name'] == 'Edit' and c['input']['file_path'] == OLD_ROOT + 'tools/check_mode_replay.py', n
        target = tree / 'tools/check_mode_replay.py'
        text = target.read_text()
        old, new = c['input']['old_string'], c['input']['new_string']
        assert text.count(old) == 1, (n, 'old string occurs %d times' % text.count(old))
        target.write_text(text.replace(old, new))
        print('replayed Edit', n, 'tools/check_mode_replay.py')
    for n in (34, 35, 36, 37, 38, 39, 48):
        c = calls[n]
        assert c['name'] == 'Edit' and c['input']['file_path'] == OLD_ROOT + 'tests/test_check_mode_workflows.py', n
        assert c['input']['new_string'] in (tree / 'tests/test_check_mode_workflows.py').read_text(), (n, 'not in diff')
    # the custody digest fill (call 91 ran sed on the placeholder)
    cmd = calls[91]['input']['command']
    digest = re.search(r"s/'@@CUSTODY@@'/'([0-9a-f]{64})'/", cmd).group(1)
    test = tree / 'tests/test_check_mode_workflows.py'
    text = test.read_text()
    assert text.count("'@@CUSTODY@@'") == 1
    test.write_text(text.replace("'@@CUSTODY@@'", "'%s'" % digest))
    print('filled custody digest', digest)
    # Edits made to the test file after call 91 (calls 95 and 102)
    for n in (95, 102):
        c = calls[n]
        old, new = c['input']['old_string'], c['input']['new_string']
        text = test.read_text()
        assert text.count(old) == 1, (n, 'old string occurs %d times' % text.count(old))
        test.write_text(text.replace(old, new))
        print('replayed Edit', n, 'tests/test_check_mode_workflows.py')
    # the helper Edit made after call 91 (call 99)
    c = calls[99]
    helper = tree / 'tools/check_mode_replay.py'
    text = helper.read_text()
    old, new = c['input']['old_string'], c['input']['new_string']
    assert text.count(old) == 1
    helper.write_text(text.replace(old, new))
    print('replayed Edit 99 tools/check_mode_replay.py')
    # SCOPE.md: Write 71, python 73, Edit 76, python 104
    scope_rel = 'reviews/c8_check_mode_wrappers_batch3_20261027/SCOPE.md'
    scope = tree / scope_rel
    scope.parent.mkdir(parents=True, exist_ok=True)
    scope.write_text(calls[71]['input']['content'])
    for n in (73, 76, 104):
        c = calls[n]
        if c['name'] == 'Edit':
            text = scope.read_text()
            old, new = c['input']['old_string'], c['input']['new_string']
            assert text.count(old) == 1, (n, 'SCOPE old string occurs %d times' % text.count(old))
            scope.write_text(text.replace(old, new))
        else:
            code = heredoc(c['input']['command'])
            subprocess.run([sys.executable, '-I', '-B', '-S', '-'], input=code.encode(), cwd=tree, check=True)
        print('replayed SCOPE step', n)
    # the contract merge script: printed at call 15, edited by python at calls 40 and 41
    printed = calls[15]['result']
    body = printed[:printed.index('----\n#!/bin/bash')]
    merge = scripts / 'merge_contracts_b3.py'
    merge.write_text(body.rstrip('\n') + '\n')
    for n in (40, 41):
        code = heredoc(calls[n]['input']['command'])
        subprocess.run([sys.executable, '-I', '-B', '-S', '-'], input=code.encode(), cwd=scripts, check=True)
        print('replayed merge-script edit', n)


if __name__ == '__main__':
    main()
