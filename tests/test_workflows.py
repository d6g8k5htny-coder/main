"""CI workflow honesty: no checker step may mask its own failure.

A step of the form ``python tools/x_check.py || echo skipped`` exits 0 when
the checker exits 1, so the job is green and the checker never ran as far as
the log shows.  research.yml carried exactly that shape for its lane
invariants until 2026-09-19.  These tests read the workflow files as text
(the repository has no YAML dependency) and refuse the shape wherever a
repository checker or the test suite is invoked.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

CHECKER = re.compile(r"python3?\s+(tools/\S+\.py|-m\s+pytest)")
MASK = re.compile(r"\|\|\s*(?:echo|true|:)(?:\s|$)")


def run_blocks(text: str) -> list[str]:
    """Every ``run:`` value, single-line or block scalar, as one string each."""
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(\s*)(?:-\s+)?run:\s*(.*)$", lines[i])
        if not m:
            i += 1
            continue
        indent, rest = m.groups()
        if rest.strip() in ("|", ">", "|-", ">-"):
            body = []
            i += 1
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > len(indent)):
                body.append(lines[i])
                i += 1
            blocks.append("\n".join(body))
        else:
            blocks.append(rest)
            i += 1
    return blocks


def masked_checker_lines(text: str) -> list[str]:
    bad = []
    for block in run_blocks(text):
        for line in block.splitlines():
            if CHECKER.search(line) and MASK.search(line):
                bad.append(line.strip())
    return bad


def test_workflow_files_exist():
    assert [p.name for p in WORKFLOWS] == ["ci.yml", "research.yml"]


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_checker_step_masks_its_failure(path):
    assert masked_checker_lines(path.read_text()) == []


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_repository_checker_step_is_seen(path):
    # The scanner must actually find the checker invocations, or the test
    # above passes vacuously.
    seen = [l for b in run_blocks(path.read_text()) for l in b.splitlines() if CHECKER.search(l)]
    assert len(seen) >= 3, seen


def test_negative_control_masked_step_is_refused():
    text = 'steps:\n  - run: python tools/lanes_check.py || echo "skipped"\n'
    assert masked_checker_lines(text) == ['python tools/lanes_check.py || echo "skipped"']


def test_negative_control_masked_step_in_block_scalar_is_refused():
    text = "steps:\n  - run: |\n      set -e\n      python3 -m pytest -q || true\n  - run: echo done\n"
    assert masked_checker_lines(text) == ["python3 -m pytest -q || true"]


def test_negative_control_the_pre_2026_09_19_line_is_refused():
    line = 'test -f tools/lanes_check.py && python tools/lanes_check.py || echo "skipped"'
    assert masked_checker_lines("  - run: " + line + "\n") == [line]


def test_unmasked_checker_passes():
    assert masked_checker_lines("  - run: python tools/lanes_check.py\n") == []


# ---------------------------------------------------------------------------
# The other way a green build can be a lie: a guard for a tool that is gone
# ---------------------------------------------------------------------------
#
# Most steps read ``if [ -f tools/x_check.py ]; then python tools/x_check.py;
# else echo "skipped (tool absent)"; fi``.  That is honest about a checker that
# has not landed yet -- and it also means deleting a checker turns its step
# green and silent.  The mask test above cannot see it, because there is no
# ``|| echo``.  So: every guarded path must exist in the tree.  A checker may
# still be removed deliberately, but only by removing its CI step in the same
# commit, which is a visible act in the diff.

GUARD = re.compile(r"\[\s*-f\s+(\S+?)\s*\]")


def guarded_paths(text: str) -> list[str]:
    return [m.group(1) for block in run_blocks(text) for m in GUARD.finditer(block)]


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_guarded_tool_exists_in_the_tree(path):
    missing = [p for p in guarded_paths(path.read_text()) if not (ROOT / p).is_file()]
    assert missing == [], f"{path.name} guards a path that is not in the tree: {missing}"


def test_the_guard_scanner_is_not_vacuous():
    """Exercise the parser without requiring production to use weaker guards."""
    text = ('steps:\n  - run: if [ -f tools/claims_check.py ]; then '
            'python tools/claims_check.py; fi\n'
            '  - run: python tools/quarantine_check.py\n')
    assert guarded_paths(text) == ['tools/claims_check.py']


def test_negative_control_a_guard_for_a_missing_tool_is_refused():
    text = ('steps:\n  - run: if [ -f tools/no_such_check.py ]; then '
            'python tools/no_such_check.py; else echo "skipped (tool absent)"; fi\n')
    assert [p for p in guarded_paths(text) if not (ROOT / p).is_file()] == ["tools/no_such_check.py"]


def test_negative_control_a_guard_for_a_present_tool_passes():
    text = ('steps:\n  - run: if [ -f tools/claims_check.py ]; then '
            'python tools/claims_check.py; else echo "skipped (tool absent)"; fi\n')
    assert [p for p in guarded_paths(text) if not (ROOT / p).is_file()] == []


def test_the_mirror_quote_checker_runs_unguarded_in_ci():
    """It landed with its tool, so it needs no guard -- and must not acquire one."""
    blocks = run_blocks((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    calls = [b for b in blocks if "tools/mirror_quotes_check.py" in b]
    assert len(calls) == 1, calls
    assert "if [" not in calls[0] and "-f " not in calls[0], calls[0]
