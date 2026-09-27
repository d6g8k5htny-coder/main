"""CI workflow honesty: no checker step may mask its own failure.

A step of the form ``python tools/x_check.py || echo skipped`` exits 0 when
the checker exits 1, so the job is green and the checker never ran as far as
the log shows.  research.yml carried exactly that shape for its lane
invariants until 2026-09-19.  These tests read the workflow files as text
(the repository has no YAML dependency) and refuse the shape wherever a
repository checker or the test suite is invoked.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

CHECKER = re.compile(r"python3?\s+(?:-O\s+)?(tools/\S+\.py|-m\s+(?:pytest|unittest))")
MASK = re.compile(r"\|\|\s*(?:echo|true|:)(?:\s|$)")


def run_blocks(text: str) -> list[str]:
    """Read this repository's plain, JSON-quoted, single-quoted or block runs.

    JSON string quoting is a subset of YAML double-quoted scalar syntax. This
    reader deliberately does not pretend to validate the entire YAML grammar.
    """
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
            if rest.startswith('"'):
                blocks.append(json.loads(rest))
            elif rest.startswith("'"):
                if not re.fullmatch(r"'(?:[^']|'')*'", rest):
                    raise ValueError("unsupported single-quoted YAML run scalar")
                blocks.append(rest[1:-1].replace("''", "'"))
            else:
                blocks.append(rest)
            i += 1
    return blocks


def unsafe_plain_run_scalars(text: str) -> list[str]:
    """Detect colon separators in plain run values, not colons in block bodies.

    Shell quotes inside a plain YAML scalar do not protect YAML's ': ' token.
    Quote the whole scalar or use a block. This is a bounded syntax regression
    check, not a replacement for GitHub's workflow parser.
    """
    bad = []
    for line in text.splitlines():
        match = re.match(r"^\s*(?:-\s+)?run:\s*(.*)$", line)
        if not match:
            continue
        value = match[1]
        if value.startswith(('"', "'", "|", ">")):
            continue
        if re.search(r":(?:\s|$)", value):
            bad.append(value)
    return bad


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_plain_run_scalars_do_not_contain_yaml_mapping_separator(path):
    assert unsafe_plain_run_scalars(path.read_text()) == []


@pytest.mark.parametrize("command", [
    "python -m pip install --require-hashes --only-binary=:all: -r requirements-ci.lock",
    'echo "result: passed"',
    "echo result:\tpassed",
])
def test_negative_control_plain_run_colon_separator_is_rejected(command):
    assert unsafe_plain_run_scalars("  - run: " + command + "\n") == [command]


@pytest.mark.parametrize("quote", ['"', "'"])
def test_whole_scalar_quoting_preserves_pip_command(quote):
    command = "python -m pip install --require-hashes --only-binary=:all: -r requirements-ci.lock"
    text = "  - run: " + quote + command + quote + "\n"
    assert unsafe_plain_run_scalars(text) == []
    assert run_blocks(text) == [command]


def test_colon_in_block_body_or_without_whitespace_is_not_mapping_separator():
    text = 'steps:\n  - run: |\n      echo "result: passed"\n  - run: curl https://example.invalid\n'
    assert unsafe_plain_run_scalars(text) == []


def masked_checker_lines(text: str) -> list[str]:
    bad = []
    for block in run_blocks(text):
        for line in block.splitlines():
            if CHECKER.search(line) and MASK.search(line):
                bad.append(line.strip())
    return bad


def test_workflow_files_exist():
    # Keep an exact reviewed inventory: do not accept arbitrary new workflows.
    # navigation.yml runs only the declared page checker and its synthetic tests.
    assert [p.name for p in WORKFLOWS] == [
        "ci.yml", "navigation.yml", "research.yml", "withdrawal-governance.yml"]


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


def test_negative_control_optimized_pilot_tests_cannot_mask_failure():
    line = "python3 -O -m unittest discover -s tests -p test_withdrawal.py || true"
    assert masked_checker_lines("  - run: " + line + "\n") == [line]


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
