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

CHECKER = re.compile(
    r"\bpython3?(?:\s+-O)?\s+(tools/\S+\.py|engine/solver_pilots/\S+\.py|-m\s+(?:pytest|unittest)\b)"
)
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
            blocks.append(" ".join(line.strip() for line in body) if rest.strip().startswith(">") else "\n".join(body))
        else:
            blocks.append(rest)
            i += 1
    return blocks


def logical_lines(block: str) -> list[str]:
    # A mask after a continued CLI argument still masks the checker process.
    return re.sub(r"\\\s*\n\s*", " ", block).splitlines()


def masked_checker_lines(text: str) -> list[str]:
    bad = []
    for block in run_blocks(text):
        for line in logical_lines(block):
            if CHECKER.search(line) and MASK.search(line):
                bad.append(line.strip())
    return bad


def test_workflow_files_exist():
    assert [p.name for p in WORKFLOWS] == ["ci.yml", "h3-solver-pilot.yml", "research.yml", "withdrawal-governance.yml"]


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_checker_step_masks_its_failure(path):
    assert masked_checker_lines(path.read_text()) == []


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_repository_checker_step_is_seen(path):
    # The scanner must actually find the checker invocations, or the test
    # above passes vacuously.
    seen = [l for b in run_blocks(path.read_text()) for l in logical_lines(b) if CHECKER.search(l)]
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


@pytest.mark.parametrize("command", [
    "python3 -m unittest discover -s tests -p test_h3_solver_pilot.py -v",
    "python3 -O -m unittest discover -s tests -p test_h3_solver_pilot.py -v",
    "python3 -O tools/withdrawal_check.py check",
    "python3 engine/solver_pilots/h3_20260921/h3_solver.py --cold",
])
def test_new_workflow_checker_mask_is_refused(command):
    assert masked_checker_lines("  - run: " + command + " || true\n") == [command + " || true"]
    assert masked_checker_lines("  - run: " + command + "\n") == []


def test_wrapped_h3_cli_mask_is_refused():
    text = ("  - run: |\n"
            "      python3 engine/solver_pilots/h3_20260921/h3_solver.py \\\n"
            "        --out result.json --cold || echo skipped\n")
    bad = masked_checker_lines(text)
    assert len(bad) == 1 and "h3_solver.py" in bad[0] and "|| echo skipped" in bad[0]


def test_folded_checker_mask_is_refused():
    text = "  - run: >-\n      python3 tools/withdrawal_check.py check\n      || true\n"
    assert masked_checker_lines(text) == ["python3 tools/withdrawal_check.py check || true"]
