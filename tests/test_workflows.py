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
    """If this finds nothing, the test above passes without checking anything."""
    found = [p for w in WORKFLOWS for p in guarded_paths(w.read_text())]
    assert len(found) >= 8, found
    assert all(p.startswith("tools/") for p in found), found


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


# --------------------------------------------------------------------------
# README.md must not understate what CI enforces
#
# Six checkers CI runs -- consumers, operations, mirror_quotes, mirrors_index,
# operator_directive and the drive_index stats step -- were absent from
# README.md's "What CI enforces" block. A reader auditing this repository's
# safety net from the README would have seen fifteen of twenty-one. Nothing
# compared the two lists, so the block drifted every time a checker landed.
# --------------------------------------------------------------------------

README = ROOT / "README.md"
CI = ROOT / ".github" / "workflows" / "ci.yml"
TOOL = re.compile(r"tools/[A-Za-z0-9_]+\.py")


def ci_tools() -> set[str]:
    return {m.group(0) for block in run_blocks(CI.read_text(encoding="utf-8"))
            for m in TOOL.finditer(block)}


def readme_ci_block(text: str | None = None) -> str:
    """The fenced bash block under the "What CI enforces" heading."""
    text = README.read_text(encoding="utf-8") if text is None else text
    after = text.split("## What CI enforces", 1)
    assert len(after) == 2, "README.md has no 'What CI enforces' heading"
    return after[1].split("```")[1]


def readme_tools(text: str | None = None) -> set[str]:
    return {m.group(0) for m in TOOL.finditer(readme_ci_block(text))}


def test_the_readme_names_every_checker_ci_runs():
    missing = sorted(ci_tools() - readme_tools())
    assert not missing, (
        f"CI runs these and README.md's block does not name them: {missing}. "
        f"A reader auditing the safety net from the README would undercount it.")


def test_the_readme_names_no_checker_ci_does_not_run():
    extra = sorted(readme_tools() - ci_tools())
    assert not extra, (
        f"README.md claims CI runs these and it does not: {extra}. "
        f"Overstating the safety net is the worse direction of the same error.")


def test_the_readme_block_also_runs_the_test_suite():
    assert "pytest" in readme_ci_block()


def test_negative_control_a_checker_dropped_from_the_readme_is_refused():
    text = README.read_text(encoding="utf-8")
    victim = sorted(ci_tools())[0]
    block = readme_ci_block(text)
    mutated = text.replace(block, "\n".join(
        l for l in block.splitlines() if victim not in l) + "\n")
    assert victim not in readme_tools(mutated)
    assert sorted(ci_tools() - readme_tools(mutated)) == [victim]


def test_negative_control_a_checker_invented_in_the_readme_is_refused():
    text = README.read_text(encoding="utf-8")
    block = readme_ci_block(text)
    mutated = text.replace(block, block + "python3 tools/not_a_checker.py   # invented\n")
    assert "tools/not_a_checker.py" in sorted(readme_tools(mutated) - ci_tools())


def test_the_tool_scanner_is_not_vacuous():
    assert len(ci_tools()) >= 20
    assert "tools/claims_check.py" in ci_tools()


# --------------------------------------------------------------------------
# research/interval/README.md's import example must show the real exports
#
# CLAUDE.md tells new code to read that package's public API and not
# reimplement it. The example a reader is pointed at omitted four of the
# sixteen exported names.
# --------------------------------------------------------------------------

INTERVAL_README = ROOT / "research" / "interval" / "README.md"
INTERVAL_INIT = ROOT / "research" / "interval" / "__init__.py"
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _all_names(path: pathlib.Path) -> list[str]:
    body = path.read_text(encoding="utf-8").split("__all__", 1)[1]
    return re.findall(r'"([^"]+)"', body.split("]", 1)[0])


def test_the_interval_readme_import_example_lists_every_export():
    exported = set(_all_names(INTERVAL_INIT))
    text = INTERVAL_README.read_text(encoding="utf-8")
    example = text.split("from research.interval import", 1)[1].split("```", 1)[0]
    shown = set(NAME.findall(example))
    missing = sorted(exported - shown)
    assert not missing, f"exported but not shown in the README example: {missing}"


def test_the_interval_readme_states_the_right_export_count():
    n = len(_all_names(INTERVAL_INIT))
    words = {14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen"}
    assert words[n] in INTERVAL_README.read_text(encoding="utf-8"), (
        f"__all__ has {n} names; the README must say so in words")


def test_the_transcendental_submodule_exports_what_the_package_re_exports():
    """`from ...transcendental import *` must not be missing a name the package lists."""
    package = set(_all_names(INTERVAL_INIT))
    sub = set(_all_names(ROOT / "research" / "interval" / "transcendental.py"))
    core = set(_all_names(ROOT / "research" / "interval" / "core.py"))
    missing = sorted(package - sub - core)
    assert not missing, f"re-exported by the package, in no submodule __all__: {missing}"


# --------------------------------------------------------------------------
# The other direction: a checker no workflow runs is invisible
#
# `test_every_guarded_tool_exists_in_the_tree` travels workflow-text -> tree,
# so a step naming a deleted tool fails. Nothing travelled tree -> workflow, so
# a checker could land in `tools/` and never run in CI, and the only signal
# would be someone noticing. CLAUDE.md already worries about the reverse case
# ("A CI step guarded by `[ -f tools/x.py ]` goes green if the tool is
# deleted"); this is the same hole from the other side.
# --------------------------------------------------------------------------

#: Tools deliberately not run by any workflow, each with the reason. An entry
#: here is a claim a reviewer can check, which is the point of listing them
#: rather than filtering them out silently.
NOT_IN_CI = {
    "tools/disclosure_check.py":
        "pre-commit only. It compares the working tree against the committed "
        "text, so after the commit it self-satisfies and a CI run would prove "
        "nothing. CLAUDE.md's 'Before you commit' block says so in the same "
        "words.",
}


def workflow_tools() -> set[str]:
    out: set[str] = set()
    for path in WORKFLOWS:
        for block in run_blocks(path.read_text(encoding="utf-8")):
            out |= {m.group(0) for m in TOOL.finditer(block)}
    return out


def tools_in_tree() -> set[str]:
    return {f"tools/{p.name}" for p in sorted((ROOT / "tools").glob("*.py"))}


def test_every_checker_in_the_tree_runs_in_a_workflow_or_says_why_not():
    unrun = sorted(tools_in_tree() - workflow_tools() - set(NOT_IN_CI))
    assert not unrun, (
        f"these are in tools/ and no workflow runs them: {unrun}. Add a step, "
        f"or add an entry to NOT_IN_CI in this file giving the reason.")


def test_the_exemption_list_names_only_tools_that_exist_and_are_unrun():
    for rel, reason in NOT_IN_CI.items():
        assert (ROOT / rel).is_file(), f"{rel} is exempted and not in the tree"
        assert rel not in workflow_tools(), (
            f"{rel} is exempted from CI and a workflow runs it; drop the exemption")
        assert len(reason) > 40, f"{rel}: the reason must say something"


def test_the_pre_commit_block_and_the_exemption_agree():
    """CLAUDE.md's block is where a contributor meets the pre-commit-only tool."""
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    for rel in NOT_IN_CI:
        assert rel.split("/")[-1] in claude, (
            f"{rel} runs in no workflow and CLAUDE.md does not mention it, so "
            f"nothing tells a contributor to run it")


def test_negative_control_an_unrun_checker_is_refused():
    invented = "tools/not_wired_up_check.py"
    unrun = (tools_in_tree() | {invented}) - workflow_tools() - set(NOT_IN_CI)
    assert invented in unrun


def test_negative_control_a_stale_exemption_is_refused():
    stale = "tools/claims_check.py"           # CI does run this one
    assert stale in workflow_tools()


def test_the_tree_scanner_is_not_vacuous():
    assert len(tools_in_tree()) >= 20
    assert "tools/verify_manifests.py" in tools_in_tree()
