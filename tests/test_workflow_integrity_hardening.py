"""Execute isolated workflow guards with stubs, not the research computations.

The text readers deliberately accept only the repository's simple step forms;
a workflow refactor must update these tests, not be silently skipped.
These tests do not authenticate an actor or configure server-side protection.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
CI_CHECKERS = (
    "registers_import", "registers_check", "provenance_check", "claims_check",
    "quarantine_check", "verify_manifests", "manifest_integrity_check",
    "reviews_check", "recovery_check", "collision_proposal_check", "carriers_verify",
    "lanes_check", "slack_check", "receipts_check", "frozen_check", "bridge_check",
    "math_status_check", "operations_check", "drive_coverage", "hermite_envelope_report", "drive_reconcile", "native_export_check", "rn_moment_report",
    "rn_certificate", "rn_side24_check", "rn_side24_density_check", "rn_side24_spatial_check", "parallel_math_check", "research_frontier",
    "closure_pipeline",
    "lpw_amplitude_check",
)
RESEARCH_CHECKERS = ("receipts_check", "lanes_check", "claims_check")
CASES = [("ci.yml", name) for name in CI_CHECKERS] + [("research.yml", name) for name in RESEARCH_CHECKERS]


def workflow(name):
    return (ROOT / ".github/workflows" / name).read_text()


def command_for(text, checker):
    candidates = [m[1].strip() for m in re.finditer(r"^\s+run: (.+)$", text, flags=re.M)
                  if f"tools/{checker}.py" in m[1]]
    assert candidates, (checker, candidates)
    return candidates


def shell(command, cwd):
    bash = shutil.which("bash")
    assert bash, "bash is required to test the actual workflow shell semantics"
    return subprocess.run([bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", command],
                          cwd=cwd, text=True, capture_output=True, timeout=10)


@pytest.mark.parametrize("filename,checker", CASES)
@pytest.mark.parametrize("status", ["pass", "fail", "missing"])
def test_every_required_checker_propagates_status(tmp_path, filename, checker, status):
    commands = command_for(workflow(filename), checker)
    (tmp_path / "tools").mkdir()
    if status != "missing":
        (tmp_path / "tools" / f"{checker}.py").write_text("raise SystemExit(" + ("0" if status == "pass" else "7") + ")\n")
    for command in commands:
        assert command.startswith(f"python tools/{checker}.py"), command
        result = shell(command, tmp_path)
        assert result.returncode == {"pass": 0, "fail": 7, "missing": 2}[status], result


@pytest.mark.parametrize("filename", ["ci.yml", "research.yml"])
def test_explicit_read_only_token_and_checkout(filename):
    text = workflow(filename)
    assert re.search(r"^permissions:\n  contents: read\n", text, re.M)
    assert "persist-credentials: false" in text
    assert "fetch-depth: 0" in text  # history-based freeze checks need more than two commits
    assert "timeout-minutes: 30" in text
    assert "    shell: bash" in text


def test_receipt_upload_cannot_succeed_without_files(tmp_path):
    text = workflow("research.yml")
    assert "if-no-files-found: error" in text
    assert "if-no-files-found: warn" not in text
    pattern = re.search(r"^          path: (.+)$", text, re.M)[1]
    folder = tmp_path / "engine/receipts"
    folder.mkdir(parents=True)
    (folder / "README.md").write_text("This is not a receipt.\n")
    assert list(tmp_path.glob(pattern)) == []
    (folder / "lane").mkdir()
    receipt = folder / "lane/run.json"
    receipt.write_text("{}\n")
    assert list(tmp_path.glob(pattern)) == [receipt]


def governed_command(text):
    match = re.search(r"      - name: Check governed paths at end of run\n.*?        run: \|\n(.*?)(?=\n      - name:)",
                      text, re.S)
    assert match, "governed-path step missing or refactored"
    return "\n".join(line[10:] for line in match[1].splitlines() if line.strip())


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return result.stdout


def init_repo(root):
    git(root, "init", "-q")
    (root / "claims").mkdir()
    (root / "claims/state.json").write_text("{}\n")
    (root / ".gitignore").write_text("claims/ignored-*\n")
    git(root, "add", ".")
    git(root, "-c", "user.name=Test Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")


@pytest.mark.parametrize("kind", ["clean", "tracked", "staged", "untracked", "ignored", "outside"])
def test_final_governed_paths_detect_tracked_new_and_ignored(tmp_path, kind):
    init_repo(tmp_path)
    if kind in ("tracked", "staged"):
        (tmp_path / "claims/state.json").write_text('{"changed":true}\n')
        if kind == "staged":
            git(tmp_path, "add", "claims/state.json")
    elif kind == "untracked":
        (tmp_path / "claims/new.json").write_text("{}\n")
    elif kind == "ignored":
        (tmp_path / "claims/ignored-evidence.json").write_text("{}\n")
    elif kind == "outside":
        (tmp_path / "scratch.txt").write_text("allowed local output\n")
    result = shell(governed_command(workflow("research.yml")), tmp_path)
    assert result.returncode == (0 if kind in ("clean", "outside") else 1), result


def test_old_diff_only_guard_misses_new_file_negative_control(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "claims/new.json").write_text("{}\n")
    assert shell("git diff --quiet HEAD -- claims engine/lanes registers drive", tmp_path).returncode == 0
    assert shell(governed_command(workflow("research.yml")), tmp_path).returncode == 1


def test_governed_check_fails_outside_a_git_repository(tmp_path):
    assert shell(governed_command(workflow("research.yml")), tmp_path).returncode != 0
