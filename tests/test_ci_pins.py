"""Enforce the reviewed CI inputs and fail-closed pilot integration.

These tests do not attest hosted runner bytes or install server merge rules.
Official action refs and PyPI wheel metadata were checked on 2026-09-22.
"""
from pathlib import Path
import re

import pytest

from test_workflows import run_blocks
from test_workflow_integrity_hardening import command_for, shell

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github/workflows").glob("*.yml"))
ACTION_PINS = {
    "actions/checkout": "11d5960a326750d5838078e36cf38b85af677262",
    "actions/setup-python": "a26af69be951a213d495a4c3e4e4022e16d87065",
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
}
INSTALL = "python -m pip install --require-hashes --only-binary=:all: -r requirements-ci.lock"
LOCK_PACKAGES = {"pytest", "iniconfig", "packaging", "pluggy", "pygments"}


def action_pin_errors(text):
    references = re.findall(r"^\s*(?:-\s+)?uses:\s*(\S+)", text, re.M)
    errors = []
    if not references:
        errors.append("no action references found")
    for reference in references:
        name, _, commit = reference.partition("@")
        if not re.fullmatch(r"[0-9a-f]{40}", commit) or ACTION_PINS.get(name) != commit:
            errors.append(reference)
    return errors


def lock_entries(text):
    entries = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([a-z][a-z0-9-]*)==([0-9]+(?:\.[0-9]+)+) --hash=sha256:([0-9a-f]{64})", line)
        if not match or match[1] in entries:
            raise ValueError("lock requires unique exact versions and SHA-256 wheel hashes")
        entries[match[1]] = (match[2], match[3])
    if set(entries) != LOCK_PACKAGES:
        raise ValueError("lock must cover the reviewed Linux/macOS pytest dependency closure")
    return entries


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_every_action_uses_its_reviewed_full_commit(path):
    assert action_pin_errors(path.read_text()) == []


@pytest.mark.parametrize("reference", [
    "actions/checkout@v4", "actions/setup-python@main",
    "actions/upload-artifact@ea165f8", "unreviewed/action@" + "a" * 40,
    "actions/checkout@" + "0" * 40,
])
def test_mutable_partial_or_unreviewed_action_is_rejected(reference):
    assert action_pin_errors("steps:\n  - uses: " + reference + "\n")


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_workflows_select_exact_python_and_declared_os_series(path):
    text = path.read_text()
    assert re.findall(r"^\s+runs-on:\s*(\S+)", text, re.M) == ["ubuntu-24.04"]
    assert re.findall(r'^\s+python-version:\s*"([^"]+)"', text, re.M) == ["3.11.16"]
    assert "actions/setup-python@" + ACTION_PINS["actions/setup-python"] in text
    assert re.search(r"^permissions:\n  contents: read\n", text, re.M)
    assert "persist-credentials: false" in text
    assert "    shell: bash" in text


@pytest.mark.parametrize("name", ["ci.yml", "research.yml"])
def test_pytest_install_is_hash_checked_and_wheel_only(name):
    commands = run_blocks((ROOT / ".github/workflows" / name).read_text())
    installs = [command for command in commands if "pip install" in command]
    assert installs == [INSTALL]


def test_lock_covers_only_the_reviewed_test_dependency_closure():
    lock_entries((ROOT / "requirements-ci.lock").read_text())


@pytest.mark.parametrize("mutation", ["missing_hash", "range", "missing_dependency", "duplicate", "extra_dependency"])
def test_incomplete_or_unpinned_lock_is_rejected(mutation):
    text = (ROOT / "requirements-ci.lock").read_text()
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if mutation == "missing_hash":
        lines[0] = lines[0].split(" --hash")[0]
    elif mutation == "range":
        lines[0] = lines[0].replace("==", ">=", 1)
    elif mutation == "missing_dependency":
        lines.pop()
    elif mutation == "duplicate":
        lines.append(lines[0])
    else:
        lines.append("unreviewed==1.0 --hash=sha256:" + "0" * 64)
    with pytest.raises(ValueError):
        lock_entries("\n".join(lines))


@pytest.mark.parametrize("checker", ["withdrawal_check", "rollout_guard"])
@pytest.mark.parametrize("status", ["pass", "fail", "missing"])
def test_full_ci_pilot_checks_cannot_pass_when_failed_or_deleted(tmp_path, checker, status):
    commands = command_for((ROOT / ".github/workflows/ci.yml").read_text(), checker)
    assert len(commands) == 1
    assert commands[0].startswith(f"python tools/{checker}.py ")
    (tmp_path / "tools").mkdir()
    if status != "missing":
        (tmp_path / "tools" / (checker + ".py")).write_text(
            "raise SystemExit(" + ("0" if status == "pass" else "7") + ")\n")
    result = shell(commands[0], tmp_path)
    assert result.returncode == {"pass": 0, "fail": 7, "missing": 2}[status]
