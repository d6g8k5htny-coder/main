"""The slack registry's CI checker passes on the real registry and fails when
an invariant is reported broken.

``research/slack/registry.py`` already carries 52 tests and was mutation-tested
by its author; this file only pins the thin checker that wires it into CI, with
the one control that matters for a checker: it must be able to fail.
"""

from __future__ import annotations

import os
import subprocess
import sys
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "tools", "slack_check.py")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_checker_passes_on_the_real_registry():
    proc = subprocess.run([sys.executable, CHECKER], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "problems=0" in proc.stdout
    assert "never correctness" in proc.stdout       # the disclaimer must survive


def test_checker_fails_when_an_invariant_is_reported_broken():
    """Negative control: a reported problem must turn into exit 1."""
    from tools import slack_check
    from research.slack import registry

    with mock.patch.object(registry, "audit_registry", return_value=["synthetic invariant failure"]):
        assert slack_check.main([]) == 1


def test_checker_fails_on_forbidden_report_language():
    """Negative control: promotion or correctness language in the rendered
    report is a defect even when every record's arithmetic is fine."""
    from tools import slack_check
    from research.slack import registry

    with mock.patch.object(registry, "forbidden_words_in_report", return_value=["discharged"]):
        assert slack_check.main([]) == 1
