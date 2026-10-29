#!/usr/bin/env python3
"""Adverse mutation check for pr_evidence.py.

Each mutant breaks one detection or rule. The test suite must fail on every
mutant; a surviving mutant means the tests do not protect that behaviour.
Exit status 0 only when every mutant is killed and the unmutated suite passes.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUTANTS = [
    ("drop sorry token", '"sorryAx", "sorry", ', '"sorryAx", '),
    ("drop native_decide token", ', "native_decide")', ")"),
    ("ignore line comments wrongly", 'return line if idx < 0 else line[:idx]', "return line"),
    ("miss removed tests", "if m and m.group(1) not in added_names:", "if False:"),
    ("miss unittest skips", r'r"unittest\.skip(?:If|Unless)?\(",', 'r"NEVERMATCH",'),
    ("miss skipTest", r'r"\.skipTest\(",', 'r"NEVERMATCH",'),
    ("miss pytest mark skip", r'r"pytest\.mark\.skip",', 'r"NEVERMATCH",'),
    ("miss pytest skip call", r'r"pytest\.skip\(",', 'r"NEVERMATCH",'),
    ("miss xfail", r'r"pytest\.mark\.xfail",', 'r"NEVERMATCH",'),
    ("miss expectedFailure", r'r"unittest\.expectedFailure",', 'r"NEVERMATCH",'),
    ("miss deleted test files", "deleted_files.append(path)", "pass"),
    ("miss float literals", 'counts["float_literals"] += 1', "pass"),
    ("treat all math calls as exact", "func.attr not in EXACT_MATH", "False"),
    ("ancestry always false", "== 0\n\n\ndef merge_base", "== 999\n\n\ndef merge_base"),
    ("empty change set counts as landed", 'if ev["changed_paths"] and all(', "if all("),
    ("deleted path never matches", "result[path] = blob(repo, target, path) is None",
     "result[path] = False"),
    ("conflicts reported clean", "return proc.returncode == 0, tree, lines[1:]",
     "return True, tree, lines[1:]"),
    ("not-for-merge ignored", 'if ev["not_for_merge"]:', "if False:"),
    ("title not-for-merge missed", "m = NOT_FOR_MERGE_TITLE_RE.search(title or \"\")", "m = None"),
    ("retarget never offered", 'if ev.get("base_on_target"):', "if False:"),
    ("stale review accepted", 'if r["at_head"] and pass_verdict', "if pass_verdict"),
    ("line filter ignored", "if lines is not None and getattr(node, \"lineno\", None) not in lines:",
     "if False:"),
    ("hunk start off by one", "lines.update(range(start, start + count))",
     "lines.update(range(start + 1, start + count + 1))"),
    ("certificate floats ignored", "or certificate_float(sig[\"python_float\"]):", "or False:"),
    ("all floats treated as certificate", 'return {p: c for p, c in signals.items() if c.get("certificate_like")}',
     "return dict(signals)"),
    ("headlines published", 'out["last_comments"] = [{k: c[k] for k in ("id", "created_at")} for c in ev["last_comments"]]',
     'out["last_comments"] = ev["last_comments"]'),
    ("failing checks ignored", 'if ev.get("failing_checks"):', "if False:"),
    ("timed-out checks ignored", '("failure", "timed_out", "cancelled", "action_required")',
     '("failure", "cancelled", "action_required")'),
    ("ACCEPT not positive", "(PASS|ACCEPT|CONFIRM)", "(PASS|CONFIRM)"),
    ("HOLD not excluded", "(FAIL|REJECT|HOLD)", "(FAIL|REJECT)"),
    ("FAIL not excluded", "(FAIL|REJECT|HOLD)", "(REJECT|HOLD)"),
]


def run_suite(directory):
    proc = subprocess.run([sys.executable, "-B", "-I", str(directory / "test_pr_evidence.py")],
                          cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode == 0


def main():
    source = (HERE / "pr_evidence.py").read_text()
    if not run_suite(HERE):
        print("BASELINE FAIL: the unmutated suite does not pass")
        return 2
    survivors = []
    for name, old, new in MUTANTS:
        if source.count(old) != 1:
            print(f"MUTANT NOT APPLICABLE: {name} ({source.count(old)} matches)")
            survivors.append(name)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "pr_evidence.py").write_text(source.replace(old, new))
            shutil.copy(HERE / "test_pr_evidence.py", tmp / "test_pr_evidence.py")
            killed = not run_suite(tmp)
        print(f"{'KILLED  ' if killed else 'SURVIVED'} {name}")
        if not killed:
            survivors.append(name)
    print(f"{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutants killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
