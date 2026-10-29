#!/usr/bin/env python3
"""Adverse mutation check for pr_evidence.py.

Each mutant breaks one detection or rule. The test suite must fail on every
mutant; a surviving mutant means the tests do not protect that behaviour.
Exit status 0 only when every mutant is killed and the unmutated suite passes.
"""

import shutil
import re
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
    ("HOLD not excluded", "(FAIL|REJECT|HOLD|BLOCK|AMEND)", "(FAIL|REJECT|BLOCK|AMEND)"),
    ("FAIL not excluded", "(FAIL|REJECT|HOLD|BLOCK|AMEND)", "(REJECT|HOLD|BLOCK|AMEND)"),
    ("API local head binding dropped", 'if head != pull["head"]["sha"]:', 'if False:'),
    ("PR number binding dropped", 'if pull.get("number") != n:', 'if False:'),
    ("recommendation binding dropped", 'if ev.get("head_matches_api") is not True:', 'if False:'),
    ("review identity omitted", '"reviewer": {k: (r.get("user") or {}).get(k) for k in ("id", "login")}',
     '"reviewer": {"id": None, "login": None}'),
    ("public review identity omitted", '| {"reviewer": r.get("reviewer", {"id": None, "login": None})}',
     '| {"reviewer": {"id": None, "login": None}}'),
    ("stale raw reused", 'pull = api(f"repos/{slug}/pulls/{n}")',
     'if (out / f"{n}.json").exists():\n            continue\n        pull = api(f"repos/{slug}/pulls/{n}")'),
    ("inflight head change ignored", 'if latest["head"]["sha"] != head or latest["updated_at"] != pull["updated_at"]:',
     'if False:'),
    ("BLOCK not excluded", "(FAIL|REJECT|HOLD|BLOCK|AMEND)", "(FAIL|REJECT|HOLD|AMEND)"),
    ("AMEND not excluded", "(FAIL|REJECT|HOLD|BLOCK|AMEND)", "(FAIL|REJECT|HOLD|BLOCK)"),
]

EXPECTED = dict(zip([m[0] for m in MUTANTS], [
    "test_each_token_is_found", "test_each_token_is_found",
    "test_line_comments_and_longer_identifiers_are_not_hits",
    "test_removed_and_skipped_tests_are_signals",
    *(["test_every_skip_form_is_matched"] * 6), "test_deleted_test_file_is_a_signal",
    "test_float_constructs_are_counted", "test_float_constructs_are_counted",
    "test_ancestry_positive_case", "test_empty_change_set_is_not_superseded",
    "test_deleted_path_matches_only_when_target_also_lacks_it",
    "test_merge_tree_reports_conflicts", "test_each_rule_in_order",
    "test_explicit_statements_match", "test_each_rule_in_order", "test_each_rule_in_order",
    "test_line_filter_counts_only_given_lines", "test_only_added_lines_count_for_float_signals",
    "test_each_rule_in_order", "test_each_rule_in_order",
    "test_free_text_is_dropped_and_verdict_class_kept", "test_each_rule_in_order",
    "test_only_unsuccessful_conclusions_count", "test_each_rule_in_order",
    "test_each_rule_in_order", "test_each_rule_in_order",
    "test_api_local_head_mismatch_is_refused_before_analysis",
    "test_api_wrong_pr_number_is_refused", "test_unbound_metadata_never_recommends_disposition",
    "test_review_identity_preserved_in_public_projection",
    "test_review_identity_preserved_in_public_projection",
    "test_existing_raw_is_refreshed_and_api_reread",
    "test_head_change_during_fetch_preserves_previous_raw",
    "test_block_and_amend_are_not_positive_verdicts", "test_block_and_amend_are_not_positive_verdicts",
], strict=True))

CENSUS_MUTANTS = [
    ("census blob identity dropped", 'if blob != entry["blob"]:', 'if False:', "test_wrong_blob_fails"),
    ("census prefix accepted", 'if blob != entry["blob"]:', 'if not blob.startswith(entry["blob"]):',
     "test_short_blob_prefix_is_not_an_exact_identity"),
    ("census line range dropped", 'if not 1 <= first <= last <= len(lines):', 'if False:',
     "test_line_range_past_the_end_fails"),
    ("census anchor dropped", 'if anchor not in window:', 'if False:', "test_anchor_outside_its_range_fails"),
    ("census target count dropped", 'if len(targets) != 31:', 'if False:', "test_wrong_target_count_refused"),
    ("census axioms dropped", 'if manifest["allowed_axioms"] != ["propext", "Classical.choice", "Quot.sound"]:',
     'if False:', "test_unapproved_axiom_inventory_refused"),
]


def run_suite(directory, filename="test_pr_evidence.py"):
    mode = ["-O"] if sys.flags.optimize else []
    return subprocess.run([sys.executable, "-B", *mode, "-I", str(directory / filename)],
                          cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def intended_failure(proc, expected, count):
    output = proc.stdout
    return (proc.returncode == 1
            and re.search(r"^Ran " + str(count) + r" tests? in ", output, re.M) is not None
            and re.search(r"^FAILED \(failures=[1-9][0-9]*\)$", output, re.M) is not None
            and re.search(r"^FAIL: " + re.escape(expected) + r"[ (]", output, re.M) is not None
            and "\nERROR:" not in output
            and re.search(r"^SyntaxError(?::|$)", output, re.M) is None)


def main():
    source = (HERE / "pr_evidence.py").read_text()
    baseline = run_suite(HERE)
    if baseline.returncode != 0 or re.search(r"^OK$", baseline.stdout, re.M) is None:
        print("BASELINE FAIL: the unmutated suite does not pass")
        return 2
    count_match = re.search(r"^Ran ([0-9]+) tests? in ", baseline.stdout, re.M)
    if count_match is None:
        print("BASELINE FAIL: no complete unittest census")
        return 2
    count = int(count_match.group(1))
    survivors = []
    for name, old, new in MUTANTS:
        if source.count(old) != 1:
            print(f"MUTANT NOT APPLICABLE: {name} ({source.count(old)} matches)")
            survivors.append(name)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            candidate = source.replace(old, new)
            compile(candidate, "pr_evidence.py", "exec")
            (tmp / "pr_evidence.py").write_text(candidate)
            shutil.copy(HERE / "test_pr_evidence.py", tmp / "test_pr_evidence.py")
            result = run_suite(tmp)
            killed = intended_failure(result, EXPECTED[name], count)
            if not killed:
                print(result.stdout)
        print(f"{'KILLED  ' if killed else 'SURVIVED'} {name}")
        if not killed:
            survivors.append(name)
    print(f"{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutants killed")
    census = HERE.parent / "main-obligation-census"
    source = (census / "build_census.py").read_text()
    baseline = run_suite(census, "test_build_census.py")
    if baseline.returncode != 0 or re.search(r"^OK$", baseline.stdout, re.M) is None:
        print("CENSUS BASELINE FAIL", baseline.stdout)
        return 2
    count = int(re.search(r"^Ran ([0-9]+) tests? in ", baseline.stdout, re.M).group(1))
    for name, old, new, expected in CENSUS_MUTANTS:
        if source.count(old) != 1:
            print(f"MUTANT NOT APPLICABLE: {name}")
            survivors.append(name)
            continue
        # Keep the disposable module inside this clone so its immutable Git
        # citation fixtures resolve through the same repository as baseline.
        with tempfile.TemporaryDirectory(prefix=".census-mutant-", dir=HERE) as tmp:
            tmp = Path(tmp)
            candidate = source.replace(old, new)
            compile(candidate, "build_census.py", "exec")
            (tmp / "build_census.py").write_text(candidate)
            shutil.copy(census / "test_build_census.py", tmp / "test_build_census.py")
            result = run_suite(tmp, "test_build_census.py")
            killed = intended_failure(result, expected, count)
        print(f"{'KILLED  ' if killed else 'SURVIVED'} {name}")
        if not killed:
            print(result.stdout)
            survivors.append(name)
    print(f"{len(MUTANTS) + len(CENSUS_MUTANTS) - len(survivors)}/"
          f"{len(MUTANTS) + len(CENSUS_MUTANTS)} total mutants killed")
    baseline = run_suite(HERE, "test_mutation_check.py")
    if baseline.returncode != 0 or re.search(r"^OK$", baseline.stdout, re.M) is None:
        print("HARNESS BASELINE FAIL", baseline.stdout)
        return 2
    harness_count = int(re.search(r"^Ran ([0-9]+) tests? in ", baseline.stdout, re.M).group(1))
    harness = Path(__file__).read_text()
    checks = [
        ("harness exit guard dropped", "return (proc.returncode == 1", "return (proc.returncode != 0"),
        ("harness census guard dropped",
         'and re.search(r"^Ran " + str(count) + r" tests? in ", output, re.M) is not None', 'and True'),
        ("harness syntax guard dropped", 'and re.search(r"^SyntaxError(?::|$)", output, re.M) is None)', 'and True)'),
    ]
    for name, old, new in checks:
        # The mutation anchor is present once as executable code and once in
        # this literal inventory; mutate only the executable function body.
        before, separator, after = harness.partition("\ndef main():")
        if not separator or before.count(old) != 1:
            print(f"MUTANT NOT APPLICABLE: {name}")
            survivors.append(name)
            continue
        candidate = before.replace(old, new) + separator + after
        compile(candidate, "mutation_check.py", "exec")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "mutation_check.py").write_text(candidate)
            shutil.copy(HERE / "test_mutation_check.py", tmp / "test_mutation_check.py")
            result = run_suite(tmp, "test_mutation_check.py")
            killed = intended_failure(result,
                "test_wrong_test_error_skip_incomplete_syntax_and_success_refused", harness_count)
        print(f"{'KILLED  ' if killed else 'SURVIVED'} {name}")
        if not killed:
            print(result.stdout)
            survivors.append(name)
    total = len(MUTANTS) + len(CENSUS_MUTANTS) + len(checks)
    print(f"{total - len(survivors)}/{total} all tool and harness mutants killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
