#!/usr/bin/env python3
"""Adverse mutation check for tools/cbz/reference_crawler.py (governance A8 / B-006).

Each mutant disables or corrupts one detection, policy rule, extractor or topology
behaviour. A mutant counts as killed only when tests/test_cbz_reference_crawler.py ran
at least one test and reported FAILED. A syntax-broken copy must come out invalid,
never killed. Exit 0 only if every mutant is killed.
Run from the repository root: python3 -B -S tools/cbz/reference_crawler_mutation_check.py
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "cbz" / "reference_crawler.py"
TEST = ROOT / "tests" / "test_cbz_reference_crawler.py"
RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
MUTANTS = [
    ("blob identity ignored", "for b in ref.blobs if b != oid]", "for b in ref.blobs if False]"),
    ("sha256 ignored", "for d in ref.sha256s if d != digest]", "for d in ref.sha256s if False]"),
    ("bytes ignored", "for n in ref.sizes if n != len(data)]", "for n in ref.sizes if False]"),
    ("missing path silent", 'findings.append(finding(cls, target.name, path, f"path absent at {commit}", ref))', "pass"),
    ("missing commit assumed present", "if not live and not mirror.ensure_commit(commit):", "if False:"),
    ("live drift misclassified", 'cls = "LIVE_DRIFT" if live else "IDENTITY_MISMATCH"', 'cls = "IDENTITY_MISMATCH"'),
    ("unknown repository ignored", 'findings.append(finding("UNKNOWN_REPOSITORY",', '(finding("UNKNOWN_REPOSITORY",'),
    ("foreign reference silent", 'findings.append(finding("FOREIGN_REPOSITORY",', '(finding("FOREIGN_REPOSITORY",'),
    ("same-name foreign repository conflated", "        target = None  # another owner's repository with the same short name",
     "        pass"),
    ("strict ignored", '(strict and row["class"] == "FROZEN_CHANGED" and interface_changed(row))', "False"),
    ("lean statements blind", "**diff_maps(lean_statements(old), lean_statements(new))", "**diff_maps({}, {})"),
    ("python signatures drop arguments", 'return f"{prefix} {fn.name}({ast.unparse(fn.args)}){ret}"',
     'return f"{prefix} {fn.name}(){ret}"'),
    ("json shape shallow", "SHAPE_DEPTH = 6", "SHAPE_DEPTH = 0"),
    ("foreign owners included", "        if owner not in owners:\n            continue", "        pass"),
    ("unparseable silent", 'findings.append(finding("UNPARSEABLE", source, path,', "(finding(\"UNPARSEABLE\", source, path,"),
    ("unpinned not reported", 'findings.append(finding("TOPOLOGY_UNPINNED", node.name, "", "registered but not pinned"))',
     "pass"),
    ("blocking ignored at exit", 'return 1 if report["blocking"] else 0', "return 0"),
    ("nested comments not tracked", "depth, i = depth + 1, i + 2", "depth, i = depth, i + 2"),
    ("line comments kept", 'elif text.startswith("--", i):', "elif False:"),
    ("pin skew silent", 'findings.append(finding("PIN_SKEW", node.name, "", f"pinned {node.commit}, tip {tip}"))', "pass"),
    ("require-current-pins ignored", '(require_current_pins and row["class"] == "PIN_SKEW")', "False"),
    ("unreachable halts the crawl",
     'findings.append(finding("UNREACHABLE_REPOSITORY", node.name, "", str(exc)))', "raise"),
    ("unreachable target not blocking", '"TARGET_UNREACHABLE", target.name', '"FROZEN_REMOVED", target.name'),
    ("missing pin silent",
     'findings.append(finding("MISSING_PIN", node.name, "", f"pinned commit {node.commit} not found"))', "pass"),
    ("frozen change unreported", 'elif current[1] != oid:', "elif False:"),
    ("url path not decoded", 'path = urllib.parse.unquote(m.group("path").rstrip(".:;"))',
     'path = m.group("path").rstrip(".:;")'),
    ("excerpt lines never read", 'lines = (obj["line_start"], obj["line_end"]) if', "lines = None if"),
    ("excerpt checked as whole file", 'if ref.lines and kind == "blob":', "if False:"),
    ("excerpt off by one", 'excerpt = "".join(file_lines[start - 1:end])', 'excerpt = "".join(file_lines[start:end])'),
    ("excerpt identity unchecked", 'problems += identity_problems(excerpt, ref, f" in lines {start}-{end}")', "pass"),
    ("excerpt range unchecked", "if not 1 <= start <= end <= len(file_lines):", "if False:"),
    ("excerpt text ignored", 'if ref.text is not None and ref.text.encode("utf-8") != excerpt:', "if False:"),
    ("fixtures not skipped", "if not include_fixtures and is_fixture(path):", "if False:"),
    ("include-fixtures ignored", "references.extend(collect(node, owners, findings, include_fixtures, skipped))",
     "references.extend(collect(node, owners, findings, False, skipped))"),
    ("skipped fixtures not counted", "skipped[node.name] = skipped.get(node.name, 0) + 1", "pass"),
    ("fixture match includes file names", 'for part in path.split("/")[:-1])', 'for part in path.split("/"))'),
    ("skip count hidden in markdown", 'skipped = sum(report["skipped_fixture_files"].values())', "skipped = 0"),
    ("json strings not decoded",
     "for string in json_strings(doc):\n                text_references(string, source, path, owners, refs)",
     "text_references(text, source, path, owners, refs)"),
    ("backslash allowed in url paths", 'URL_PATH_CHAR = r"[^\\s()\\]>\\"\'`#?|,\\\\]"',
     'URL_PATH_CHAR = r"[^\\s()\\]>\\"\'`#?|,]"'),
    ("balanced parentheses dropped", "|\\({URL_PATH_CHAR}*\\))+)", ")+)"),
    ("extra blob keys ignored",
     'BLOB_KEYS = ("blob", "git_blob", "observed_blob", "blob_sha", "blob_sha1", "git_blob_sha", "git_blob_sha1")',
     'BLOB_KEYS = ("blob", "git_blob", "observed_blob")'),
    ("extra commit keys ignored", 'COMMIT_KEYS = ("commit", "ref", "source_commit", "cut", "head", "at_commit")',
     'COMMIT_KEYS = ("commit", "ref", "source_commit")'),
    ("raw_sha256 ignored", 'SHA256_KEYS = ("sha256", "raw_sha256")', 'SHA256_KEYS = ("sha256",)'),
    ("byte_length ignored", 'BYTES_KEYS = ("bytes", "byte_length")', 'BYTES_KEYS = ("bytes",)'),
    ("only the first recorded value checked", "if valid(obj.get(k)) and obj[k] not in out:",
     "if valid(obj.get(k)) and not out:"),
    ("byte-identical claims ignored",
     "for key in sorted(k for k in obj if k.startswith(ALSO_AT_PREFIX) and hex40(obj[k])):", "for key in ():"),
]


def run_suite(tool_dir):
    """Run the crawler tests against the copy of the tool in tool_dir."""
    work = Path(tool_dir)
    (work / "tests").mkdir(exist_ok=True)
    shutil.copy(TEST, work / "tests" / TEST.name)
    proc = subprocess.run([sys.executable, "-B", "-S", "-m", "unittest", f"tests.{TEST.stem}"],
                          cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ran = RAN.search(proc.stdout)
    if not ran or int(ran.group(1)) == 0:
        return "invalid"
    if proc.returncode == 0 and "\nOK" in proc.stdout:
        return "pass"
    return "fail" if "FAILED (" in proc.stdout else "invalid"


def with_tool(source):
    tmp = tempfile.TemporaryDirectory()
    (Path(tmp.name) / "tools" / "cbz").mkdir(parents=True)
    (Path(tmp.name) / "tools" / "cbz" / "reference_crawler.py").write_text(source)
    return tmp


def main():
    source = TOOL.read_text()
    with with_tool(source) as tmp:
        if run_suite(tmp) != "pass":
            print("BASELINE FAIL: the unmutated suite does not pass")
            return 2
    with with_tool(source + "\ndef (:\n") as tmp:
        if run_suite(tmp) != "invalid":
            print("CONTROL FAIL: a syntax-broken copy was not reported invalid")
            return 2
    killed = 0
    for name, old, new in MUTANTS:
        if source.count(old) != 1:
            print(f"NOT APPLICABLE {name} ({source.count(old)} matches)")
            continue
        with with_tool(source.replace(old, new)) as tmp:
            outcome = run_suite(tmp)
        killed += outcome == "fail"
        print(f"{'KILLED  ' if outcome == 'fail' else 'INVALID ' if outcome == 'invalid' else 'SURVIVED'} {name}")
    print(f"{killed}/{len(MUTANTS)} mutants killed")
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
