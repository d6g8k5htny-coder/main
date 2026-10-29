#!/usr/bin/env python3
"""Disposition evidence for open pull requests, for CBZ-0 Phase 2.

Two steps, kept separate so the analysis is reproducible offline:

  fetch    saves each PR's GitHub metadata, reviews, files and recent
           comments to raw/<n>.json (needs the `gh` CLI and network);
  analyze  combines those files with local git objects into
           dispositions.json and DISPOSITIONS.md (git only, no network).

The output is evidence for a human or agent deciding each PR's disposition.
It certifies nothing. The Layer A fields are lexical signals over the PR's
own diff: a signal means "look here", and the absence of a signal does not
show that a change satisfies Layer A.

Usage:
  pr_evidence.py fetch   --repo-slug OWNER/REPO --out DIR N [N ...]
  pr_evidence.py analyze --git DIR --target REF --out DIR N [N ...]

Before `analyze`, fetch each head into refs/cbz/pr/<n> and each base branch
into refs/remotes/origin/<base>.
"""

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

LEAN_TOKENS = ("sorryAx", "sorry", "admit", "axiom", "native_decide")
LEAN_TOKEN_RE = re.compile(r"(?<![\w.'])(" + "|".join(LEAN_TOKENS) + r")(?![\w'])")
TEST_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(test\w*)\s*\(")
SKIP_PATTERNS = (
    r"unittest\.skip(?:If|Unless)?\(",
    r"\.skipTest\(",
    r"pytest\.mark\.skip",
    r"pytest\.skip\(",
    r"pytest\.mark\.xfail",
    r"unittest\.expectedFailure",
)
SKIP_RE = re.compile("|".join(SKIP_PATTERNS))
# Explicit statements that this PR is not to be merged. Phrases such as "no merge
# occurred" or "this custodian does not merge" describe actions, not intent, so they
# are deliberately excluded; every match is recorded with its context for review.
NOT_FOR_MERGE_TITLE_RE = re.compile(r"do not merge|not for merge", re.I)
NOT_FOR_MERGE_BODY_RE = re.compile(
    r"\bnot for merge\b|\bmust not be merged\b|\bnot to be merged\b|\(do not merge\)"
    r"|\bdo not merge this\b", re.I)
CERTIFICATE_PATH_RE = re.compile(r"certif|interval|bound|proof|exact", re.I)
# math functions whose results are exact for integer arguments.
EXACT_MATH = {"gcd", "lcm", "isqrt", "comb", "perm", "factorial", "prod"}


def git(repo, *args, check=True):
    """Run git in repo and return stdout; raise on failure when check is set."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def rev(repo, ref):
    return git(repo, "rev-parse", "--verify", ref + "^{commit}").stdout.strip()


def is_ancestor(repo, a, b):
    return git(repo, "merge-base", "--is-ancestor", a, b, check=False).returncode == 0


def merge_base(repo, a, b):
    proc = git(repo, "merge-base", a, b, check=False)
    return proc.stdout.strip() or None


def changed_paths(repo, base, head):
    """Return [(status, path)] for base..head, with renames split into D and A."""
    out = git(repo, "diff", "--no-renames", "--name-status", base, head).stdout
    rows = []
    for line in out.splitlines():
        if line.strip():
            status, path = line.split("\t", 1)
            rows.append((status[0], path))
    return rows


def blob(repo, commit, path):
    proc = git(repo, "rev-parse", "--verify", "--quiet", f"{commit}:{path}", check=False)
    return proc.stdout.strip() or None


def content_on_target(repo, head, target, paths):
    """Per changed path, whether target already has the head's content.

    An added or modified path matches when its blob is equal. A deleted path
    matches when the target no longer has it either.
    """
    result = {}
    for status, path in paths:
        if status == "D":
            result[path] = blob(repo, target, path) is None
        else:
            result[path] = blob(repo, head, path) == blob(repo, target, path)
    return result


def merge_tree(repo, target, head):
    """Trial three-way merge without touching any ref or worktree.

    Returns (clean, tree, conflicted_paths).
    """
    proc = git(repo, "merge-tree", "--write-tree", "--name-only", "--no-messages",
               target, head, check=False)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"merge-tree failed: {proc.stderr.strip()}")
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    tree = lines[0] if lines else None
    return proc.returncode == 0, tree, lines[1:]


def diff_lines(repo, base, head, path):
    """Return (added, removed) line lists for one path in base..head."""
    out = git(repo, "diff", "--no-renames", "-U0", base, head, "--", path).stdout
    added, removed = [], []
    for line in out.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    return added, removed


def strip_lean_line_comment(line):
    """Drop a `--` comment from one Lean line. Block comments are not tracked."""
    idx = line.find("--")
    return line if idx < 0 else line[:idx]


def lean_signals(added_lines):
    hits = {}
    for line in added_lines:
        for match in LEAN_TOKEN_RE.finditer(strip_lean_line_comment(line)):
            hits[match.group(1)] = hits.get(match.group(1), 0) + 1
    return hits


def is_test_path(path):
    name = path.rsplit("/", 1)[-1]
    return path.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py")
                                    or "/tests/" in "/" + path)


def test_signals(repo, base, head, paths):
    """Removed test functions, deleted test files and added skips."""
    removed_tests, added_skips, deleted_files = [], [], []
    for status, path in paths:
        if not is_test_path(path):
            continue
        if status == "D":
            deleted_files.append(path)
            continue
        added, removed = diff_lines(repo, base, head, path)
        added_names = {m.group(1) for l in added if (m := TEST_DEF_RE.match(l))}
        for line in removed:
            m = TEST_DEF_RE.match(line)
            if m and m.group(1) not in added_names:
                removed_tests.append(f"{path}::{m.group(1)}")
        for line in added:
            if SKIP_RE.search(line):
                added_skips.append(f"{path}: {line.strip()[:120]}")
    return {"removed_test_functions": removed_tests, "deleted_test_files": deleted_files,
            "added_skips": added_skips}


HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def added_line_numbers(repo, base, head, path):
    """Line numbers, in the head version of path, of lines that base..head adds."""
    out = git(repo, "diff", "--no-renames", "-U0", base, head, "--", path).stdout
    lines = set()
    for line in out.splitlines():
        m = HUNK_RE.match(line)
        if m:
            start, count = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            lines.update(range(start, start + count))
    return lines


def float_signals(source, lines=None):
    """Count floating-point constructs in Python source, optionally only on the given lines.

    Returns None when the source does not parse.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    counts = {"float_literals": 0, "float_calls": 0, "inexact_math_calls": 0,
              "float_libraries": 0}
    for node in ast.walk(tree):
        if lines is not None and getattr(node, "lineno", None) not in lines:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            counts["float_literals"] += 1
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "float":
                counts["float_calls"] += 1
            elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                  and func.value.id == "math" and func.attr not in EXACT_MATH):
                counts["inexact_math_calls"] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            if any(n.split(".")[0] in ("numpy", "scipy") for n in names):
                counts["float_libraries"] += 1
    return counts


def python_float_signals(repo, base, head, paths):
    """Float constructs on the lines this PR adds, for every added or modified .py file.

    Each entry records whether its path looks like certification code; only those
    entries count as Layer A signals in `recommend`. The rest are listed for review.
    """
    found = {}
    for status, path in paths:
        if status == "D" or not path.endswith(".py"):
            continue
        source = git(repo, "show", f"{head}:{path}").stdout
        counts = float_signals(source, added_line_numbers(repo, base, head, path))
        if counts is None:
            found[path] = {"unparsed": True, "certificate_like": bool(CERTIFICATE_PATH_RE.search(path))}
        elif any(counts.values()):
            found[path] = counts | {"certificate_like": bool(CERTIFICATE_PATH_RE.search(path))}
    return found


def certificate_float(signals):
    return {p: c for p, c in signals.items() if c.get("certificate_like")}


def review_summary(raw):
    head = raw["pull"]["head"]["sha"]
    rows = []
    for r in raw.get("reviews", []):
        first = (r.get("body") or "").strip().splitlines()
        rows.append({"id": r["id"], "state": r["state"], "commit": r.get("commit_id"),
                     "at_head": r.get("commit_id") == head,
                     "headline": first[0][:160] if first else ""})
    return rows


def pass_verdict(text):
    """A positive review headline: PASS, ACCEPT or CONFIRM, with no FAIL, REJECT or HOLD."""
    return (bool(re.search(r"\b(PASS|ACCEPT|CONFIRM)", text))
            and not re.search(r"\b(FAIL|REJECT|HOLD)", text))


def failing_checks(check_runs):
    """Check runs at the head whose conclusion is failure, timed_out or cancelled."""
    return [c for c in check_runs
            if c.rsplit("/", 1)[-1] in ("failure", "timed_out", "cancelled", "action_required")]


def not_for_merge_evidence(title, body):
    """Return the matched context, or None when the PR does not say it is not for merge."""
    m = NOT_FOR_MERGE_TITLE_RE.search(title or "")
    if m:
        return f"title: {title.strip()[:160]}"
    m = NOT_FOR_MERGE_BODY_RE.search(body or "")
    if m:
        a, b = max(0, m.start() - 80), min(len(body), m.end() + 40)
        return "body: ..." + " ".join(body[a:b].split()) + "..."
    return None


def recommend(ev):
    """Apply the documented rules in order; return (disposition, reason)."""
    if ev["head_is_ancestor_of_target"]:
        return "SUPERSEDED", "the head is already an ancestor of the target"
    if ev["changed_paths"] and all(ev["content_on_target"].values()):
        return "SUPERSEDED", "the target already holds this PR's content on every changed path"
    if ev["base_ref"] != ev["target_ref"]:
        if ev.get("parent_pr"):
            return "STACK-DEPENDENT", f"stacked on open PR #{ev['parent_pr']}; disposition the stack from the bottom"
        if ev.get("base_on_target"):
            return "RETARGET-CANDIDATE", (f"its base branch {ev['base_ref']} is already on the target "
                                          "and no open PR owns it; retarget to the target before deciding")
        return "STACK-DEPENDENT", (f"stacked on branch {ev['base_ref']}, which no open PR owns "
                                   "and which is not on the target")
    if ev["not_for_merge"]:
        return "QUARANTINE-ARCHIVE", "the PR itself says it is not for merge"
    if not ev["merge_clean"]:
        return "QUARANTINE-ARCHIVE", f"conflicts with the target in {len(ev['conflicts'])} path(s)"
    if ev.get("failing_checks"):
        return "REPAIR-OR-ARCHIVE", f"{len(ev['failing_checks'])} check run(s) at the head did not pass"
    sig = ev["layer_a_signals"]
    if sig["lean"] or sig["tests"]["removed_test_functions"] or sig["tests"]["deleted_test_files"] \
            or sig["tests"]["added_skips"] or certificate_float(sig["python_float"]):
        return "LAYER-A-REVIEW-FIRST", "lexical Layer A signals in its diff; review them before any integration"
    passes = [r for r in ev["reviews"] if r["at_head"] and pass_verdict(r["headline"])]
    if passes:
        return "INTEGRATE-CANDIDATE", "clean merge, no Layer A signals, a PASS review at the current head"
    return "INTEGRATE-CANDIDATE-UNREVIEWED", "clean merge and no Layer A signals, but no PASS review at the current head"


def analyze_one(repo, n, raw, target_ref, open_heads):
    pull = raw["pull"]
    head = rev(repo, f"refs/cbz/pr/{n}")
    base_ref = pull["base"]["ref"]
    target = rev(repo, target_ref)
    base_commit = target if base_ref == "main" else rev(repo, f"refs/remotes/origin/{base_ref}")
    mb = merge_base(repo, base_commit, head)
    paths = changed_paths(repo, mb, head) if mb else []
    lean = {}
    for status, path in paths:
        if path.endswith(".lean") and status != "D":
            for tok, count in lean_signals(diff_lines(repo, mb, head, path)[0]).items():
                lean[f"{path}:{tok}"] = count
    clean, tree, conflicts = merge_tree(repo, target, head)
    ev = {
        "number": n,
        "title": pull.get("title"),
        "draft": pull.get("draft"),
        "created_at": pull.get("created_at"),
        "updated_at": pull.get("updated_at"),
        "head": head,
        "head_matches_api": head == pull["head"]["sha"],
        "head_ref": pull["head"]["ref"],
        "base_ref": base_ref,
        "target_ref": "main",
        "target": target,
        "merge_base": mb,
        "parent_pr": open_heads.get(base_ref),
        "head_is_ancestor_of_target": is_ancestor(repo, head, target),
        "changed_paths": [f"{s}\t{p}" for s, p in paths],
        "content_on_target": content_on_target(repo, head, target, paths),
        "merge_clean": clean,
        "merge_tree": tree if clean else None,
        "conflicts": conflicts,
        "layer_a_signals": {
            "lean": lean,
            "tests": test_signals(repo, mb, head, paths) if mb else {},
            "python_float": python_float_signals(repo, mb, head, paths) if mb else {},
        },
        "not_for_merge": not_for_merge_evidence(pull.get("title"), pull.get("body")),
        "base_on_target": base_ref != "main" and is_ancestor(repo, base_commit, target),
        "reviews": review_summary(raw),
        "head_checks": raw.get("check_runs", []),
        "failing_checks": failing_checks(raw.get("check_runs", [])),
        "last_comments": raw.get("comments", [])[-3:],
    }
    ev["recommendation"], ev["reason"] = recommend(ev)
    return ev


def fetch(slug, numbers, out):
    out.mkdir(parents=True, exist_ok=True)
    def api(path):
        cmd = ["gh", "api", path]
        return json.loads(subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout or "null")

    def pages(path):
        # Explicit page numbers on repos/OWNER/REPO paths: `gh --paginate` follows
        # Link headers that name repositories/{id}/..., which some proxies refuse.
        rows, page = [], 1
        while True:
            batch = api(f"{path}{'&' if '?' in path else '?'}per_page=100&page={page}")
            rows.extend(batch)
            if len(batch) < 100:
                return rows
            page += 1

    for n in numbers:
        if (out / f"{n}.json").exists():
            continue
        pull = api(f"repos/{slug}/pulls/{n}")
        head = pull["head"]["sha"]
        runs = api(f"repos/{slug}/commits/{head}/check-runs?per_page=100").get("check_runs", [])
        comments = pages(f"repos/{slug}/issues/{n}/comments")
        raw = {
            "pull": {k: pull[k] for k in ("number", "state", "draft", "title", "body",
                                          "created_at", "updated_at", "mergeable_state")}
                    | {"head": {"ref": pull["head"]["ref"], "sha": head},
                       "base": {"ref": pull["base"]["ref"], "sha": pull["base"]["sha"]}},
            "reviews": [{"id": r["id"], "state": r["state"], "commit_id": r["commit_id"],
                         "submitted_at": r.get("submitted_at"), "body": r.get("body")}
                        for r in pages(f"repos/{slug}/pulls/{n}/reviews")],
            "check_runs": sorted({f"{c['name']}={c['status']}/{c.get('conclusion') or '-'}" for c in runs}),
            "comments": [{"id": c["id"], "created_at": c["created_at"],
                          "headline": (c.get("body") or "").strip().splitlines()[0][:200]
                          if (c.get("body") or "").strip() else ""} for c in comments],
        }
        (out / f"{n}.json").write_text(json.dumps(raw, indent=1, sort_keys=True) + "\n")


def publishable(ev):
    """Copy of one PR's evidence without quoted third-party text.

    Review and comment text stays in the local raw/ files, which are not
    published; the output keeps identities, states and the derived verdict class.
    """
    out = dict(ev)
    out["reviews"] = [{k: r[k] for k in ("id", "state", "commit", "at_head")}
                      | {"positive_verdict": pass_verdict(r["headline"])} for r in ev["reviews"]]
    out["last_comments"] = [{k: c[k] for k in ("id", "created_at")} for c in ev["last_comments"]]
    out["title"] = ev["title"]
    return out


def render(rows, target):
    lines = ["# Stale open PR disposition evidence", "",
             f"Target `main` at `{target}`. These are recommendations from the documented rules in "
             "`pr_evidence.py`, not dispositions. Layer A fields are lexical signals, not certification.", "",
             "| PR | Base | Recommendation | Reason | Merge onto main | Signals |",
             "|---|---|---|---|---|---|"]
    for ev in rows:
        sig = ev["layer_a_signals"]
        tests = sig.get("tests") or {}
        parts = []
        if sig["lean"]:
            parts.append("lean " + ", ".join(f"{k}×{v}" for k, v in sig["lean"].items()))
        for key in ("removed_test_functions", "deleted_test_files", "added_skips"):
            if tests.get(key):
                parts.append(f"{key.replace('_', ' ')}: {len(tests[key])}")
        cert = certificate_float(sig["python_float"])
        if cert:
            parts.append(f"float on added lines of {len(cert)} certificate-like file(s)")
        other = len(sig["python_float"]) - len(cert)
        if other:
            parts.append(f"float on added lines of {other} other .py file(s), for review")
        merge = "clean" if ev["merge_clean"] else f"{len(ev['conflicts'])} conflict(s)"
        lines.append(f"| #{ev['number']} | `{ev['base_ref']}` | **{ev['recommendation']}** | "
                     f"{ev['reason']} | {merge} | {'; '.join(parts) or 'none'} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--repo-slug", required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("numbers", type=int, nargs="+")
    a = sub.add_parser("analyze")
    a.add_argument("--git", type=Path, required=True)
    a.add_argument("--target", required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--open-heads", type=Path,
                   help="JSON object mapping each open PR's head branch to its number")
    a.add_argument("numbers", type=int, nargs="+")
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        fetch(args.repo_slug, args.numbers, args.out / "raw")
        return 0
    open_heads = json.loads(args.open_heads.read_text()) if args.open_heads else {}
    rows = []
    for n in args.numbers:
        raw = json.loads((args.out / "raw" / f"{n}.json").read_text())
        rows.append(analyze_one(args.git, n, raw, args.target, open_heads))
    target = rev(args.git, args.target)
    (args.out / "dispositions.json").write_text(
        json.dumps({"target": target, "prs": [publishable(r) for r in rows]},
                   indent=1, sort_keys=True) + "\n")
    (args.out / "DISPOSITIONS.md").write_text(render(rows, target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
