#!/usr/bin/env python3
"""Source-bound Lean evidence gate for the main-side formal package (stdlib only).

Follows the contract of the Math- formal lane (docs/FORMAL_VERIFICATION.md,
work item #95): a manifest is an evidence sidecar, not a scientific register.

Default (source-only, no Lean run): the manifest is well formed and cannot
self-award execution or review; every bound file matches its SHA-256; every
Lean module is registered; no `sorry`, `native_decide` or `axiom` appears
outside comments; the root module imports exactly the registered modules; the
target inventory equals the theorem declarations found in those modules; every
pinned informal source has a local byte copy with the recorded size and hash;
every target's informal anchor is a verbatim substring of its source.

`--run-lean`: additionally performs a fresh build of the pinned package, runs a
transitive `#print axioms` audit restricted to propext / Classical.choice /
Quot.sound, captures elaborated types, executes real negative controls
(tightened inequalities, a wrong constant, `sorry`, an indirect custom axiom,
`native_decide`) and writes logs plus a receipt under `.lake/formal-evidence/`.
Only that receipt carries `kernel-checked`; the committed manifest says
`proved`. A receipt from arbitrary input is untrusted until matched to the
actual workflow run and checked Git commit.

`--axioms-output FILE`: audit a captured `#print axioms` output without running
Lean (for tests and split CI steps).

`--alignment REVIEW.json`: validate an independently authored alignment record
against the current manifest and scope digests. This checks structure,
identity, coverage and lineage independence; it does not authenticate that the
review happened and never promotes scientific status.

Nothing here edits STATUS.md, PROOF_INDEX.md or any review verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "formal/manifest.json"
ALLOWED = frozenset({"propext", "Classical.choice", "Quot.sound"})
NAME = r"[A-Za-z_][A-Za-z0-9_.']*"
AXIOM_LINE = re.compile(r"'(" + NAME + r")' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
PUBLIC_REPOS = {"d6g8k5htny-coder/main", "d6g8k5htny-coder/Math-", "d6g8k5htny-coder/query-",
                "d6g8k5htny-coder/meta-framework", "d6g8k5htny-coder/Universal-Law-Workspace"}
THEOREM = re.compile(r"^(?:theorem|lemma)\s+(" + NAME + r")", re.M)
NAMESPACE = re.compile(r"^namespace\s+(\S+)", re.M)
TOOLCHAIN = "leanprover/lean4:v4.34.1"
LEAN_COMMIT = "5045d0056413266e57c625dcd7c365b10e377c52"
BUILD_DIR = ".lake"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(text: str):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, "duplicate JSON key: " + key)
            obj[key] = value
        return obj
    return json.loads(text, object_pairs_hook=unique)


def safe_path(root: Path, relative) -> Path:
    require(isinstance(relative, str) and relative and "\\" not in relative, f"invalid path: {relative!r}")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts and str(pure) == relative, f"unsafe path: {relative}")
    path = root
    for part in pure.parts:
        path = path / part
        require(not path.is_symlink(), f"symlink refused: {relative}")
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()), f"missing or escaping file: {relative}")
    return path


def strip_lean_comments(text: str) -> str:
    """Remove `--` line comments and nested `/- -/` block comments. Comment text
    cannot smuggle a proof term; the kernel audit of `sorryAx` is authoritative."""
    out, i, depth, n = [], 0, 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if depth:
            if two == "/-":
                depth, i = depth + 1, i + 2
            elif two == "-/":
                depth, i = depth - 1, i + 2
            else:
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
        elif two == "/-":
            depth, i = 1, i + 2
        elif two == "--":
            end = text.find("\n", i)
            i = n if end < 0 else end
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def audit_axioms(text: str, targets: list) -> dict:
    require(isinstance(targets, list) and targets and len(set(targets)) == len(targets), "empty or duplicate target list")
    records = {}
    for match in AXIOM_LINE.finditer(text):
        name, raw = match.groups()
        require(name in targets and name not in records, "unexpected or duplicate axiom target: " + name)
        axioms = [] if raw is None or not raw.strip() else [a.strip() for a in raw.split(",")]
        require(all(re.fullmatch(NAME, a) for a in axioms), "malformed axiom name")
        require(set(axioms) <= ALLOWED, "forbidden transitive axiom: " + repr(sorted(set(axioms) - ALLOWED)))
        records[name] = sorted(set(axioms))
    require(not AXIOM_LINE.sub("", text).strip(), "unrecognized audit output")
    require(set(records) == set(targets), "missing target axiom report")
    return records


def source_check(root: Path, manifest_rel: str = MANIFEST) -> tuple[dict, str, Path]:
    root = root.resolve()
    raw = safe_path(root, manifest_rel).read_bytes()
    m = load_json(raw.decode("utf-8"))
    require(m.get("schema_version") == 1 and m.get("scientific_effect") == "NONE", "invalid evidence schema")
    require(m.get("scientific_status_authority") is False, "manifest must declare scientific_status_authority false")
    require(m.get("formalization_status") == "proved" and m.get("alignment_status") == "PENDING_INDEPENDENT_REVIEW",
            "source metadata cannot self-award execution or review")
    for key in ("package", "package_root", "root_module", "lean_toolchain"):
        require(isinstance(m.get(key), str) and m[key].strip(), key + " must be a nonempty string")
    require(m["lean_toolchain"] == TOOLCHAIN, "unsupported Lean toolchain in manifest")
    pkg_rel = m["package_root"]
    pkg = root / pkg_rel
    require(pkg.is_dir() and not pkg.is_symlink(), "package root missing")
    files = m.get("files")
    require(isinstance(files, dict) and files, "empty file manifest")
    for name, digest in files.items():
        require(isinstance(digest, str) and HEX64.fullmatch(digest), "invalid source hash: " + name)
        require(sha(safe_path(root, name).read_bytes()) == digest, "source hash mismatch: " + name)
    required = {f"{pkg_rel}/lean-toolchain", f"{pkg_rel}/lakefile.toml", f"{pkg_rel}/lake-manifest.json",
                f"{pkg_rel}/{m['root_module']}.lean", f"{pkg_rel}/SCOPE.md", f"{pkg_rel}/GLOSSARY.md",
                f"{pkg_rel}/README.md", "tools/formal_gate_check.py", "tests/test_formal_gate.py"}
    require(required <= set(files), "unbound control or scope file: " + repr(sorted(required - set(files))))
    require((pkg / "lean-toolchain").read_text(encoding="utf-8").strip() == TOOLCHAIN, "wrong Lean toolchain file")
    lock = load_json((pkg / "lake-manifest.json").read_text(encoding="utf-8"))
    packages = lock.get("packages", [])
    require(isinstance(packages, list), "invalid lake manifest")
    actual = {p["name"]: p["rev"] for p in packages}
    require(len(actual) == len(packages), "duplicate dependency")
    require(actual == m.get("dependency_revisions"), "dependency lock mismatch")
    require(all(HEX40.fullmatch(v) for v in actual.values()), "unpinned dependency")
    modules = m.get("source_modules")
    require(isinstance(modules, list) and modules and len(set(modules)) == len(modules), "invalid source module list")
    expected_root = "".join("import " + path[:-5].replace("/", ".") + "\n" for path in modules)
    root_text = strip_lean_comments((pkg / (m["root_module"] + ".lean")).read_text(encoding="utf-8"))
    require("".join(line + "\n" for line in root_text.splitlines() if line.strip()) == expected_root,
            "root must only import registered modules")
    names = []
    for path in modules:
        require(f"{pkg_rel}/{path}" in files and path.endswith(".lean"), "unbound source module: " + path)
        text = strip_lean_comments((pkg / path).read_text(encoding="utf-8"))
        require(not re.search(r"\bsorry\b", text), "sorry in " + path)
        require(not re.search(r"\bnative_decide\b", text), "native_decide in " + path)
        require(not re.search(r"^\s*axiom\b", text, re.M), "custom axiom declaration in " + path)
        namespaces = NAMESPACE.findall(text)
        require(len(namespaces) == 1, "each module must open exactly one namespace: " + path)
        names.extend(namespaces[0] + "." + n for n in THEOREM.findall(text))
    targets = m.get("targets")
    require(isinstance(targets, list) and targets, "empty target list")
    target_names = []
    sources = {s["id"]: s for s in m.get("sources", [])} if isinstance(m.get("sources"), list) else None
    require(isinstance(sources, dict) and sources, "sources must be a nonempty list")
    source_text = {}
    for s in m["sources"]:
        require(set(s) == {"id", "repository", "commit", "path", "bytes", "sha256", "local_copy"}, "source rows have a fixed schema")
        require(s["repository"] in PUBLIC_REPOS, "nonpublic source repository refused: " + str(s["repository"]))
        require(isinstance(s["commit"], str) and HEX40.fullmatch(s["commit"]), "invalid commit for " + s["id"])
        require(isinstance(s["sha256"], str) and HEX64.fullmatch(s["sha256"]), "invalid sha256 for " + s["id"])
        require(type(s["bytes"]) is int and s["bytes"] > 0, "invalid byte count for " + s["id"])
        payload = safe_path(root, s["local_copy"]).read_bytes()
        require(len(payload) == s["bytes"] and sha(payload) == s["sha256"], "local copy differs from pinned bytes: " + s["id"])
        require(s["local_copy"] in files, "pinned source copy not bound in files: " + s["local_copy"])
        source_text[s["id"]] = payload.decode("utf-8")
    for t in targets:
        require(isinstance(t, dict) and set(t) == {"name", "module", "title", "source", "informal_anchor", "does_not_claim"},
                "target rows have a fixed schema")
        require(isinstance(t["name"], str) and re.fullmatch(NAME, t["name"]), "invalid target name")
        require(t["module"] in modules, "target module not registered: " + t["name"])
        for key in ("title", "does_not_claim"):
            require(isinstance(t[key], str) and t[key].strip(), f"{t['name']}: {key} must be nonempty")
        require(t["source"] in source_text, f"{t['name']}: unknown source")
        require(isinstance(t["informal_anchor"], str) and t["informal_anchor"].strip(), f"{t['name']}: empty informal anchor")
        require(t["informal_anchor"] in source_text[t["source"]], f"{t['name']}: informal anchor is not a verbatim substring of its source")
        target_names.append(t["name"])
    require(target_names == names, "target inventory differs from source declarations")
    lean_files = {p.relative_to(pkg).as_posix() for p in pkg.rglob("*.lean") if BUILD_DIR not in p.relative_to(pkg).parts}
    require(lean_files == set(modules) | {m["root_module"] + ".lean"}, "unregistered Lean module: " + repr(sorted(lean_files - set(modules) - {m['root_module'] + '.lean'})))
    for p in pkg.rglob("*"):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and BUILD_DIR not in p.relative_to(pkg).parts and rel not in files:
            require(rel in set(m.get("unbound_files", [])), "file in package not bound or declared unbound: " + rel)
    return m, sha(raw), pkg


def lake_binary() -> str | None:
    found = shutil.which("lake")
    if found:
        return found
    candidate = Path.home() / ".elan/bin/lake"
    return str(candidate) if candidate.is_file() else None


def check_lean_version(text: str) -> str:
    pin = re.fullmatch(r"leanprover/lean4:v(\d+\.\d+\.\d+)", TOOLCHAIN)
    require(pin is not None, "unsupported pinned Lean toolchain")
    version = text.strip()
    record = (r"Lean \(version " + re.escape(pin.group(1))
              + r", [A-Za-z0-9_.+-]+, commit " + LEAN_COMMIT + r", Release\)")
    require(re.fullmatch(record, version) is not None,
            "unexpected running Lean version")
    return version


def run(command, label, out: Path, cwd: Path, env, expect_success=True):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900, env=env)
    (out / (label + ".log")).write_text(result.stdout, encoding="utf-8")
    print(label + ": exit " + str(result.returncode), flush=True)
    require((result.returncode == 0) == expect_success, "unexpected process outcome: " + label + "\n" + result.stdout[-6000:])
    return result.stdout


def negative_controls(m: dict, pkg: Path) -> dict:
    """Real mutation controls: each must be rejected by Lean or by the axiom gate."""
    root_import = "import " + m["root_module"] + "\n"
    cases = {}
    for label, spec in m.get("negative_controls", {}).items():
        module_text = (pkg / spec["module"]).read_text(encoding="utf-8")
        require(spec["replace"] in module_text, "negative control text not found: " + label)
        cases[label] = (module_text.replace(spec["replace"], spec["with"]), False, None)
    cases["sorry"] = (root_import + "theorem injected : False := by sorry\n#print axioms injected\n", True, "injected")
    cases["custom_imported"] = (root_import + "axiom hiddenPremise : False\ntheorem middle : False := hiddenPremise\n"
                                "theorem injected : False := middle\n#print axioms injected\n", True, "injected")
    cases["native"] = (root_import + "theorem injected : (2 : Nat) + 2 = 4 := by native_decide\n#print axioms injected\n", True, "injected")
    return cases


def execute(m: dict, digest: str, pkg: Path, root: Path) -> dict:
    lake = lake_binary()
    require(lake is not None, "lake not found on PATH or in ~/.elan/bin")
    env = dict(os.environ, PATH=f"{Path(lake).parent}{os.pathsep}{os.environ.get('PATH', '')}")
    out = pkg / BUILD_DIR / "formal-evidence"
    out.mkdir(parents=True, exist_ok=True)
    require(not (pkg / BUILD_DIR).is_symlink() and not (pkg / BUILD_DIR / "build").is_symlink(), "symlink build directory")
    check_lean_version(run([lake, "env", "lean", "--version"], "preflight-version", out, pkg, env))
    if (pkg / BUILD_DIR / "build").exists():
        shutil.rmtree(pkg / BUILD_DIR / "build")
    targets = [t["name"] for t in m["targets"]]
    run([lake, "build"], "build", out, pkg, env)
    run([lake, "env", "leanchecker", m["root_module"]], "leanchecker", out, pkg, env)
    audit = out / "Audit.lean"
    audit.write_text("import " + m["root_module"] + "\n" + "\n".join("#print axioms " + n for n in targets) + "\n", encoding="utf-8")
    axioms = audit_axioms(run([lake, "env", "lean", str(audit)], "axioms", out, pkg, env), targets)
    types = out / "Types.lean"
    types.write_text("import " + m["root_module"] + "\nset_option pp.explicit true\n" + "\n".join("#check " + n for n in targets) + "\n", encoding="utf-8")
    run([lake, "env", "lean", str(types)], "elaborated-types", out, pkg, env)
    outcomes = {}
    for label, (source, succeeds, target) in negative_controls(m, pkg).items():
        path = out / (label + ".lean")
        path.write_text(source, encoding="utf-8")
        log = run([lake, "env", "lean", str(path)], label, out, pkg, env, succeeds)
        if target:
            matches = list(AXIOM_LINE.finditer(log))
            require(matches, "negative control missing axiom report: " + label)
            try:
                audit_axioms("\n".join(x.group() for x in matches), [target])
            except ValueError:
                outcomes[label] = "REJECTED_BY_AXIOM_GATE"
            else:
                raise ValueError("negative control escaped: " + label)
        else:
            require("is false" in log or "unsolved goals" in log or "failed" in log, "negative control did not fail on its proof goal: " + label)
            outcomes[label] = "REJECTED_BY_LEAN"
    _, after, _ = source_check(root)
    require(after == digest, "manifest changed during execution")
    version = check_lean_version(run([lake, "env", "lean", "--version"], "version", out, pkg, env))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True).stdout.strip()
    require(HEX40.fullmatch(head or ""), "missing exact checked commit")
    receipt = dict(schema_version=1, scientific_effect="NONE", scientific_status_authority=False,
                   package=m["package"], formalization_status="kernel-checked",
                   alignment_status="PENDING_INDEPENDENT_REVIEW", manifest_sha256=digest, checked_commit=head,
                   repository=os.environ.get("GITHUB_REPOSITORY"), workflow_run_id=os.environ.get("GITHUB_RUN_ID"),
                   lean_version=version, dependency_revisions=m["dependency_revisions"], axioms=axioms,
                   negative_controls=outcomes)
    receipt["logs"] = {p.name: sha(p.read_bytes()) for p in sorted(out.glob("*.log"))}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def check_alignment(review: dict, manifest_digest: str, targets: list, scope_digest: str):
    require(isinstance(review, dict), "review must be object")
    require(review.get("disposition") == "ACCEPTED", "alignment not accepted")
    require(review.get("manifest_sha256") == manifest_digest, "stale alignment manifest")
    require(review.get("scope_sha256") == scope_digest, "stale alignment scope")
    covered = review.get("targets", [])
    require(isinstance(covered, list) and len(covered) == len(set(covered)) and set(covered) == set(targets), "partial or ambiguous review")
    a, r = review.get("author", {}), review.get("reviewer", {})
    for key in ("provider", "family", "agent"):
        require(isinstance(a.get(key), str) and a[key].strip() and isinstance(r.get(key), str) and r[key].strip(), "missing review lineage")
        require(a[key].strip().casefold() != r[key].strip().casefold(), "lineage not independent: " + key)
    e = review.get("evidence", {})
    require(isinstance(e.get("repository"), str) and re.fullmatch(r"[^/\s]+/[^/\s]+", e["repository"]), "missing evidence repo")
    require(isinstance(e.get("commit"), str) and HEX40.fullmatch(e["commit"]), "mutable review ref")
    require(isinstance(e.get("sha256"), str) and HEX64.fullmatch(e["sha256"]), "missing review hash")
    require(isinstance(e.get("path"), str) and e["path"] and not PurePosixPath(e["path"]).is_absolute()
            and ".." not in PurePosixPath(e["path"]).parts, "missing or unsafe review path")


def render_alignment(m: dict) -> str:
    lines = ["# Informal–formal alignment table", "",
             f"Generated from `{MANIFEST}` by `tools/formal_gate_check.py --write-alignment`; do not hand-edit.",
             "Each row pairs one Lean target with the verbatim informal text it is attached to. The kernel checks the",
             "declaration; the independent alignment review (see REVIEW_LANE.md) checks that the declaration says what",
             "the anchor says. Source-level formalization status of every row is `proved`; only a trusted run receipt",
             "reports `kernel-checked`. Nothing here is a Layer 0 review outcome.", "",
             "| Lean target | Title | Informal anchor | Source | Does not claim |", "|---|---|---|---|---|"]
    for t in m["targets"]:
        anchor = "`" + " ".join(t["informal_anchor"].split()).replace("`", "'").replace("|", "\\|") + "`"
        cells = [f"`{t['name']}`", t["title"].replace("|", "\\|"), anchor, t["source"], t["does_not_claim"].replace("|", "\\|")]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def refresh_hashes(root: Path, manifest_rel: str = MANIFEST) -> int:
    path = safe_path(root, manifest_rel)
    m = load_json(path.read_text(encoding="utf-8"))
    changed = 0
    for name in list(m["files"]):
        digest = sha(safe_path(root, name).read_bytes())
        if m["files"][name] != digest:
            m["files"][name] = digest
            changed += 1
    path.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--run-lean", action="store_true", help="fresh build, leanchecker, axiom audit, negative controls, receipt")
    p.add_argument("--axioms-output", type=Path, help="audit a captured #print axioms output without running Lean")
    p.add_argument("--alignment", type=Path, help="validate an independently authored alignment record; never promotes science")
    p.add_argument("--alignment-table", default="formal/ALIGNMENT.md", help="generated table to compare against the manifest")
    p.add_argument("--write-alignment", action="store_true", help="regenerate the alignment table and exit")
    p.add_argument("--refresh-hashes", action="store_true", help="rewrite manifest file hashes from disk and exit")
    args = p.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.refresh_hashes:
            print(f"updated {refresh_hashes(root, args.manifest)} file hashes in {args.manifest}")
            return 0
        m, digest, pkg = source_check(root, args.manifest)
        if args.write_alignment:
            (root / args.alignment_table).write_text(render_alignment(m), encoding="utf-8")
            print("wrote " + args.alignment_table)
            return 0
        require(safe_path(root, args.alignment_table).read_text(encoding="utf-8") == render_alignment(m),
                "alignment table differs from the manifest rendering; regenerate with --write-alignment")
        if args.alignment:
            check_alignment(load_json(args.alignment.read_text(encoding="utf-8")), digest,
                            [t["name"] for t in m["targets"]], m["files"][f"{m['package_root']}/SCOPE.md"])
            print("ALIGNMENT_RECORD_STRUCTURE_PASS (identity, coverage and lineage only; the review itself must be retrieved and authenticated)")
        if args.axioms_output:
            axioms = audit_axioms(args.axioms_output.read_text(encoding="utf-8"), [t["name"] for t in m["targets"]])
            print(json.dumps({"axiom_audit": axioms, "manifest_sha256": digest}, indent=2, sort_keys=True))
        if args.run_lean:
            print(json.dumps(execute(m, digest, pkg, root), indent=2, sort_keys=True))
        elif not args.axioms_output:
            print("SOURCE_IDENTITY_PASS (not a Lean build or scientific acceptance): " + digest)
            print(json.dumps({"package": m["package"], "targets": len(m["targets"]), "sources": len(m["sources"]),
                              "formalization_status": m["formalization_status"], "alignment_status": m["alignment_status"]},
                             sort_keys=True))
    except (ValueError, KeyError, TypeError, OSError, UnicodeDecodeError, subprocess.SubprocessError) as e:
        print("FORMAL_GATE_FAIL: " + str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
