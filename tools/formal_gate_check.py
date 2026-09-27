#!/usr/bin/env python3
"""Fail-closed gate for the Lean formalization registry (Layer 1).

Checks, offline and without writing:

* `formal/registry.json` is well formed (unique keys, known statuses, pinned
  toolchain, allowed/forbidden axiom lists).
* Every Lean file under the project is listed with a matching SHA-256; no
  `sorry`, `native_decide`, or unregistered `axiom` appears in any of them.
* Every byte-pinned informal source has a local copy with the recorded size and
  SHA-256, and every registered informal anchor is a verbatim substring of it.
* Every claim with a Lean status names a declaration that exists in the listed
  file; every `none` claim names nothing.
* With `--axioms-output FILE` (the captured output of
  `lake env lean UniversalLaw/Audit.lean`) or `--run-lean`, every
  `kernel-checked` declaration depends only on allowed axioms and every
  `proved` declaration avoids the forbidden ones. Missing declarations fail.

`--static-only` skips the axiom lane and says so in the output. The check never
changes a status: it can only refuse one. A green run establishes that the
registered arithmetic is kernel-checked; it is not mathematical acceptance of
any Layer 0 claim.
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

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "formal/registry.json"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
PUBLIC_REPOS = {"d6g8k5htny-coder/main", "d6g8k5htny-coder/Math-", "d6g8k5htny-coder/query-",
                "d6g8k5htny-coder/meta-framework", "d6g8k5htny-coder/Universal-Law-Workspace"}
LEAN_STATUSES = ("specified", "proved", "kernel-checked")
AUDIT_LINE = re.compile(r"^'(?P<name>[^']+)' (?:does not depend on any axioms|depends on axioms: \[(?P<axioms>[^\]]*)\])\s*$")
DECL = re.compile(r"^\s*(?:private\s+|protected\s+|noncomputable\s+)*(?:theorem|lemma|def|abbrev|instance|structure|inductive)\s+([^\s:({]+)", re.M)
AXIOM_DECL = re.compile(r"^\s*axiom\s+([^\s:({]+)", re.M)
NAMESPACE = re.compile(r"^\s*namespace\s+([^\s]+)", re.M)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def local_file(root: Path, relative) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"expected nonempty relative path: {relative!r}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe path: {relative}")
    path = root
    for part in pure.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink refused: {relative}")
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or outside-root file: {relative}")
    return path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_registry(root: Path, registry: str) -> dict:
    data = json.loads(local_file(root, registry).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    require(isinstance(data, dict) and data.get("schema") == 1, "unknown registry schema")
    require(data.get("scope") == "FORMALIZATION_STATUS_ONLY", "unknown registry scope")
    require(data.get("scientific_status_authority") is False, "registry must declare scientific_status_authority false")
    for key in ("lean_toolchain", "project_root", "audit_module"):
        require(isinstance(data.get(key), str) and data[key], f"{key} must be a nonempty string")
    for key in ("allowed_axioms", "forbidden_axioms"):
        value = data.get(key)
        require(isinstance(value, list) and all(isinstance(x, str) and x for x in value), f"{key} must list axiom names")
    require(not set(data["allowed_axioms"]) & set(data["forbidden_axioms"]), "an axiom cannot be both allowed and forbidden")
    levels = data.get("status_levels")
    require(isinstance(levels, dict) and set(levels) == {"none", *LEAN_STATUSES}, "status_levels must define exactly none/specified/proved/kernel-checked")
    alignment = data.get("alignment_review_levels")
    require(isinstance(alignment, dict) and set(alignment) == {"open", "reviewed", "not-applicable"}, "alignment_review_levels must define open/reviewed/not-applicable")
    for key in ("lean_files", "sources", "claims", "cross_checks"):
        require(isinstance(data.get(key), list), f"{key} must be a list")
    axioms = data.get("project_axioms", [])
    require(isinstance(axioms, list), "project_axioms must be a list")
    for item in axioms:
        require(isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("scope_note"), str)
                and item["name"] and item["scope_note"], "each project axiom needs a name and a scope_note")
    return data


def strip_lean_comments(text: str) -> str:
    """Remove `--` line comments and (nested) `/- -/` block comments.

    Comment text cannot smuggle a proof term, and the kernel audit of `sorryAx`
    is the authoritative guard; this only keeps the cheap textual scan honest.
    """
    out, i, depth, n = [], 0, 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if depth:
            if two == "/-":
                depth += 1
                i += 2
            elif two == "-/":
                depth -= 1
                i += 2
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


def check_lean_files(root: Path, data: dict, problems: list) -> dict:
    project = data["project_root"]
    listed = {}
    for item in data["lean_files"]:
        require(isinstance(item, dict) and set(item) == {"path", "sha256"}, "lean_files rows need exactly path and sha256")
        require(isinstance(item["sha256"], str) and HEX64.fullmatch(item["sha256"]), f"invalid sha256 for {item['path']}")
        require(item["path"] not in listed, f"duplicate lean file: {item['path']}")
        path = local_file(root, item["path"])
        payload = path.read_bytes()
        listed[item["path"]] = payload
        if sha256(payload) != item["sha256"]:
            problems.append(f"{item['path']}: sha256 differs from registry")
    toolchain_rel = f"{project}/lean-toolchain"
    require(toolchain_rel in listed, "lean-toolchain must be a listed lean file")
    require(f"{project}/lakefile.toml" in listed or f"{project}/lakefile.lean" in listed, "lakefile must be a listed lean file")
    require(data["audit_module"] in listed, "audit module must be a listed lean file")
    if listed[toolchain_rel].decode("utf-8").strip() != data["lean_toolchain"]:
        problems.append("lean-toolchain file differs from registry lean_toolchain")
    project_dir = local_file(root, toolchain_rel).parent
    for path in sorted(project_dir.rglob("*.lean")):
        if ".lake" in path.relative_to(project_dir).parts:
            continue
        relative = path.relative_to(root.resolve()).as_posix()
        if relative not in listed:
            problems.append(f"{relative}: Lean file present but not registered")
    registered_axioms = {item["name"] for item in data.get("project_axioms", [])}
    for relative, payload in listed.items():
        if not relative.endswith(".lean"):
            continue
        text = strip_lean_comments(payload.decode("utf-8"))
        if re.search(r"\bsorry\b", text):
            problems.append(f"{relative}: contains sorry")
        if re.search(r"\bnative_decide\b", text):
            problems.append(f"{relative}: contains native_decide")
        for name in AXIOM_DECL.findall(text):
            if name not in registered_axioms and not any(name.endswith("." + n) or n.endswith("." + name) for n in registered_axioms):
                problems.append(f"{relative}: axiom {name} is not registered in project_axioms")
    return listed


def check_sources(root: Path, data: dict, problems: list) -> dict:
    sources = {}
    for item in data["sources"]:
        require(isinstance(item, dict) and set(item) == {"id", "repository", "commit", "path", "bytes", "sha256", "local_copy"}, "source rows have a fixed schema")
        require(item["repository"] in PUBLIC_REPOS, f"nonpublic source repository refused: {item['repository']}")
        require(isinstance(item["commit"], str) and HEX40.fullmatch(item["commit"]), f"invalid commit for {item['id']}")
        require(isinstance(item["sha256"], str) and HEX64.fullmatch(item["sha256"]), f"invalid sha256 for {item['id']}")
        require(type(item["bytes"]) is int and item["bytes"] > 0, f"invalid byte count for {item['id']}")
        require(isinstance(item["id"], str) and item["id"] and item["id"] not in sources, f"invalid or duplicate source id: {item['id']}")
        payload = local_file(root, item["local_copy"]).read_bytes()
        if len(payload) != item["bytes"] or sha256(payload) != item["sha256"]:
            problems.append(f"{item['id']}: local copy {item['local_copy']} differs from pinned bytes")
        sources[item["id"]] = payload.decode("utf-8")
    return sources


def declared_names(text: str) -> set:
    names = set()
    namespaces = NAMESPACE.findall(text)
    for short in DECL.findall(text):
        names.add(short)
        for ns in namespaces:
            names.add(f"{ns}.{short}")
    return names


def check_claims(root: Path, data: dict, lean_files: dict, sources: dict, problems: list) -> dict:
    ids = set()
    counts = {status: 0 for status in data["status_levels"]}
    alignment_counts = {level: 0 for level in data["alignment_review_levels"]}
    audited = {}
    declared_cache = {}
    for claim in data["claims"]:
        require(isinstance(claim, dict), "claims must be objects")
        cid = claim.get("id")
        require(isinstance(cid, str) and re.fullmatch(r"[a-z0-9][a-z0-9-]*", cid or ""), f"invalid claim id: {cid!r}")
        require(cid not in ids, f"duplicate claim id: {cid}")
        ids.add(cid)
        status = claim.get("status")
        require(status in data["status_levels"], f"{cid}: unknown status {status!r}")
        counts[status] += 1
        for key in ("layer0_object", "title", "does_not_claim"):
            require(isinstance(claim.get(key), str) and claim[key].strip(), f"{cid}: {key} must be a nonempty string")
        review = claim.get("alignment_review")
        require(isinstance(review, dict) and set(review) == {"status", "author", "reviewer", "record"}, f"{cid}: alignment_review schema")
        require(review["status"] in data["alignment_review_levels"], f"{cid}: unknown alignment status")
        alignment_counts[review["status"]] += 1
        if status == "none":
            for key in ("lean_declaration", "lean_file", "source", "informal_anchor"):
                require(claim.get(key) is None, f"{cid}: status none cannot carry {key}")
            require(review["status"] == "not-applicable", f"{cid}: status none requires alignment not-applicable")
            continue
        require(review["status"] != "not-applicable", f"{cid}: a Lean statement needs an alignment status")
        require(isinstance(review["author"], str) and review["author"], f"{cid}: alignment author required")
        if review["status"] == "reviewed":
            require(isinstance(review["reviewer"], str) and review["reviewer"], f"{cid}: reviewed alignment needs a reviewer")
            require(isinstance(review["record"], str), f"{cid}: reviewed alignment needs a record path")
            local_file(root, review["record"])
        decl, lean_file, source_id, anchor = (claim.get(k) for k in ("lean_declaration", "lean_file", "source", "informal_anchor"))
        require(isinstance(decl, str) and re.fullmatch(r"[A-Za-z_][\w'.]*", decl), f"{cid}: invalid lean_declaration")
        require(lean_file in lean_files and lean_file.endswith(".lean"), f"{cid}: lean_file must be a registered .lean file")
        if lean_file not in declared_cache:
            declared_cache[lean_file] = declared_names(strip_lean_comments(lean_files[lean_file].decode("utf-8")))
        if decl not in declared_cache[lean_file]:
            problems.append(f"{cid}: declaration {decl} not found in {lean_file}")
        require(source_id in sources, f"{cid}: unknown source {source_id!r}")
        require(isinstance(anchor, str) and anchor.strip(), f"{cid}: informal_anchor must be a nonempty string")
        if anchor not in sources[source_id]:
            problems.append(f"{cid}: informal anchor is not a verbatim substring of source {source_id}")
        if status in ("proved", "kernel-checked"):
            audited[decl] = (cid, status)
    return {"counts": counts, "alignment": alignment_counts, "audited": audited}


def parse_audit(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        match = AUDIT_LINE.match(line.strip())
        if match:
            axioms = match.group("axioms")
            result[match.group("name")] = set() if axioms is None else {a.strip() for a in axioms.split(",") if a.strip()}
    return result


def check_axioms(data: dict, audited: dict, audit_text: str, problems: list) -> dict:
    if re.search(r"\berror\b", audit_text):
        problems.append("axiom audit output reports an error")
    parsed = parse_audit(audit_text)
    allowed, forbidden = set(data["allowed_axioms"]), set(data["forbidden_axioms"])
    for decl, (cid, status) in audited.items():
        if decl not in parsed:
            problems.append(f"{cid}: {decl} missing from axiom audit output")
            continue
        used = parsed[decl]
        if status == "kernel-checked" and not used <= allowed:
            problems.append(f"{cid}: {decl} depends on non-allowed axioms {sorted(used - allowed)}")
        if used & forbidden:
            problems.append(f"{cid}: {decl} depends on forbidden axioms {sorted(used & forbidden)}")
    return {"declarations_reported": len(parsed), "declarations_required": len(audited)}


def lake_binary() -> str | None:
    found = shutil.which("lake")
    if found:
        return found
    candidate = Path.home() / ".elan/bin/lake"
    return str(candidate) if candidate.is_file() else None


def run_lean(root: Path, data: dict) -> str:
    lake = lake_binary()
    if lake is None:
        raise OSError("lake not found on PATH or in ~/.elan/bin")
    project = local_file(root, f"{data['project_root']}/lean-toolchain").parent
    env = dict(os.environ, PATH=f"{Path(lake).parent}{os.pathsep}{os.environ.get('PATH', '')}")
    build = subprocess.run([lake, "build"], cwd=project, capture_output=True, text=True, env=env)
    if build.returncode != 0:
        raise OSError("lake build failed:\n" + build.stdout[-4000:] + build.stderr[-4000:])
    audit_rel = PurePosixPath(data["audit_module"]).relative_to(data["project_root"]).as_posix()
    audit = subprocess.run([lake, "env", "lean", audit_rel], cwd=project, capture_output=True, text=True, env=env)
    if audit.returncode != 0:
        raise OSError("axiom audit failed:\n" + audit.stdout[-4000:] + audit.stderr[-4000:])
    return audit.stdout


def render_alignment(data: dict) -> str:
    lines = ["# Informal–formal alignment table", "",
             "Generated from `formal/registry.json` by `tools/formal_gate_check.py --write-alignment`; do not hand-edit.",
             "Each row pairs one Lean declaration with the verbatim informal text it is attached to. The kernel checks the",
             "declaration; the alignment review lane checks that the declaration says what the anchor says. Status here is",
             "formalization status only and changes no Layer 0 review outcome.", "",
             "| Claim | Layer 0 object | Status | Alignment | Lean declaration | Informal anchor | Does not claim |",
             "|---|---|---|---|---|---|---|"]
    for claim in data["claims"]:
        anchor = claim["informal_anchor"]
        anchor_cell = "—" if anchor is None else "`" + " ".join(anchor.split()).replace("`", "'").replace("|", "\\|") + "`"
        decl = "—" if claim["lean_declaration"] is None else f"`{claim['lean_declaration']}`"
        cells = [claim["id"], claim["layer0_object"], claim["status"], claim["alignment_review"]["status"], decl,
                 anchor_cell, claim["does_not_claim"]]
        lines.append("| " + " | ".join(c.replace("|", "\\|") if i != 5 else c for i, c in enumerate(cells)) + " |")
    lines.append("")
    return "\n".join(lines)


def check(root: Path, registry: str = REGISTRY, axioms_output: Path | None = None, static_only: bool = False,
          run: bool = False, alignment: str | None = None) -> dict:
    root = root.resolve()
    data = load_registry(root, registry)
    problems: list[str] = []
    lean_files = check_lean_files(root, data, problems)
    sources = check_sources(root, data, problems)
    claims = check_claims(root, data, lean_files, sources, problems)
    audit_summary: dict | str
    if run:
        audit_summary = check_axioms(data, claims["audited"], run_lean(root, data), problems)
    elif axioms_output is not None:
        audit_summary = check_axioms(data, claims["audited"], Path(axioms_output).read_text(encoding="utf-8"), problems)
    elif static_only:
        audit_summary = "NOT RUN (--static-only): kernel and axiom lane unchecked in this invocation"
    else:
        raise ValueError("axiom lane required: pass --axioms-output FILE, --run-lean, or --static-only")
    if alignment is not None:
        try:
            current = local_file(root, alignment).read_text(encoding="utf-8")
        except ValueError as exc:
            problems.append(f"alignment table: {exc}")
        else:
            if current != render_alignment(data):
                problems.append(f"{alignment}: differs from the registry rendering; regenerate with --write-alignment")
    return {"scope": data["scope"], "lean_toolchain": data["lean_toolchain"], "lean_files": len(lean_files),
            "sources": len(sources), "claims": sum(claims["counts"].values()), "status_counts": claims["counts"],
            "alignment_counts": claims["alignment"], "cross_checks": len(data["cross_checks"]),
            "axiom_audit": audit_summary, "problems": problems}


def refresh_hashes(root: Path, registry: str) -> int:
    path = local_file(root, registry)
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    changed = 0
    for item in data["lean_files"]:
        digest = sha256(local_file(root, item["path"]).read_bytes())
        if item["sha256"] != digest:
            item["sha256"] = digest
            changed += 1
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument("--axioms-output", type=Path, help="captured output of `lake env lean <audit module>`")
    parser.add_argument("--run-lean", action="store_true", help="run lake build and the axiom audit now")
    parser.add_argument("--static-only", action="store_true", help="skip the kernel/axiom lane and say so")
    parser.add_argument("--alignment", default="formal/ALIGNMENT.md", help="alignment table to compare against the registry")
    parser.add_argument("--no-alignment", action="store_true", help="do not compare the alignment table")
    parser.add_argument("--write-alignment", action="store_true", help="regenerate the alignment table and exit")
    parser.add_argument("--refresh-hashes", action="store_true", help="rewrite lean_files hashes from disk and exit")
    args = parser.parse_args(argv)
    try:
        if args.write_alignment:
            data = load_registry(args.root.resolve(), args.registry)
            (args.root.resolve() / args.alignment).write_text(render_alignment(data), encoding="utf-8")
            print(f"wrote {args.alignment}")
            return 0
        if args.refresh_hashes:
            print(f"updated {refresh_hashes(args.root, args.registry)} lean_files hashes")
            return 0
        result = check(args.root, args.registry, args.axioms_output, args.static_only, args.run_lean,
                       None if args.no_alignment else args.alignment)
    except (ValueError, OSError, TypeError, KeyError, UnicodeDecodeError) as exc:
        print(f"formal_gate_check: INVALID_INPUT: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Scope: formalization registry, Lean file identity, informal anchors, and axiom audit only. "
          "Kernel-checked arithmetic is not acceptance of the surrounding analytic argument or of any Layer 0 claim.")
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
