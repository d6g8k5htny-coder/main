#!/usr/bin/env python3
"""Which repository files a certificate binds by SHA-256, generated and verified.

Several candidate records under `research/` bind repository content by digest as
*source identity*: the certificate says "this result was produced against these
exact bytes", and the replay checkers that consume it refuse to run when a byte
has moved. That is the right discipline and nothing here weakens it.

What was missing is that **no document in the tree said which files those are**.
The only way to find out was to edit one and read a rejection that names the
dependency but not the certificate:

    tools/twelve_project_check.py      REJECTED
    tools/h3_rn_n6_check.py            REJECTED: repository dependency mismatch:
                                                 quarantine/EXCLUSIONS.json

Three separate changes paid a cycle for that in four days, on
`research/bands/ladder.py`, `research/cover/ledger.py` and
`quarantine/EXCLUSIONS.json`. Two of the bound files are checkers in `tools/`
and one is a document in `docs/`, which nobody would guess from either.

So the inventory is generated. `research/PINNED_SOURCES.md` is written by this
tool from the certificates themselves, recomputed on every run, and any drift
fails. Every binding is also verified against the content in this tree, so a
mismatch is reported once -- by file, with the certificate that names it and
both digests -- instead of as four opaque replay rejections.

## Four shapes, and one genuine ambiguity

A tool that knows only one shape under-reports the surface:

    path   -> {"bytes": 37974, "sha256": "58ebe18f..."}
    path   -> "58ebe18f..."                       a bare digest string
    a::b   -> {...}                               member `b` of ZIP carrier `a`
    {"path", "bytes", "sha256"}                   siblings in a list, not keys

The ambiguity is not the shape but the *root*. The recovered Lean bundle keys
its `members` inside the bundle, not from the repository root, and one of them
is called `README.md`. Resolving it here reports a mismatch against a file the
certificate was never talking about -- the same defect this repository already
found and fixed once in its seal checker, which "resolved each declared path by
its last component, so a line naming a file in a subdirectory was compared
against a same-named sibling".

`classify()` settles it from evidence rather than from a hand-kept list: a
container is repository-relative when at least one of its keys resolves here to
the bytes it pins. A container that is repository-relative and has drifted
*entirely* would be reclassified by that rule, so the generated index records
every container's classification and pin count, and the drift comparison fails
by name when one flips.

## What this does NOT establish

A pin is an identity statement and nothing more: these are the bytes the
certificate was produced against. It says nothing about whether the certificate
is correct, whether its mathematics holds, or whether the file is current or
authoritative. A green run means the bytes still match. No claim, premise,
obligation or gate is discharged by anything here, and no status moves.

Run:  python3 tools/pinned_sources_check.py            # recompute and compare
      python3 tools/pinned_sources_check.py --write    # regenerate the index
Every path is resolved when `main()` runs, never at import time.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH_REL = "research"
INDEX_REL = os.path.join("research", "PINNED_SOURCES.md")

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def identity(raw: bytes) -> dict:
    """The house shape, as tools/twelve_project_check.py writes it."""
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def looks_like_a_pin(key: str, value) -> dict | None:
    """The pin this key/value pair states, in the house shape, or None.

    `bytes` is absent from the bare-digest shape and is then not compared; the
    digest alone already fixes the content.
    """
    if isinstance(value, dict) and SHA256.match(str(value.get("sha256", ""))):
        pin = {"sha256": value["sha256"]}
        if isinstance(value.get("bytes"), int):
            pin["bytes"] = value["bytes"]
        return pin
    if isinstance(value, str) and SHA256.match(value) and ("/" in key or "." in key):
        return {"sha256": value}
    return None


def list_record_pin(node) -> tuple[str, dict] | None:
    """A `{path, bytes, sha256}` record, or None.

    Map pins name the file in the key. This shape names it in a sibling field
    of a record, and certificates nest those records in a list. A walker that
    only looks at keys never sees them: `path`, `bytes` and `sha256` are not
    themselves pins, so the whole list is skipped. Extra sibling fields
    (`extraction`, `drive_id`, `module`, `members`) are ignored here and walked
    separately, which is how an archive record can carry both its own path and
    a nested member list.
    """
    if not isinstance(node, dict):
        return None
    path = node.get("path")
    digest = node.get("sha256")
    size = node.get("bytes")
    if not isinstance(path, str) or ("/" not in path and "." not in path):
        return None
    if not isinstance(digest, str) or not SHA256.match(digest):
        return None
    if isinstance(size, bool) or not isinstance(size, int):
        return None
    return path, {"sha256": digest, "bytes": size}


# ---------------------------------------------------------------------------
# Pins declared INSIDE an archive member, and enforced against the tree.
#
# This is the shape the first version of this index missed completely, and the
# miss had a consequence: the index said `research/rn/moment_envelope.py` was not
# pinned, a NON-CERTIFYING label was added to it on that basis, and
# tools/rn_bernstein_sharp_check.py refused the tree on replay. The index whose
# whole purpose is to name the files a certificate binds by digest did not name
# thirteen of them.
#
# These are NOT discovered heuristically. A campaign archive holds a hundred and
# thirty-seven pin-shaped JSON members, and most are manifests of the archive's
# own contents or records of what was in an agent's context -- `context_only_reads`
# and `governance_context` blocks naming CLAUDE.md and README.md at byte counts
# they no longer have, because nothing enforces them and nothing should. Listing
# those as "do not edit" would be a false statement in the one document people
# consult before editing, which is worse than the omission it replaced.
#
# So liveness is taken from the checkers themselves: a declaration is live when a
# tools/ checker passes it to `repository_inputs`, which compares it against the
# tree and refuses. Each row below names that checker, and
# `enforcing_checkers()` + `tests/test_pinned_sources.py` refuse a tools/ checker
# that starts enforcing an archive declaration without appearing here.
#
# (archive, member, key holding repository paths, enforcing checker, count the
#  checker itself asserts)
ARCHIVE_DEPENDENCY_SOURCES = (
    ("research/campaigns/h3_rn_n6_20260920_v1.zip", "verification_plan.json",
     "repository_inputs", "tools/h3_rn_n6_check.py", 25),
    ("research/campaigns/rn_bernstein_sharp_variance_20260921_v1.zip",
     "bernstein/DEPENDENCIES.json", "files", "tools/rn_bernstein_sharp_check.py", 30),
    ("research/campaigns/rn_bernstein_sharp_variance_20260921_v1.zip",
     "sharp_variance/DEPENDENCIES.json", "files", "tools/rn_bernstein_sharp_check.py", 32),
    ("research/campaigns/q0_twelve_20260920_v1.zip", "verification_plan.json",
     "repository_dependencies", "tools/twelve_project_check.py", 24),
)

# A checker may also hold pins in its own source rather than in an archive.
# tools/twelve_project_check.py unions SUPPLEMENTAL_DEPENDENCIES into what it
# enforces, so those paths are pinned by code with no certificate behind them.
CHECKER_HELD_DEPENDENCIES = (
    ("tools/twelve_project_check.py", "SUPPLEMENTAL_DEPENDENCIES"),
)


def dependency_rows(node) -> dict[str, dict]:
    """{path: {bytes, sha256}} from a dict-of-records or a list-of-records.

    The four declarations use both shapes: `bernstein/DEPENDENCIES.json` keys its
    records by path, `sharp_variance/DEPENDENCIES.json` carries a list of
    `{path, bytes, sha256}` rows.
    """
    out: dict[str, dict] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict) and isinstance(value.get("sha256"), str):
                out[value.get("path", key)] = {"bytes": value.get("bytes"),
                                               "sha256": value["sha256"]}
        return out
    if isinstance(node, list):
        for row in node:
            if isinstance(row, dict) and isinstance(row.get("sha256"), str) and row.get("path"):
                out[row["path"]] = {"bytes": row.get("bytes"), "sha256": row["sha256"]}
    return out


def enforcing_checkers(root: str) -> set[str]:
    """tools/ modules that call `repository_inputs(`, i.e. enforce a pin set.

    A literal scan on purpose. The point is to notice a NEW enforcing checker
    that ARCHIVE_DEPENDENCY_SOURCES does not mention, and a scan that reads the
    source sees one the moment it lands. `repository_inputs` is also the name of
    a JSON key, so the call form with the open bracket is what is matched.
    """
    found = set()
    tools_dir = os.path.join(root, "tools")
    for name in sorted(os.listdir(tools_dir)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(tools_dir, name), encoding="utf-8") as handle:
            body = handle.read()
        # `runtime.repository_inputs(...)` is the call form in two of the three
        # campaign checkers, so an attribute access must count. A negative lookbehind
        # excluding `.` found only one of the three and would have declared the table
        # complete while two checkers went unlisted. The JSON KEY of the same name is
        # never followed by an open bracket, so requiring the bracket is enough.
        if re.search(r"repository_inputs\s*\(", body) and name != "pinned_sources_check.py":
            found.add("tools/" + name)
    return found


def archive_declared(root: str) -> tuple[dict[str, dict], list[str], dict]:
    """(files, problems, counts) for every live archive-declared repository pin."""
    files: dict[str, dict] = {}
    problems: list[str] = []
    counts = {"declarations": 0, "declared_sites": 0, "archive_scope": "read"}

    # Scoped to a root that actually holds the campaign tree. The controls for this
    # tool run it against synthetic roots of two or three files; requiring the real
    # campaign archives there would fail every one of them for the wrong reason, and
    # the fix for that must not be to weaken the requirement on the real tree. So the
    # rule is explicit: a root WITH research/campaigns must have every declared
    # archive, and a root without one is a different tree.
    # `test_the_real_tree_reads_its_archive_declarations` asserts this repository is
    # never the second case, so the skip cannot drift onto it.
    if not os.path.isdir(os.path.join(root, "research", "campaigns")):
        counts["archive_scope"] = "no research/campaigns in this root"
        return files, problems, counts

    def add(path: str, pin: dict, named_by: str) -> None:
        counts["declared_sites"] += 1
        rec = files.setdefault(path, {"pin": pin, "certificates": []})
        rec["certificates"].append(named_by)
        if rec["pin"]["sha256"] != pin["sha256"]:
            problems.append(
                f"{path}: two archive declarations pin different bytes -- "
                f"{rec['pin']['sha256'][:16]} and {pin['sha256'][:16]}; one of them "
                f"cannot be the content in this tree")

    for arc, member, key, checker, declared in ARCHIVE_DEPENDENCY_SOURCES:
        full = os.path.join(root, arc)
        if not os.path.isfile(full):
            problems.append(f"{arc}: declared as an archive pin source and absent from "
                            f"this tree, so {checker} cannot be enforcing it")
            continue
        try:
            with zipfile.ZipFile(full) as handle:
                doc = json.loads(handle.read(member))
        except (OSError, KeyError, ValueError) as exc:
            problems.append(f"{arc}::{member}: unreadable as JSON ({exc}); the pins "
                            f"{checker} enforces cannot be listed")
            continue
        if key not in doc:
            problems.append(f"{arc}::{member}: no {key!r} key, so this index cannot see "
                            f"what {checker} enforces")
            continue
        rows = dependency_rows(doc[key])
        if len(rows) != declared:
            problems.append(
                f"{arc}::{member}[{key}]: reads {len(rows)} paths and {checker} asserts "
                f"{declared}. This index must read exactly what the checker reads, or it "
                f"understates what may not be edited")
            continue
        counts["declarations"] += 1
        for path, pin in rows.items():
            add(path, pin, f"{arc}::{member}")

    for checker, attr in CHECKER_HELD_DEPENDENCIES:
        module_path = os.path.join(root, checker)
        if not os.path.isfile(module_path):
            problems.append(f"{checker}: declared as holding {attr} and absent from this tree")
            continue
        try:
            spec = importlib.util.spec_from_file_location("_pin_probe_" + attr, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            held = getattr(module, attr)
        except Exception as exc:                                   # noqa: BLE001
            problems.append(f"{checker}: {attr} could not be read ({type(exc).__name__}: "
                            f"{exc}). A pin set held in code and not readable here is a pin "
                            f"this index cannot report")
            continue
        for path, pin in dependency_rows(held).items():
            add(path, pin, f"{checker} ({attr})")

    if not os.path.isdir(os.path.join(root, "tools")):
        counts["archive_scope"] = "no tools/ in this root"
        return files, problems, counts
    missing = sorted(enforcing_checkers(root)
                     - {row[3] for row in ARCHIVE_DEPENDENCY_SOURCES}
                     - {row[0] for row in CHECKER_HELD_DEPENDENCIES})
    for checker in missing:
        problems.append(
            f"{checker}: enforces a repository pin set against the tree and is named by no "
            f"row in ARCHIVE_DEPENDENCY_SOURCES or CHECKER_HELD_DEPENDENCIES, so whatever "
            f"it pins is absent from this index. That is the exact defect this section "
            f"exists to prevent: add the row before the next replay refuses somebody")
    return files, problems, counts


def certificate_paths(root: str, search_rel: str) -> list[str]:
    """Every JSON under the searched directory, repository-relative, sorted."""
    found = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, search_rel)):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git")]
        for fn in filenames:
            if fn.endswith(".json"):
                found.append(os.path.relpath(os.path.join(dirpath, fn), root))
    return sorted(found)


def pins_in(doc) -> dict[str, dict[str, dict]]:
    """container -> {key: pin} for every pin this certificate states.

    A container is the JSON path of the dict holding the pin, with list indices
    collapsed to `[]` so the eleven `auxiliary/pieces[i]/source_binding/...`
    blocks read as one container rather than eleven. A `{path, bytes, sha256}`
    record is pinned under the path of that record, so siblings in one list
    share a container such as `sources[]`.
    """
    out: dict[str, dict[str, dict]] = {}

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            record = list_record_pin(node)
            if record is not None:
                key, pin = record
                out.setdefault(path, {})[key] = pin
                for k, v in node.items():
                    if k in ("path", "bytes", "sha256"):
                        continue
                    walk(v, f"{path}/{k}" if path else k)
                return
            for k, v in node.items():
                pin = looks_like_a_pin(k, v)
                if pin is not None:
                    out.setdefault(path, {})[k] = pin
                else:
                    walk(v, f"{path}/{k}" if path else k)
        elif isinstance(node, list):
            for v in node:
                walk(v, f"{path}[]")

    walk(doc, "")
    return out


def resolve(root: str, key: str) -> dict | None:
    """The identity of what `key` names in this tree, or None if it names nothing.

    `<archive>::<member>` is read out of the archive, which is how these
    certificates name content sealed inside a ZIP carrier.
    """
    if "::" in key:
        arc, member = key.split("::", 1)
        full = os.path.join(root, arc)
        if not os.path.isfile(full):
            return None
        try:
            with zipfile.ZipFile(full) as handle:
                return identity(handle.read(member))
        except (OSError, KeyError, ValueError):
            return None
    full = os.path.join(root, key)
    if not os.path.isfile(full) or os.path.islink(full):
        return None
    with open(full, "rb") as handle:
        return identity(handle.read())


def classify(root: str, block: dict[str, dict]) -> str:
    """"repository" when at least one key resolves here to the bytes it pins."""
    for key, pin in block.items():
        got = resolve(root, key)
        if got is not None and got["sha256"] == pin["sha256"]:
            return "repository"
    return "other"


def recorded_kinds(index_path: str) -> dict[tuple[str, str], str]:
    """The classification the generated index records, read back out of it.

    The index is the declaration, not a decoration. `classify()` bootstraps it
    from evidence under `--write`; every later run uses what was written, so a
    container whose every pin has drifted keeps being compared and keeps naming
    its files, instead of quietly reclassifying itself out of the check.
    """
    kinds: dict[tuple[str, str], str] = {}
    if not os.path.isfile(index_path):
        return kinds
    row = re.compile(r"^\| `([^`]+)` \| `([^`]+)` \| \d+ \| (repository|other) \|$")
    with open(index_path, encoding="utf-8") as handle:
        for line in handle:
            m = row.match(line.strip())
            if m:
                cert, container, kind = m.groups()
                kinds[(cert, "" if container == "/" else container)] = kind
    return kinds


def survey(root: str, search_rel: str, recorded: dict | None = None):
    """Returns (files, containers, problems, counts)."""
    files: dict[str, dict] = {}
    containers: list[dict] = []
    problems: list[str] = []
    counts = {"certificates": 0, "sites": 0, "other_sites": 0}

    for cert in certificate_paths(root, search_rel):
        try:
            with open(os.path.join(root, cert), encoding="utf-8") as handle:
                doc = json.load(handle)
        except (OSError, ValueError):
            continue
        found = pins_in(doc)
        if not found:
            continue
        counts["certificates"] += 1
        for container in sorted(found):
            block = found[container]
            derived = classify(root, block)
            kind = (recorded or {}).get((cert, container), derived)
            containers.append({"certificate": cert, "container": container,
                               "kind": kind, "derived": derived,
                               "pins": len(block)})
            if kind != "repository":
                counts["other_sites"] += len(block)
                continue
            for key in sorted(block):
                counts["sites"] += 1
                rec = files.setdefault(key, {"pin": block[key], "certificates": []})
                rec["certificates"].append(cert)
                if rec["pin"]["sha256"] != block[key]["sha256"]:
                    problems.append(
                        f"{key}: two certificates pin different bytes -- "
                        f"{rec['pin']['sha256'][:16]} and "
                        f"{block[key]['sha256'][:16]}; one of them cannot be the "
                        f"content in this tree")
    return files, containers, problems, counts


def full_survey(root: str, search_rel: str, recorded: dict | None = None):
    """`survey()` plus the pins declared inside campaign archives.

    ONE entry point for both kinds, used by the CLI and by the controls. They were
    two paths for one commit: `main()` merged the archive declarations and
    `test_the_index_is_what_the_certificates_say` called `survey()` + `render()`
    directly, so the test compared the index against a render that had never seen
    an archive declaration. A second path is exactly how one of two kinds of pin
    ends up unchecked -- which is the defect this whole section exists to repair,
    reproduced in the repair.
    """
    files, containers, problems, counts = survey(root, search_rel, recorded)
    declared, declared_problems, declared_counts = archive_declared(root)
    problems = problems + declared_problems
    counts.update(declared_counts)
    for path, rec in declared.items():
        existing = files.get(path)
        if existing is None:
            files[path] = rec
            continue
        existing["certificates"] += rec["certificates"]
        if existing["pin"]["sha256"] != rec["pin"]["sha256"]:
            problems.append(
                f"{path}: a certificate and an archive declaration pin different bytes -- "
                f"{existing['pin']['sha256'][:16]} and {rec['pin']['sha256'][:16]}; one of "
                f"them cannot be the content in this tree")
    return files, containers, problems, counts


def verify(root: str, files: dict):
    """Compare every repository binding against the content in this tree."""
    problems, matched, unresolved, compared = [], 0, 0, 0
    for path in sorted(files):
        rec = files[path]
        want = rec["pin"]
        got = resolve(root, path)
        named = ", ".join(sorted(set(rec["certificates"])))
        if got is None:
            unresolved += 1
            problems.append(
                f"{path}: pinned by {named} and this tree holds nothing there; "
                f"a replay of that certificate will refuse this tree")
            continue
        compared += 1
        if got["sha256"] != want["sha256"]:
            problems.append(
                f"{path}: pinned at {want['sha256'][:16]} and this tree holds "
                f"{got['sha256'][:16]}. {named} names these bytes as source "
                f"identity, so its replay will refuse this tree. The remedy is a "
                f"re-pin on the lane that owns the certificate, or restoring the "
                f"bytes -- never editing the expected digest to match")
            continue
        if "bytes" in want and got["bytes"] != want["bytes"]:
            problems.append(
                f"{path}: the digest matches and the byte count does not "
                f"({got['bytes']} here, {want['bytes']} pinned); the certificate "
                f"is internally inconsistent")
            continue
        rec["actual"] = got
        matched += 1
    return problems, matched, unresolved, compared


PREAMBLE = """<!-- Generated by tools/pinned_sources_check.py. Do not edit by hand:
     the checker recomputes every row from the certificates and refuses drift. -->

# Repository content a certificate binds by digest

Candidate records under `research/`, and dependency declarations held **inside**
the campaign archives, name repository content by SHA-256 as **source identity**
— "this result was produced against these exact bytes". Editing one of these
files makes the replay checkers that consume the record refuse the tree, by
design.

This page exists because nothing else in the repository said which files those
are. The rejection a replay prints names the dependency and not the certificate,
so the only way to learn a binding was to break it — and three separate changes
did, in four days.

**A fourth did, after this page existed.** Its first version read only the JSON
files on disk under `research/`, so it never saw the declarations sealed inside a
campaign `.zip`. Thirteen pinned files were therefore absent from the index whose
only purpose is to name them, including `research/rn/moment_envelope.py`: a
NON-CERTIFYING label was added to that file *because this page said it was not
pinned*, and `tools/rn_bernstein_sharp_check.py` refused the tree on replay. The
bytes were restored and the archive declarations are now read. A "the only way to
learn a binding was to break it" page that is itself incomplete is worse than no
page, because it is trusted.

**Some of the files below are checkers in `tools/`** — two of them are pinned by
the very campaign they enforce — **and several are documents in `docs/`,
`governance/` and `drive/`.** Counts are in the summary line after the table;
they are computed, not typed.

Before editing anything in this table, read
[`tools/pinned_sources_check.py`](../tools/pinned_sources_check.py). The remedy
for a deliberate change is a re-pin on the lane that owns the certificate, never
a quiet edit to the expected digest — and never a NON-CERTIFYING label, a
docstring fix or a typo correction on the assumption that a small edit is safe.
The digest does not care how small the edit was.

**What a pin is not.** It is an identity statement and nothing more. It says
nothing about whether the certificate is correct, whether its mathematics holds,
or whether the file is current or authoritative. A green run means the bytes
still match. No claim, premise, obligation or gate is discharged by anything on
this page.
"""

HEADER = "| file | bytes | SHA-256 | pinned by |"
RULE = "|---|---:|---|---|"
CHEADER = "| certificate | container | keys | keys are |"
CRULE = "|---|---|---:|---|"


def source_label(name: str) -> str:
    """Short, UNAMBIGUOUS name for whatever pinned a file.

    A bare basename is what the first version used, and it collides: two archives
    each hold a `bernstein/DEPENDENCIES.json`-shaped member, so `DEPENDENCIES.json`
    in the "pinned by" column would name neither of them. An archive declaration
    keeps `<archive>::<member>`; a checker that holds pins in code keeps its own
    path, because that is where a reader has to go.
    """
    if "::" in name:
        arc, member = name.split("::", 1)
        return f"{os.path.basename(arc)}::{member}"
    if name.startswith("tools/"):
        return name
    return os.path.basename(name)


def render(files: dict, containers: list, counts: dict) -> str:
    out = [PREAMBLE, "", HEADER, RULE]
    for path in sorted(files):
        rec = files[path]
        certs = ", ".join(sorted({source_label(c) for c in rec["certificates"]}))
        out.append(f"| `{path}` | {rec['pin'].get('bytes', '—')} | "
                   f"`{rec['pin']['sha256'][:16]}…` | {certs} |")
    out += [
        "",
        f"{len([k for k in files if '::' not in k])} files and "
        f"{len([k for k in files if '::' in k])} members sealed inside a ZIP "
        f"carrier, bound at {counts['sites']} sites across "
        f"{counts['certificates']} certificates, plus "
        f"{counts.get('declared_sites', 0)} sites in "
        f"{counts.get('declarations', 0)} dependency declarations held INSIDE campaign "
        f"archives and enforced by the checkers named below.",
        "",
        "## Every container of pins, and what its keys are relative to",
        "",
        "A container whose keys are **repository** paths is verified above. One",
        "marked **other** keys its entries inside a bundle the certificate",
        "carries, so resolving them here would compare the wrong files; those are",
        "counted and left alone. The classification is read off the evidence, not",
        "a list — see `classify()` — and it is recorded here so that a container",
        "flipping between the two fails this index's drift comparison by name.",
        "",
        CHEADER, CRULE,
    ]
    for c in sorted(containers, key=lambda r: (r["certificate"], r["container"])):
        out.append(f"| `{c['certificate']}` | `{c['container'] or '/'}` | "
                   f"{c['pins']} | {c['kind']} |")
    out += [
        "",
        "## Dependency declarations held inside a campaign archive",
        "",
        "These are not certificates on disk. Each is a JSON member **inside** a",
        "campaign `.zip` that names repository paths by digest, and each is live",
        "because a checker in `tools/` passes it to `repository_inputs`, which",
        "compares it against this tree and refuses. The files they name are in the",
        "table above, so this section says who will refuse and where the digest",
        "came from.",
        "",
        "A campaign archive also holds many pin-*shaped* members that are **not**",
        "pins: manifests of the archive's own contents, and blocks named",
        "`context_only_reads` or `governance_context` recording what an agent had",
        "open — several name `CLAUDE.md` and `README.md` at byte counts they no",
        "longer have, because nothing enforces them and nothing should. Those are",
        "deliberately absent. Listing them as \"do not edit\" would put a false",
        "statement in the one document people read before editing, which is worse",
        "than the omission this section replaced.",
        "",
        "| archive | member | key | enforced by | paths |",
        "|---|---|---|---|---:|",
    ]
    for arc, member, key, checker, declared in ARCHIVE_DEPENDENCY_SOURCES:
        out.append(f"| `{os.path.basename(arc)}` | `{member}` | `{key}` | "
                   f"`{checker}` | {declared} |")
    for checker, attr in CHECKER_HELD_DEPENDENCIES:
        out.append(f"| *(none — held in code)* | — | `{attr}` | `{checker}` | "
                   f"see that file |")
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=REPO)
    parser.add_argument("--search", default=SEARCH_REL,
                        help="directory scanned for certificates")
    parser.add_argument("--index", default=INDEX_REL)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    index_path = os.path.join(root, args.index)

    recorded = {} if args.write else recorded_kinds(index_path)
    files, containers, problems, counts = full_survey(root, args.search, recorded)
    verify_problems, matched, unresolved, compared = verify(root, files)
    problems += verify_problems

    for c in containers:
        where = (c["certificate"], c["container"])
        was = recorded.get(where)
        if was is None and recorded:
            problems.append(
                f"{c['certificate']}: pins under {c['container'] or '/'!r} and "
                f"{args.index} does not record that container; regenerate the "
                f"index with --write so the new binding surface is declared")
        elif was is not None and was != c["derived"]:
            problems.append(
                f"{c['certificate']}: {args.index} records {c['container'] or '/'!r} "
                f"as {was} and the evidence now reads {c['derived']} -- either "
                f"every one of its {c['pins']} pins has drifted, or the container "
                f"changed meaning. The pins are still being compared as {was}; "
                f"read the mismatches above before regenerating")

    if not compared:
        # The floor verify_manifests.py and quarantine_check.py already carry. A
        # run that compared nothing must not read as "every binding holds". The
        # test is `compared`, not `matched`: a run in which every binding was
        # read and every one of them differed is a loud failure, not a vacuous
        # one, and calling it vacuous would bury the mismatches it just found.
        problems.append(
            f"VACUOUS RUN: {counts['certificates']} certificates and "
            f"{len(files)} repository bindings under {args.search}/, and not one "
            f"digest was compared. That is not the same as nothing being wrong")

    want = render(files, containers, counts)

    if args.write:
        for where, was in sorted(recorded_kinds(index_path).items()):
            now = next((c["kind"] for c in containers
                        if (c["certificate"], c["container"]) == where), None)
            if now is not None and now != was:
                print(f"pinned_sources_check: {where[0]}: {where[1] or '/'} moves "
                      f"from {was} to {now}; a repository container becomes `other` "
                      f"only when every one of its pins has drifted")
        with open(index_path, "w", encoding="utf-8") as handle:
            handle.write(want)
        print(f"pinned_sources_check: wrote {args.index} files={len(files)}")
        return 0

    if not os.path.isfile(index_path):
        problems.append(f"{args.index} is missing; run --write")
    else:
        with open(index_path, encoding="utf-8") as handle:
            have = handle.read()
        if have != want:
            for line in sorted(set(have.splitlines()) ^ set(want.splitlines())):
                if line.startswith("|") and not line.startswith("|---"):
                    side = "the index says" if line in have else "the certificates say"
                    problems.append(f"{args.index}: {side} {line.strip()}")
            problems.append(
                f"{args.index} differs from what the certificates say; regenerate "
                f"it with --write rather than editing it")

    for p in problems:
        print(f"pinned_sources_check: {p}")
    print(f"pinned_sources_check: certificates={counts['certificates']} "
          f"containers={len(containers)} "
          f"pinned_files={len([k for k in files if '::' not in k])} "
          f"pinned_archive_members={len([k for k in files if '::' in k])} "
          f"archive_declarations={counts.get('declarations', 0)} "
          f"archive_scope={counts.get('archive_scope', 'read')!r} "
          f"sites={counts['sites']} compared={compared} digest_matches={matched} "
          f"unresolved={unresolved} "
          f"keys_not_repository_relative={counts['other_sites']} "
          f"problems={len(problems)}")
    print("A pass says the content a certificate names is still the content in this tree. "
          "It verifies no mathematics, reviews no certificate and moves no status.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
