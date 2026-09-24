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

## Three shapes, and one genuine ambiguity

A tool that knows only one shape under-reports the surface:

    path   -> {"bytes": 37974, "sha256": "58ebe18f..."}
    path   -> "58ebe18f..."                       a bare digest string
    a::b   -> {...}                               member `b` of ZIP carrier `a`

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
import json
import os
import re
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
    blocks read as one container rather than eleven.
    """
    out: dict[str, dict[str, dict]] = {}

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
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

Several candidate records under `research/` name repository content by SHA-256
as **source identity** — "this result was produced against these exact bytes".
Editing one of these files makes the replay checkers that consume the
certificate refuse the tree, by design.

This page exists because nothing else in the repository said which files those
are. The rejection a replay prints names the dependency and not the certificate,
so the only way to learn a binding was to break it — and three separate changes
did, in four days. **Two of the files below are checkers in `tools/` and one is
a document in `docs/`.**

Before editing anything in this table, read
[`tools/pinned_sources_check.py`](../tools/pinned_sources_check.py). The remedy
for a deliberate change is a re-pin on the lane that owns the certificate, never
a quiet edit to the expected digest.

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


def render(files: dict, containers: list, counts: dict) -> str:
    out = [PREAMBLE, "", HEADER, RULE]
    for path in sorted(files):
        rec = files[path]
        certs = ", ".join(sorted({os.path.basename(c) for c in rec["certificates"]}))
        out.append(f"| `{path}` | {rec['pin'].get('bytes', '—')} | "
                   f"`{rec['pin']['sha256'][:16]}…` | {certs} |")
    out += [
        "",
        f"{len([k for k in files if '::' not in k])} files and "
        f"{len([k for k in files if '::' in k])} members sealed inside a ZIP "
        f"carrier, bound at {counts['sites']} sites across "
        f"{counts['certificates']} certificates.",
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
    files, containers, problems, counts = survey(root, args.search, recorded)
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
          f"sites={counts['sites']} compared={compared} digest_matches={matched} "
          f"unresolved={unresolved} "
          f"keys_not_repository_relative={counts['other_sites']} "
          f"problems={len(problems)}")
    print("A pass says the content a certificate names is still the content in this tree. "
          "It verifies no mathematics, reviews no certificate and moves no status.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
