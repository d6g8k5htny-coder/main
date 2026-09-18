#!/usr/bin/env python3
"""Identity check for every repository artifact that mirrors a Drive object.

OP-PROT-019 §2: "The digest establishes identity, not truth or authorization;
compare against live routing and retained source bytes."

This repository holds reading copies, not objects. That is legitimate — a git
repository cannot hold a Google Docs native body — but it must never be possible
to mistake one for the other. The nonauthor review of RV-OPS-R17 found exactly
that mistake in this repository's own migration:
``governance/protocols/OP-PROT-019-v1.1_R17.md`` has the same byte count as the
object the register names (14,073) and a different SHA-256, so a byte-count check
confirms the wrong bytes.

This checker enforces four things:

1. every record in ``governance/PROVENANCE.json`` still matches the file on disk,
   so the record cannot silently drift from what it describes;
2. no record claims ``byte_exact`` unless a full 64-hex declared digest is present
   and actually matches;
3. the dangerous case — byte count matches while the digest does not — is called
   out by name rather than passing quietly;
4. no mirrored artifact is described anywhere in the repository as "verbatim" or
   "byte-exact" unless its record says it is.

Rule 4 is the one that matters most in practice. The word is the defect: two
ported reports carried a provenance header reading "Ported verbatim", and that
header was itself what made the file non-verbatim.

Exit code 0 when clean, 1 otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROVENANCE = os.path.join(ROOT, "governance", "PROVENANCE.json")

# Claims of exactness we refuse to let stand on a copy that is not exact.
#
# This must match a claim ABOUT A COPY IN THIS REPOSITORY, not the corpus's own
# discussion of byte-exactness as subject matter — "the byte-exact TB-G2 algebra
# capsules" is a description of Drive objects and is none of this checker's
# business. So the pattern requires an exactness word joined to a porting verb.
EXACTNESS_CLAIM = re.compile(
    r"\b("
    r"(ported|mirrored|copied|reproduced|carried|transcribed)\s+"
    r"(here\s+)?(verbatim|byte-exact|byte\s+exact|byte-identical)"
    r"|"
    r"(verbatim|byte-exact|byte-identical)\s+(port|copy|mirror|reproduction)"
    r")\b",
    re.I,
)

# Where such a claim would be made about a mirrored artifact.
PROSE_ROOTS = ("docs", "governance", "registers", "research", "claims", "engine", "reviews")


def load(path: str | None = None) -> dict:
    # Resolve the module global at call time so tests can point the checker at a
    # different record without the default argument freezing it at import.
    with open(path or PROVENANCE, encoding="utf-8") as f:
        return json.load(f)


def sha256_of(path: str) -> tuple[str, int]:
    with open(path, "rb") as f:
        b = f.read()
    return hashlib.sha256(b).hexdigest(), len(b)


def prose_files(root: str) -> list[str]:
    out = []
    for sub in PROSE_ROOTS:
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = [x for x in dirnames if x not in ("__pycache__", ".git")]
            for fn in filenames:
                if fn.endswith((".md", ".py", ".json")):
                    out.append(os.path.join(dirpath, fn))
    for fn in ("README.md", "CLAUDE.md", "CONTRIBUTING.md"):
        p = os.path.join(root, fn)
        if os.path.isfile(p):
            out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    global PROVENANCE
    argv = argv if argv is not None else sys.argv[1:]
    root = ROOT
    if len(argv) >= 2 and argv[0] == "--record":
        PROVENANCE = argv[1]
        argv = argv[2:]
    if len(argv) >= 2 and argv[0] == "--root":
        root = argv[1]

    rec = load()
    arts = rec.get("artifacts", [])
    problems: list[str] = []
    count_only: list[str] = []

    by_path = {}
    for a in arts:
        p = a.get("path")
        if not p:
            problems.append("a record has no path")
            continue
        if p in by_path:
            problems.append(f"{p}: duplicate provenance record")
        by_path[p] = a

        full = os.path.join(root, p)
        if not os.path.isfile(full):
            problems.append(f"{p}: recorded but missing from the repository")
            continue

        digest, nbytes = sha256_of(full)
        if a.get("repo_sha256") != digest:
            problems.append(
                f"{p}: record says repo_sha256 {str(a.get('repo_sha256'))[:16]}… "
                f"but the file hashes to {digest[:16]}… — the record has drifted"
            )
        if a.get("repo_bytes") != nbytes:
            problems.append(
                f"{p}: record says {a.get('repo_bytes')} bytes, file is {nbytes}"
            )

        declared = a.get("source_declared_sha256")
        claims_exact = bool(a.get("byte_exact"))
        if claims_exact:
            if not declared or len(declared) < 64:
                problems.append(
                    f"{p}: claims byte_exact without a full 64-hex declared digest"
                )
            elif declared != digest:
                problems.append(
                    f"{p}: claims byte_exact but the declared digest does not match"
                )

        # The trap the RV-OPS-R17 review found.
        if a.get("byte_count_matches") and not claims_exact:
            count_only.append(p)

    # Rule 4: no exactness claim in prose about a non-exact mirrored artifact.
    mirrored = {p for p, a in by_path.items() if not a.get("byte_exact")}
    basenames = {os.path.basename(p): p for p in mirrored}
    for f in prose_files(root):
        rel = os.path.relpath(f, root)
        try:
            with open(f, encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if not EXACTNESS_CLAIM.search(line):
                continue
            # Which non-exact artifact is this line talking about? Either it
            # names one, or the line sits inside one and is describing itself
            # (which is how the "Ported verbatim" headers got in).
            named = [p for bn, p in basenames.items() if bn in line]
            if not named and rel in mirrored and not rel.startswith("registers/source/"):
                # The self-reference fallback is deliberately not applied to bulk
                # exported source data. A row inside the register export that says
                # some Drive artifact was reproduced verbatim is the export's
                # content describing a third object — not the export claiming its
                # own exactness — and that file may not be edited anyway.
                named.append(rel)
            for p in set(named):
                problems.append(
                    f"{rel}:{lineno}: claims exactness for {p}, which is not byte-exact "
                    f"— {line.strip()[:90]}"
                )

    for p in sorted(set(count_only)):
        print(f"  BYTE COUNT MATCHES, DIGEST DOES NOT: {p}")
        print("     a byte-count check on this file confirms the wrong bytes")

    for p in problems:
        print(f"  PROBLEM {p}")

    print(
        f"provenance_check: artifacts={len(arts)} byte_exact="
        f"{sum(1 for a in arts if a.get('byte_exact'))} "
        f"count_only_traps={len(set(count_only))} problems={len(problems)}"
    )
    print(
        "provenance_check: these are reading copies. Reviewing one is not reviewing "
        "the object the registers name."
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
