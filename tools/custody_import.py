#!/usr/bin/env python3
"""Generate custody supplements from retained third-party custody publications.

Another agent may collect the frozen-object identities this repository cannot
reach and publish them. Those identities are worth having: a second party's
download of the same bytes is the only thing that makes a whole-file freeze
comparable at all once the object was frozen after the accessibility
publication ``drive/inventory.jsonl`` derives from.

What is *not* worth having is a hand-copied transcription of somebody else's
digests. So the publication is retained byte for byte under
``drive/custody/imported/`` and the supplement ``frozen_check`` reads is
generated from it, never typed: run with ``--check`` and the run fails if the
committed supplement has drifted from the retained publication, exactly as
``registers_import.py --check`` fails on a hand-edited export.

Two things this deliberately does not do. It does not relabel the collector: a
row generated from an OpenAI/ChatGPT publication says ``chatgpt``, so a reader
and ``frozen_check`` can both tell how many agreements rest on this
repository's own downloads and how many rest on somebody else's. And it does
not treat retention as verification: this tool checks that the retained file is
the publication it claims to be, and nothing further. Whether the publication's
digests are genuine downloads is not mechanically decidable, here or anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPORTED = os.path.join(ROOT, "drive", "custody", "imported")
CUSTODY = os.path.join(ROOT, "drive", "custody")

# Each retained publication, with the identity this repository measured when it
# landed. A publication whose bytes no longer match is refused, not imported.
PUBLICATIONS = [
    {
        "publication": "FROZEN_CUSTODY_MANIFEST_20260923.json",
        "generated": "2026-09-23_frozen_objects_chatgpt.json",
        "drive_id": "1tCZUSibKFUnXGzcVrmVg7D_CaosEQxYx",
        "bytes": 10044,
        "sha256": "64e54aeaa2725cc81f2b8ce1cb135009292efce420cebbe474a906a4bc6b3341",
        "collected_by": "chatgpt",
        "provider": "OpenAI",
        "landed_utc": "2026-09-26T12:45:07Z",
        "landed_by": "claude",
    },
]


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_publication(spec: dict) -> Tuple[dict, List[str]]:
    """The retained bytes, refused unless they are the identity we landed."""
    path = os.path.join(IMPORTED, spec["publication"])
    problems: List[str] = []
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        return {}, [f"{spec['publication']}: unreadable ({exc})"]
    if len(raw) != spec["bytes"]:
        problems.append(f"{spec['publication']}: {len(raw)} bytes, pinned {spec['bytes']}")
    got = sha256_bytes(raw)
    if got != spec["sha256"]:
        problems.append(f"{spec['publication']}: sha256 {got}, pinned {spec['sha256']}")
    if problems:
        return {}, problems
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return {}, [f"{spec['publication']}: not readable as UTF-8 JSON ({exc})"]
    if not isinstance(doc, dict) or not isinstance(doc.get("observations"), list):
        return {}, [f"{spec['publication']}: no observations list"]
    return doc, problems


def supplement(spec: dict, doc: dict) -> Tuple[dict, List[str]]:
    """The publication's observations, in this repository's custody schema."""
    problems: List[str] = []
    rows = []
    for obs in doc["observations"]:
        rid = str(obs.get("id") or "").strip()
        if not rid:
            problems.append(f"{spec['publication']}: an observation has no Drive id")
            continue
        if not isinstance(obs.get("bytes"), int):
            problems.append(f"{spec['publication']}: {rid} has no integer byte count")
            continue
        sha = str(obs.get("sha256") or "").strip().lower()
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            problems.append(f"{spec['publication']}: {rid} has no 64-hex sha256")
            continue
        note = (
            "Transcribed by tools/custody_import.py from the retained publication, not "
            "downloaded by this repository. The publication records access_status "
            f"{obs.get('access_status')!r} and register_binding_class_before "
            f"{obs.get('register_binding_class_before')!r} for this row."
        )
        rows.append({
            "id": rid,
            "title": obs.get("title"),
            "bytes": obs["bytes"],
            "sha256": sha,
            "frozen_object_id": obs.get("object_id"),
            "verified_utc": obs.get("verified_utc"),
            "verified_by": spec["collected_by"],
            "note": note,
        })
    out = {
        "_what_this_is": (
            f"Generated from {spec['publication']}, retained byte for byte under "
            "drive/custody/imported/. Regenerate with tools/custody_import.py; never edit "
            "this file, because tools/custody_import.py --check fails on drift."
        ),
        "_not_an_inventory": (
            "This is NOT a Drive inventory and must never be fed to a path-search consumer. "
            "It carries no paths, no folder structure and no coverage claim."
        ),
        "_the_collector_is_not_this_repository": (
            f"Every row here was collected by {spec['collected_by']} ({spec['provider']}) and "
            "says so in verified_by. frozen_check counts a row two DISTINCT collectors agree on "
            "apart from a row only one collected, and neither count is a MATCH: agreement "
            "between collectors is evidence about bytes, never a stronger status, and it earns "
            "no organizational-independence credit, which is a question about review and not "
            "about custody of a digest."
        ),
        "collected_by": spec["collected_by"],
        "provider": spec["provider"],
        "collection_method": (
            f"Not collected here. Drive object {spec['drive_id']} was downloaded on "
            f"{spec['landed_utc']} by {spec['landed_by']}, measured at {spec['bytes']} bytes and "
            f"sha256 {spec['sha256']}, and retained at "
            f"drive/custody/imported/{spec['publication']}. This file is that publication's "
            "observations mapped into this repository's custody schema by "
            "tools/custody_import.py. The publication states for itself that each object was "
            "downloaded through authenticated Drive and checked against the register; this "
            "repository did not witness those downloads and cannot."
        ),
        "source_publication": {
            "file": spec["publication"],
            "drive_id": spec["drive_id"],
            "bytes": spec["bytes"],
            "sha256": spec["sha256"],
            "publication_id": doc.get("publication_id"),
            "author": doc.get("author"),
            "count": doc.get("count"),
            "scope": doc.get("scope"),
        },
        "does_not_establish": (
            "Retaining a publication and mapping its rows establishes what that publication "
            "said, and that the retained bytes are the bytes measured when it landed. It does "
            "not establish that the downloads behind it happened, that any object is what the "
            "register says it is about, or that anything either claims mathematically holds. It "
            "re-freezes nothing, verifies no body, moves no gate, releases no obligation and "
            "earns zero organizational-independence credit."
        ),
        "rows": rows,
    }
    return out, problems


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="fail if a committed supplement differs from its publication")
    ap.add_argument("--out-dir", default=CUSTODY)
    args = ap.parse_args(argv)

    problems: List[str] = []
    written = 0
    drifted = 0
    rows_total = 0
    for spec in PUBLICATIONS:
        doc, probs = read_publication(spec)
        problems.extend(probs)
        if not doc:
            continue
        gen, probs = supplement(spec, doc)
        problems.extend(probs)
        rows_total += len(gen["rows"])
        target = os.path.join(args.out_dir, spec["generated"])
        text = render(gen)
        if args.check:
            try:
                with open(target, encoding="utf-8") as f:
                    have = f.read()
            except OSError as exc:
                problems.append(f"{spec['generated']}: unreadable ({exc})")
                drifted += 1
                continue
            if have != text:
                problems.append(f"{spec['generated']}: differs from {spec['publication']}; "
                                f"regenerate with tools/custody_import.py")
                drifted += 1
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(text)
            written += 1

    for p in problems:
        print(f"PROBLEM  {p}")
    verb = f"checked={len(PUBLICATIONS)}" if args.check else f"wrote={written}"
    print(f"custody_import: publications={len(PUBLICATIONS)} {verb} "
          f"rows={rows_total} drift={drifted} problems={len(problems)}")
    print("An imported publication is another party's record of bytes. It is retained and mapped "
          "here, never witnessed here: no row generated from one is a MATCH, and none of it "
          "verifies a body, moves a gate or earns independence credit.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
