"""Reconstruct an archived payload from a Drive ACCESS READING VOLUME export.

WHY THIS FILE EXISTS
--------------------
``LANE_RN_UNIF.md`` caveat 5 records that the frozen engine ``d3_rn_unif.py``
"is not in the RN_UNIF Drive folder; scripts import/monkey-patch it from tree --
not downloaded here", which leaves ``rnu_ds3.py`` -- the PREFERRED carrier for
the A5 lane -- un-runnable. ``drive/source_map/Archive_Members.csv`` resolves the
engine to a payload digest and a *reading copy*, and this module is the exact,
reproducible transform from that reading copy back to the archived payload.

WHAT A READING COPY ACTUALLY IS
-------------------------------
It is **not** a per-file copy. Drive doc ``1zaumNVU...`` is a Google Doc titled
``READING RESEARCH_SOURCE_CHECK_STATUS 129`` whose ``text/plain`` export is a
220,170-byte *volume* holding fourteen unrelated sources, each wrapped in::

    BEGIN SOURCE <sha256> PART 1/1
    Name: <member path>
    Original bytes: <n>
    Source SHA-256: <sha256>
    Source link: <drive link>
    Source path: <path>
    Context: ...
    Display encoding: UTF-8. Display line endings normalized; raw source
    identity remains above.
    CONTENT START <sha256>-part1
    ... the source text, as Google Docs renders it ...
    CONTENT END <sha256>-part1
    END SOURCE

The volume announces its own lossiness in that "Display ..." banner. Observed
transformations, all of them reversible for the payloads recovered here:

  1. a UTF-8 BOM (``EF BB BF``) at the head of the export;
  2. every line ending rewritten as CRLF;
  3. every maximal run of blank lines DOUBLED (Google Docs renders one source
     newline as one empty paragraph plus the paragraph break);
  4. a trailing newline belonging to the volume, not to the member.

NOT ALWAYS REVERSIBLE -- READ THIS BEFORE REUSING THE MODULE
------------------------------------------------------------
The same volume is demonstrably LOSSY for other members. In the very first
member of doc ``1zaumNVU...`` (``C097_COLLAR_GAMMA_BUDGET.md``) the LaTeX source
``\boxed{`` appears in the export as byte 0x08 followed by ``oxed{`` and
``\frac`` appears as byte 0x0C followed by ``rac`` -- backslash escapes were
interpreted somewhere in the export path, which destroys content irrecoverably.
So this module is **not** a general-purpose un-exporter. It is only ever
trustworthy because :func:`extract` refuses to return anything whose SHA-256
does not equal the digest the source map records. The digest is the proof; the
transform is only a hypothesis about how the bytes were mangled.

WHAT THIS MODULE DOES NOT ESTABLISH
-----------------------------------
* Nothing mathematical. Recovering a carrier byte-for-byte is provenance, not a
  bound and not a proof. ``D3-LEMMA-RN-UNIF`` Piece 1 and Piece 2 both remain
  OPEN, as do ``OBL-H5-JETMOD``, ``OBL-H5-ZBAND``, ``OBL-H5-REMOTE-THRESHOLD``
  and ``OBL-D1-PROMOTE``. No status here changes.
* It does not validate the recovered code, run it, or endorse any number it
  would print.
* It certifies no *other* member of any reading volume. A digest match is a
  statement about one payload only.

Standard library only (``json``, ``base64``, ``hashlib``). Python 3.11.
"""
from __future__ import annotations

import base64
import hashlib
import json

__all__ = ["volume_bytes", "members", "dehydrate", "extract"]


def volume_bytes(tool_result_path: str) -> bytes:
    """Decode the base64 ``content`` field of a Drive download result."""
    with open(tool_result_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return base64.b64decode(payload["content"])


def members(raw: bytes) -> dict[str, tuple[str, int, list[str]]]:
    """Split a reading volume into ``{sha256: (name, declared_bytes, lines)}``.

    ``lines`` are the rendered content lines strictly between the volume's
    ``CONTENT START`` and ``CONTENT END`` markers, with the BOM stripped and
    CRLF already normalised to LF.
    """
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    lines = text.split("\n")
    out: dict[str, tuple[str, int, list[str]]] = {}
    i = 0
    while i < len(lines):
        if not lines[i].startswith("BEGIN SOURCE "):
            i += 1
            continue
        digest = lines[i].split()[2]
        start = end = None
        j = i
        while j < len(lines):
            if lines[j].startswith("CONTENT START "):
                start = j + 1
            if lines[j].startswith("CONTENT END "):
                end = j
                break
            j += 1
        if start is None or end is None:
            break
        head = lines[i:start]
        name = next((l[len("Name: "):] for l in head if l.startswith("Name: ")), "?")
        declared = next((int(l.split(": ", 1)[1]) for l in head
                         if l.startswith("Original bytes: ")), -1)
        out[digest] = (name, declared, lines[start:end])
        i = end
    return out


def dehydrate(body_lines: list[str]) -> bytes:
    """Undo the display transform: halve every maximal run of blank lines.

    Google Docs renders one source newline as two export newlines, so a run of
    ``2k`` blank lines in the export corresponds to ``k`` blank lines in the
    source. The result is joined with LF and carries **no** trailing newline;
    the volume's own trailing newline is not part of the member.
    """
    out: list[str] = []
    i = 0
    while i < len(body_lines):
        if body_lines[i] == "":
            j = i
            while j < len(body_lines) and body_lines[j] == "":
                j += 1
            out.extend([""] * ((j - i) // 2))
            i = j
        else:
            out.append(body_lines[i])
            i += 1
    return "\n".join(out).encode("utf-8")


def extract(tool_result_path: str, want_sha256: str) -> bytes:
    """Return the payload with digest ``want_sha256``, or raise.

    The SHA-256 check is the whole guarantee. If the reconstruction hypothesis
    is wrong for this member -- because the export mangled a backslash escape,
    or because the member was split across parts -- this raises rather than
    returning plausible-looking bytes.
    """
    raw = volume_bytes(tool_result_path)
    found = members(raw)
    if want_sha256 not in found:
        raise KeyError(
            "digest %s is not a member of this reading volume (members: %s)"
            % (want_sha256, ", ".join(sorted(found))))
    name, declared, body = found[want_sha256]
    data = dehydrate(body)
    got = hashlib.sha256(data).hexdigest()
    if got != want_sha256:
        raise ValueError(
            "reconstruction of %r does not match the recorded payload: "
            "want %s, got %s (%d bytes reconstructed, %d declared). "
            "DO NOT store these bytes as the payload."
            % (name, want_sha256, got, len(data), declared))
    if declared >= 0 and declared != len(data):
        raise ValueError(
            "digest matched but the declared byte count disagrees: %d vs %d"
            % (declared, len(data)))
    return data
