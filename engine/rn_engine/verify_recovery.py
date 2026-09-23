"""Verify every payload recovered under ``engine/rn_engine/frozen`` -- with
negative controls that fire when a digest, a byte or the reconstruction rule is
weakened.

Four independent checks, each of which must pass:

  A. blob integrity      -- every file named in ``BINDING.json`` hashes to the
                            digest and byte count recorded there;
  B. source-map agreement -- each recorded digest appears in
                            ``drive/source_map/Archive_Members.csv`` (or
                            ``Files.csv``), with the same byte count, and EVERY
                            occurrence of that payload in the source map agrees
                            on the digest;
  C. engine self-pin     -- the recovered ``d3_rn_unif.py`` contains a ``_PINS``
                            block naming six dependencies by SHA-256; the
                            digests of the recovered dependency blobs must equal
                            those pins. This check is independent of the source
                            map: it is the engine's own fail-closed contract,
                            read back out of the recovered bytes;
  D. reconstruction rule -- the de-transform in ``reconstruct.py`` is exercised
                            against deliberately broken variants and must reject
                            each one.

Run::

    python3 engine/rn_engine/verify_recovery.py            # A, B, C
    python3 engine/rn_engine/verify_recovery.py --selftest # A, B, C, D

Exit status 0 means every check passed.

WHAT THIS SCRIPT DOES NOT ESTABLISH
-----------------------------------
Everything it checks is about BYTES. A green run says the frozen carriers in
this directory are the ones the source map names, nothing more. It does not run
the engine, does not invoke its certifier, does not validate a single
mathematical step, and promotes, closes, discharges and reclassifies nothing.
``D3-LEMMA-RN-UNIF`` Piece 1 and Piece 2 remain OPEN; ``OBL-H5-JETMOD``,
``OBL-H5-ZBAND``, ``OBL-H5-REMOTE-THRESHOLD`` and ``OBL-D1-PROMOTE`` remain
OPEN. Original prize problems solved: 0.

The digests it compares are SHA-256 over exact bytes. No floating point is
involved anywhere in this file, so nothing here is a float-path result; the
frozen engine it verifies, by contrast, is mpmath floating point throughout and
is labelled NON-CERTIFYING in ``BINDING.json`` and ``docs/ENGINE_RECOVERY.md``.

Standard library only. Python 3.11.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BINDING = os.path.join(HERE, "BINDING.json")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_binding() -> dict:
    with open(BINDING, "r", encoding="utf-8") as fh:
        return json.load(fh)


def check_blobs(doc: dict) -> list[str]:
    """A. every blob hashes to its recorded digest and byte count."""
    problems = []
    for rec in doc["carriers"]:
        path = os.path.join(ROOT, rec["blob_path"])
        if not os.path.exists(path):
            problems.append("%s: missing blob %s" % (rec["carrier_id"], rec["blob_path"]))
            continue
        with open(path, "rb") as fh:
            data = fh.read()
        if len(data) != rec["bytes"]:
            problems.append("%s: %d bytes on disk, %d recorded"
                            % (rec["carrier_id"], len(data), rec["bytes"]))
        got = _sha(data)
        if got != rec["sha256"]:
            problems.append("%s: sha256 %s on disk, %s recorded"
                            % (rec["carrier_id"], got, rec["sha256"]))
    return problems


def _source_map_rows():
    am = os.path.join(ROOT, "drive", "source_map", "Archive_Members.csv")
    fi = os.path.join(ROOT, "drive", "source_map", "Files.csv")
    members, files = [], []
    if os.path.exists(am):
        with open(am, newline="", encoding="utf-8") as fh:
            members = list(csv.DictReader(fh))
    if os.path.exists(fi):
        with open(fi, newline="", encoding="utf-8") as fh:
            files = list(csv.DictReader(fh))
    return members, files


def check_source_map(doc: dict) -> list[str]:
    """B. digests and byte counts agree with the exported source map."""
    members, files = _source_map_rows()
    if not members and not files:
        return ["drive/source_map/ not readable -- source-map agreement NOT checked"]
    problems = []
    for rec in doc["carriers"]:
        digest = rec["sha256"]
        mrows = [r for r in members if r["Payload SHA-256"] == digest]
        frows = [r for r in files if r["Source SHA-256"] == digest]
        if not mrows and not frows:
            problems.append("%s: digest %s not in the source map"
                            % (rec["carrier_id"], digest))
            continue
        for r in mrows:
            if int(r["Bytes"]) != rec["bytes"]:
                problems.append("%s: Archive_Members byte count %s != %d"
                                % (rec["carrier_id"], r["Bytes"], rec["bytes"]))
        for r in frows:
            if int(r["Bytes"]) != rec["bytes"]:
                problems.append("%s: Files byte count %s != %d"
                                % (rec["carrier_id"], r["Bytes"], rec["bytes"]))
        # every same-named member occurrence must carry the same payload digest
        names = {r["Member path"].rsplit("/", 1)[-1] for r in mrows}
        for name in names:
            same = [r for r in members if r["Member path"].rsplit("/", 1)[-1] == name]
            digests = {r["Payload SHA-256"] for r in same}
            if len(digests) != 1:
                problems.append("%s: member name %r carries %d distinct payload "
                                "digests in the source map -- ambiguous, do not "
                                "treat as recovered" % (rec["carrier_id"], name, len(digests)))
    return problems


_PIN_LINE = re.compile(r"^\s*'([^']+)':\s*'([0-9a-f]{64})',\s*$")


def engine_pins(engine_source: str) -> dict[str, str]:
    """Parse the ``_PINS = { ... }`` block out of the recovered engine source."""
    pins: dict[str, str] = {}
    inside = False
    for line in engine_source.split("\n"):
        if line.startswith("_PINS = {"):
            inside = True
            continue
        if inside:
            if line.startswith("}"):
                break
            m = _PIN_LINE.match(line)
            if m:
                pins[m.group(1)] = m.group(2)
    return pins


def check_engine_pins(doc: dict) -> list[str]:
    """C. the engine's own hash pins match the recovered dependency blobs."""
    engine = next((r for r in doc["carriers"]
                   if r["blob_path"].endswith("d3_rn_unif.py")), None)
    if engine is None:
        return ["no d3_rn_unif.py record in BINDING.json"]
    with open(os.path.join(ROOT, engine["blob_path"]), "r", encoding="utf-8") as fh:
        pins = engine_pins(fh.read())
    if len(pins) != 6:
        return ["expected 6 hash pins in the recovered engine, found %d" % len(pins)]
    have = {os.path.basename(r["blob_path"]): r["sha256"] for r in doc["carriers"]}
    problems = []
    for pinned_path, pinned_digest in sorted(pins.items()):
        base = os.path.basename(pinned_path)
        if base not in have:
            problems.append("engine pins %s (%s) but no such blob is bound"
                            % (pinned_path, pinned_digest[:12]))
        elif have[base] != pinned_digest:
            problems.append("engine pins %s = %s but the bound blob is %s"
                            % (pinned_path, pinned_digest, have[base]))
    return problems


# ---------------------------------------------------------------- controls --

def _sample_volume() -> tuple[list[str], bytes]:
    """A small synthetic reading volume plus the payload it must reconstruct to.

    Built rather than downloaded so the controls run offline and in CI.
    """
    payload = b"line one\n\nline two\n\n\nline three"          # no trailing newline
    rendered = []
    for i, part in enumerate(payload.decode().split("\n")):
        rendered.append(part)
    # apply the display transform: double every maximal blank run
    doubled: list[str] = []
    i = 0
    while i < len(rendered):
        if rendered[i] == "":
            j = i
            while j < len(rendered) and rendered[j] == "":
                j += 1
            doubled.extend([""] * (2 * (j - i)))
            i = j
        else:
            doubled.append(rendered[i])
            i += 1
    return doubled, payload


def run_controls() -> list[str]:
    """D. the reconstruction rule must REJECT each deliberate break."""
    import base64 as _b64
    import tempfile
    sys.path.insert(0, HERE)
    import reconstruct  # noqa: E402  (local module, stdlib only)

    doubled, payload = _sample_volume()
    digest = _sha(payload)
    head = [
        "ACCESS READING VOLUME SYNTHETIC",
        "",
        "",
        "BEGIN SOURCE %s PART 1/1" % digest,
        "Name: synthetic/control.txt",
        "Original bytes: %d" % len(payload),
        "Source SHA-256: %s" % digest,
        "Display encoding: UTF-8. Display line endings normalized.",
        "CONTENT START %s-part1" % digest,
    ]
    tail = ["CONTENT END %s-part1" % digest, "END SOURCE", ""]

    def as_volume(body: list[str]) -> str:
        text = "\r\n".join(head + body + tail)
        raw = b"\xef\xbb\xbf" + text.encode("utf-8")
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"content": _b64.b64encode(raw).decode(),
                       "id": "synthetic", "mimeType": "text/plain",
                       "title": "synthetic"}, fh)
        return path

    failures = []

    # control 0 (positive): the honest volume must reconstruct exactly.
    path = as_volume(doubled)
    try:
        got = reconstruct.extract(path, digest)
        if got != payload:
            failures.append("CONTROL-0 positive: reconstruction differs from the payload")
    except Exception as exc:                                    # pragma: no cover
        failures.append("CONTROL-0 positive: unexpectedly raised %r" % (exc,))
    finally:
        os.unlink(path)

    # control 1: one byte of the rendered content flipped -> must raise.
    broken = list(doubled)
    broken[0] = broken[0].replace("line one", "1ine one")
    path = as_volume(broken)
    try:
        reconstruct.extract(path, digest)
        failures.append("CONTROL-1 byte flip: reconstruction accepted altered content")
    except ValueError:
        pass
    finally:
        os.unlink(path)

    # control 2: blank-line runs NOT doubled by the exporter, i.e. the halving
    # rule is wrong for this volume -> must raise rather than silently halve.
    path = as_volume([l for l in _sample_volume()[1].decode().split("\n")])
    try:
        reconstruct.extract(path, digest)
        failures.append("CONTROL-2 un-doubled blanks: reconstruction accepted a "
                        "volume the halving rule does not fit")
    except ValueError:
        pass
    finally:
        os.unlink(path)

    # control 3: a spurious blank PAIR at the end -- one extra real blank line
    # after halving -> must raise.
    #
    # Note the boundary deliberately: appending a SINGLE blank line does NOT
    # change the reconstruction, because halving an odd run rounds down. The
    # rule is not injective on odd blank runs, so it can fail to invert a
    # volume -- but it can never invent a payload, since extract() returns only
    # on a SHA-256 match. Control 3b pins that documented insensitivity down so
    # a future change to the rule cannot alter it unnoticed.
    path = as_volume(doubled + ["", ""])
    try:
        reconstruct.extract(path, digest)
        failures.append("CONTROL-3 spurious blank pair: reconstruction accepted "
                        "a volume carrying one extra real blank line")
    except ValueError:
        pass
    finally:
        os.unlink(path)

    # control 3b (documented boundary): one odd trailing blank is absorbed.
    path = as_volume(doubled + [""])
    try:
        if reconstruct.extract(path, digest) != payload:
            failures.append("CONTROL-3b: odd trailing blank changed the payload")
    except Exception as exc:
        failures.append("CONTROL-3b: odd trailing blank now rejected (%r) -- the "
                        "halving rule changed; re-verify every recovered digest"
                        % (exc,))
    finally:
        os.unlink(path)

    # control 4: the digest we ask for is not in the volume -> must raise.
    path = as_volume(doubled)
    try:
        reconstruct.extract(path, "0" * 64)
        failures.append("CONTROL-4 wrong digest: reconstruction returned bytes for "
                        "a digest the volume does not carry")
    except KeyError:
        pass
    finally:
        os.unlink(path)

    # control 5: a weakened BINDING.json (one digit of one digest changed) must
    # be caught by check_blobs.
    doc = load_binding()
    victim = doc["carriers"][0]
    original = victim["sha256"]
    victim["sha256"] = ("0" if original[0] != "0" else "1") + original[1:]
    if not check_blobs(doc):
        failures.append("CONTROL-5 weakened digest: check_blobs accepted a "
                        "BINDING.json whose recorded digest was altered")
    victim["sha256"] = original

    # control 6: a weakened BINDING.json byte count must be caught too.
    victim["bytes"] = victim["bytes"] - 1
    if not check_blobs(doc):
        failures.append("CONTROL-6 weakened byte count: check_blobs accepted a "
                        "BINDING.json whose recorded byte count was altered")
    victim["bytes"] = victim["bytes"] + 1

    # control 7: a mutated engine pin must be caught by check_engine_pins.
    doc2 = load_binding()
    dep = next(r for r in doc2["carriers"] if r["blob_path"].endswith("cov_exact.py"))
    dep["sha256"] = "f" * 64
    if not check_engine_pins(doc2):
        failures.append("CONTROL-7 mutated pin: check_engine_pins accepted a "
                        "dependency digest that contradicts the engine's own _PINS")
    return failures


def main(argv: list[str]) -> int:
    doc = load_binding()
    sections = [("A blob integrity", check_blobs(doc)),
                ("B source-map agreement", check_source_map(doc)),
                ("C engine self-pin", check_engine_pins(doc))]
    if "--selftest" in argv:
        sections.append(("D negative controls", run_controls()))
    bad = 0
    for name, problems in sections:
        if problems:
            bad += len(problems)
            print("FAIL  %s" % name)
            for p in problems:
                print("        %s" % p)
        else:
            print("ok    %s" % name)
    print()
    print("%d carriers bound, %d problems" % (len(doc["carriers"]), bad))
    print("This verifies BYTES only. It establishes no mathematical claim and "
          "promotes no status: D3-LEMMA-RN-UNIF Pieces 1 and 2 remain OPEN.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
