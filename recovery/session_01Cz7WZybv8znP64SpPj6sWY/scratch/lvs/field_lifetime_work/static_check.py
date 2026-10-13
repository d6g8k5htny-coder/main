"""Static checks for statement-only Lean modules (no Lean needed).
NON-CERTIFYING: a lexical check, not an elaboration."""
import json, re, sys, hashlib, pathlib

PKT = pathlib.Path(sys.argv[1])
MODS = [pathlib.Path(p) for p in sys.argv[2:]]
SRC = json.loads((PKT / "sources" / "SOURCES.json").read_text())
by_sha = {}
for s in SRC["sources"]:
    b = (PKT / s["local_copy"]).read_bytes()
    assert hashlib.sha256(b).hexdigest() == s["sha256"], s["id"]
    assert len(b) == s["bytes"], s["id"]
    by_sha[s["sha256"]] = (s["id"], b)

FORBID_WORDS = ["theorem","lemma","example","instance","axiom","sorry","native_decide",
                "unsafe","opaque","partial","macro","notation","set_option"]
FORBID_SUB = ["verified","proved","closed","certified"]
fail = 0
for m in MODS:
    txt = m.read_text(encoding="utf-8")
    print(f"== {m.name}: {len(txt.encode())} bytes")
    # 1. forbidden tokens (case-sensitive word boundary) and substrings (case-insensitive)
    for w in FORBID_WORDS:
        for mm in re.finditer(r"(?<![A-Za-z0-9_.])" + re.escape(w) + r"(?![A-Za-z0-9_])", txt):
            ln = txt.count("\n", 0, mm.start()) + 1
            print(f"  FORBIDDEN token '{w}' at line {ln}"); fail += 1
        for mm in re.finditer(re.escape(w), txt, flags=re.I):
            ln = txt.count("\n", 0, mm.start()) + 1
            line = txt.splitlines()[ln-1]
            print(f"  note: case-insensitive hit '{w}' at line {ln}: {line.strip()[:90]}")
    for w in FORBID_SUB:
        for mm in re.finditer(re.escape(w), txt, flags=re.I):
            ln = txt.count("\n", 0, mm.start()) + 1
            print(f"  FORBIDDEN word '{w}' at line {ln}"); fail += 1
    # 2. docstring balance
    opens = txt.count("/--") + txt.count("/-!") + txt.count("/- ")
    closes = txt.count("-/")
    if opens != closes:
        print(f"  DOCSTRING imbalance: opens={opens} closes={closes}"); fail += 1
    # 3. namespace
    ns = re.findall(r"^namespace\s+(\S+)", txt, flags=re.M)
    ends = re.findall(r"^end\s+(\S+)", txt, flags=re.M)
    if len(ns) != 1 or ends != ns:
        print(f"  NAMESPACE problem: {ns} / {ends}"); fail += 1
    else:
        print(f"  namespace {ns[0]}")
    # 4. imports
    imps = re.findall(r"^import\s+(\S+)", txt, flags=re.M)
    print(f"  imports: {imps}")
    for i in imps:
        if not i.startswith("Mathlib"):
            print(f"  NON-MATHLIB import {i}"); fail += 1
    # 5. declarations + docstrings + anchors
    decl_re = re.compile(r"^(?:noncomputable\s+)?(def|abbrev|structure)\s+(\S+)", re.M)
    pos = 0
    decls = []
    for dm in decl_re.finditer(txt):
        kind, name = dm.group(1), dm.group(2)
        # find docstring immediately preceding
        before = txt[:dm.start()].rstrip()
        if not before.endswith("-/"):
            print(f"  NO DOCSTRING before {kind} {name}"); fail += 1; continue
        ds_start = before.rfind("/--")
        ds = before[ds_start:]
        # is it a Prop def? look at signature up to ':=' 
        sig_end = txt.find(":=", dm.start())
        sig = txt[dm.start():sig_end]
        is_prop = sig.rstrip().endswith(": Prop") or kind == "structure"
        labels = ["Source:", "Anchor:", "Interface:", "Does not claim:"]
        missing = [l for l in labels if l not in ds]
        if missing:
            print(f"  MISSING labels {missing} in docstring of {kind} {name}"); fail += 1
        # anchor verbatim
        src_sha = re.search(r"sha256 ([0-9a-f]{64})", ds)
        anchors = re.findall(r'Anchor: "(.*?)"\s*$', ds, flags=re.M)
        if not anchors:
            print(f"  NO ANCHOR parsed for {name}"); fail += 1
        for a in anchors:
            if src_sha is None or src_sha.group(1) not in by_sha:
                print(f"  UNKNOWN SOURCE for {name}"); fail += 1; continue
            sid, b = by_sha[src_sha.group(1)]
            if a.encode("utf-8") not in b:
                print(f"  ANCHOR NOT VERBATIM for {name} (source {sid}): {a[:80]!r}"); fail += 1
            if "-/" in a or "/-" in a:
                print(f"  ANCHOR contains comment delimiter for {name}"); fail += 1
        decls.append((kind, name, "Prop" if is_prop else sig.split(":")[-1].strip()[:30], sid if src_sha else None, anchors[0] if anchors else None))
    print(f"  {len(decls)} declarations")
    for d in decls:
        print("   ", d[0], d[1], "|", d[2], "|", d[3])
print("FAILURES:", fail)
sys.exit(1 if fail else 0)
