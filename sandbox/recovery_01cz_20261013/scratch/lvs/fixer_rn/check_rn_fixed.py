"""NON-CERTIFYING mechanical checks of Specifications/RN.lean after the fixer edits (no Lean).
Uses the packet's own check.py scanner (scan_module) for the lexical refusal list, column-0
commands, namespace, imports, Prop-kind detection and the four docstring labels with verbatim
anchors; then re-checks anchors/labels for ALL declarations (helpers included), the shared block
against Field.lean, and the honesty words. Exit nonzero on any failure."""
import hashlib, importlib.util, json, pathlib, re, sys

PKT = pathlib.Path("<SCRATCH>/repos/math/specifications/lean_statements_20261010")
RN = PKT / "Specifications/RN.lean"
FIELD = PKT / "Specifications/Field.lean"
raw = RN.read_bytes()
txt = raw.decode("utf-8")
print("RN.lean bytes", len(raw), "sha256", hashlib.sha256(raw).hexdigest())
fails = []

# --- packet scanner
spec = importlib.util.spec_from_file_location("pktcheck", PKT / "check.py")
pk = importlib.util.module_from_spec(spec); spec.loader.exec_module(pk)
SRC = json.loads((PKT / "sources/SOURCES.json").read_text())
sources = {}
for s in SRC["sources"]:
    b = (PKT / s["local_copy"]).read_bytes()
    if hashlib.sha256(b).hexdigest() != s["sha256"] or len(b) != s["bytes"]:
        fails.append(("source identity", s["id"]))
    sources[s["id"]] = dict(s, text=b.decode("utf-8"))
try:
    decls = pk.scan_module(txt, "RN", sources)
except Exception as e:  # noqa
    fails.append(("scan_module", repr(e))); decls = []
kinds = {}
for d in decls: kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
print("scan_module: declarations", len(decls), "kinds", kinds)
props = [d["name"] for d in decls if d["kind"] == "def_prop"]
structs = [d["name"] for d in decls if d["kind"] == "structure"]
print("def_prop:", len(props), "structure:", structs)

# --- all declarations (helpers too): docstring labels + verbatim anchor
by_sha = {s["sha256"]: s for s in sources.values()}
decl_re = re.compile(r"^(?:noncomputable\s+)?(def|abbrev|structure)\s+(\S+)", re.M)
n_all = 0
for dm in decl_re.finditer(txt):
    name = dm.group(2); n_all += 1
    before = txt[:dm.start()].rstrip()
    ds = before[before.rfind("/--"):]
    if not before.endswith(ds.rstrip()) or not ds.rstrip().endswith("-/"):
        fails.append((name, "docstring not immediately preceding"))
    for lab in ("Source:", "Anchor:", "Interface:", "Does not claim:"):
        if len(re.findall(r"^" + re.escape(lab), ds, flags=re.M)) != 1:
            fails.append((name, "label count", lab))
    m = re.search(r"^Source: d6g8k5htny-coder/Math- commit ([0-9a-f]{40}) path (\S+) sha256 ([0-9a-f]{64})\s*$", ds, flags=re.M)
    am = re.search(r'^Anchor: "(.*)"\s*$', ds, flags=re.M)
    if not m or not am:
        fails.append((name, "Source/Anchor malformed")); continue
    s = by_sha.get(m.group(3))
    if not s or s["commit"] != m.group(1) or s["path"] != m.group(2):
        fails.append((name, "Source line does not match SOURCES.json")); continue
    if am.group(1).encode("utf-8") not in (PKT / s["local_copy"]).read_bytes():
        fails.append((name, "ANCHOR NOT VERBATIM", am.group(1)))
print("all declarations (helpers included):", n_all, "anchors/labels checked")

# --- shared block identity with Field.lean (docstring + body of the nine shared declarations)
field = FIELD.read_text(encoding="utf-8")
def block(text, name, kw):
    m = re.search(r"^(?:noncomputable )?" + kw + r"\s+" + re.escape(name) + r"\b", text, flags=re.M)
    start = text.rfind("/--", 0, m.start())
    end = text.find("\n\n", m.start())
    return text[start:end]
shared = [("E", "abbrev"), ("latticeVec", "def"), ("hessian", "def"), ("IsCriticalPoint", "def"),
          ("HessianHasInertia", "def"), ("pinM", "def"), ("pinS", "def"), ("torusDist", "def"), ("FundDomain", "def")]
same_body = same_doc = 0
def split(b):
    i = b.find("-/\n"); return b[:i + 2], b[i + 3:]
for name, kw in shared:
    da, ba = split(block(txt, name, kw)); db, bb = split(block(field, name, kw))
    if ba == bb: same_body += 1
    else: fails.append((name, "shared BODY differs from Field.lean"))
    if da == db: same_doc += 1
    elif name in ("pinM", "pinS"):
        strip = lambda d: re.sub(r"^Does not claim:.*$", "", d, flags=re.M)
        if strip(da) != strip(db): fails.append((name, "pin docstring differs beyond the Does-not-claim line"))
    else: fails.append((name, "shared docstring differs from Field.lean"))
print("shared bodies identical to Field.lean:", same_body, "/ 9; docstrings identical:", same_doc, "/ 9 (pinM, pinS differ in Does-not-claim only)")
# the three gated interface fields must be byte-identical in the negative-control file
ctl = pathlib.Path("<SCRATCH>/lvs/fixer_rn/EmptyInterface_gated.lean").read_text(encoding="utf-8")
for fld in ("Q_pinned", "Z_pos", "QW_eq"):
    m = re.search(r"^  " + fld + r" : .*?\n(?=  /--|  [A-Za-zΛ]+ :)", txt, flags=re.M | re.S)
    if m.group(0) not in ctl: fails.append((fld, "gated field text not byte-identical in EmptyInterface_gated.lean"))
print("gated fields Q_pinned/Z_pos/QW_eq byte-identical in the negative control: checked")

# --- honesty words anywhere in the module text
for m in re.finditer(r"\b(verified|proved|closed|certified|proven)\b", txt, re.I):
    fails.append(("honesty word", m.group(0), txt[:m.start()].count("\n") + 1))
# --- imports and namespace
if re.findall(r"^import\s+(\S+)", txt, re.M) != ["Mathlib"]: fails.append(("imports",))

print("FAILURES:", fails)
sys.exit(1 if fails else 0)
