"""Regenerate the per-declaration records for the structured output from the module text itself
(name, kind, module, source id, anchor, does-not-claim), with claim pointers restricted to the
claims a declaration STATES (dependencies live in the does-not-claim text). Scientific effect: NONE."""
import json, re, pathlib
PKT = pathlib.Path("<SCRATCH>/repos/math/specifications/lean_statements_20261010")
OUT = pathlib.Path("<SCRATCH>/lvs/fixer_fl/decls.json")
SRC = {s["sha256"]: s["id"] for s in json.loads((PKT / "sources/SOURCES.json").read_text())["sources"]}
PARENT = ["math.uniform-matrix-cap-lifetime", "D1-parent-lifetime-Theorems-A-B-C"]
SHARED_IDS = PARENT + ["math.lifetime-remainder", "lifetime-remainder"]
THM_A = ["math.uniform-matrix-cap-lifetime / D1-THM-A", "D1-parent-lifetime-Theorem-A-(1.1)"]
THM_B = ["math.uniform-matrix-cap-lifetime / D1-THM-B", "D1-parent-lifetime-Theorem-B-(1.2)"]
THM_C = ["math.uniform-matrix-cap-lifetime / D1-THM-C", "D1-parent-lifetime-Theorem-C-(1.3)"]
ELDER = ["OA-ELDER-LOWER-DENSITY-GAP-20260928-v1"]
SPECIAL = {
    # auxiliaries consumed by no statement: attached to no claim (verifier minor finding 4)
    "pinM": [], "pinS": [],
    "IsWindowPair": PARENT + THM_B + ELDER,
    "IsWindowElderPair": PARENT + THM_B + ELDER,
    "IsUnrestrictedPair": PARENT + THM_C + ["math.lifetime-remainder", "lifetime-remainder"],
    "PairLawInterface": PARENT + THM_A + ELDER,
    "ParentUniformSelectionA": PARENT + THM_A,
    "ParentCompactWindowDensityB": PARENT + THM_B,
    "IsUnrestrictedLeadingCoefficient": PARENT + THM_C + ["math.lifetime-remainder", "lifetime-remainder",
                                                           "math.side24-coefficient / side24-coefficient"],
    "ParentUnrestrictedLeadingDensityC": PARENT + THM_C + ["math.side24-coefficient / side24-coefficient"],
    "RemainderBoundedR1": ["math.lifetime-remainder / lifetime-remainder", "math.lifetime-remainder", "lifetime-remainder",
                           "D2-lifetime-remainder-Theorem-R-(R1)"],
    "PlanarElderLowerE2": ELDER + ["D1-elder-lower-(E2)"],
    # E4 and E11 STATE the planar results; (1.1) is their hypothesis, recorded in does_not_claim, not a claim they state
    "PlanarTwoSidedRateE4": ELDER + ["D1-elder-lower-(E4)"],
    "PlanarDensityGapE11": ELDER + ["D1-elder-lower-density-gap-(E11)"],
}
decl_re = re.compile(r"^(?:noncomputable\s+)?(def|abbrev|structure)\s+(\S+)", re.M)
out = []
for mod in ["Field", "Lifetime"]:
    txt = (PKT / "Specifications" / f"{mod}.lean").read_text(encoding="utf-8")
    for dm in decl_re.finditer(txt):
        kind, name = dm.group(1), dm.group(2)
        before = txt[:dm.start()].rstrip()
        assert before.endswith("-/"), name
        ds = before[before.rfind("/--"):]
        sig = txt[dm.start():txt.find(":=", dm.start())]
        sha = re.search(r"sha256 ([0-9a-f]{64})", ds).group(1)
        anchor = re.search(r'Anchor: "(.*?)"\s*$', ds, flags=re.M).group(1)
        dnc = re.search(r"Does not claim: (.*?) -/\s*$", ds, flags=re.S).group(1).strip().replace("\n", " ")
        if kind == "structure":
            k = "structure"
        elif sig.rstrip().endswith(": Prop"):
            k = "def_prop"
        elif kind == "abbrev":
            k = "abbrev"
        else:
            k = "def"
        ids = SPECIAL.get(name, SHARED_IDS)
        if mod == "Lifetime" and name not in SPECIAL or (mod == "Lifetime" and name in ("pinM", "pinS")):
            dnc = (f"Re-declaration (same text and anchor) of UniversalLaw.Spec.Field.{name} inside "
                   f"Specifications/Lifetime.lean so that the module imports only Mathlib; claims nothing beyond that entry. ") + dnc
        out.append({"name": name, "kind": k, "module": f"Specifications/{mod}.lean (namespace UniversalLaw.Spec.{mod})",
                    "source_id": SRC[sha], "anchor": anchor, "does_not_claim": dnc,
                    "formal_progress": "specified", "claim_ids": ids})
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(len(out), "declarations written to", OUT)
for o in out:
    if o["name"] in ("pinM", "pinS", "PlanarTwoSidedRateE4", "PlanarDensityGapE11", "IsLifetimeDensityVersion", "PairLawInterface"):
        print(" ", o["module"].split(" ")[0], o["name"], o["kind"], o["claim_ids"])
        print("     DNC:", o["does_not_claim"][:160])
