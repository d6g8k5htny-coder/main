import json, re, pathlib, hashlib
PKT = pathlib.Path("<SCRATCH>/repos/math/specifications/lean_statements_20261010")
SRC = {s["sha256"]: s["id"] for s in json.loads((PKT/"sources/SOURCES.json").read_text())["sources"]}
PARENT = ["math.uniform-matrix-cap-lifetime", "D1-parent-lifetime-Theorems-A-B-C"]
SHARED_IDS = PARENT + ["math.lifetime-remainder", "lifetime-remainder"]
SPECIAL = {
 "IsWindowPair": PARENT + ["D1-parent-lifetime-Theorem-B-(1.2)", "OA-ELDER-LOWER-DENSITY-GAP-20260928-v1"],
 "IsWindowElderPair": PARENT + ["D1-parent-lifetime-Theorem-B-(1.2)", "OA-ELDER-LOWER-DENSITY-GAP-20260928-v1"],
 "IsUnrestrictedPair": PARENT + ["D1-parent-lifetime-Theorem-C-(1.3)", "math.lifetime-remainder", "lifetime-remainder"],
 "PairLawInterface": PARENT + ["D1-parent-lifetime-Theorem-A-(1.1)", "OA-ELDER-LOWER-DENSITY-GAP-20260928-v1"],
 "ParentUniformSelectionA": PARENT + ["D1-parent-lifetime-Theorem-A-(1.1)"],
 "ParentCompactWindowDensityB": PARENT + ["D1-parent-lifetime-Theorem-B-(1.2)"],
 "IsUnrestrictedLeadingCoefficient": PARENT + ["D1-parent-lifetime-Theorem-C-(1.3)", "math.lifetime-remainder", "lifetime-remainder", "side24-coefficient", "math.side24-coefficient"],
 "ParentUnrestrictedLeadingDensityC": PARENT + ["D1-parent-lifetime-Theorem-C-(1.3)", "side24-coefficient", "math.side24-coefficient"],
 "RemainderBoundedR1": ["lifetime-remainder", "math.lifetime-remainder", "D2-lifetime-remainder-Theorem-R-(R1)"],
 "PlanarElderLowerE2": ["OA-ELDER-LOWER-DENSITY-GAP-20260928-v1", "D1-elder-lower-(E2)"],
 "PlanarTwoSidedRateE4": ["OA-ELDER-LOWER-DENSITY-GAP-20260928-v1", "D1-elder-lower-(E4)", "D1-parent-lifetime-Theorem-A-(1.1)"],
 "PlanarDensityGapE11": ["OA-ELDER-LOWER-DENSITY-GAP-20260928-v1", "D1-elder-lower-density-gap-(E11)", "D1-parent-lifetime-Theorem-B-(1.2)"],
}
decl_re = re.compile(r"^(?:noncomputable\s+)?(def|abbrev|structure)\s+(\S+)", re.M)
out = []
for mod in ["Field", "Lifetime"]:
    txt = (PKT/"Specifications"/f"{mod}.lean").read_text(encoding="utf-8")
    for dm in decl_re.finditer(txt):
        kind, name = dm.group(1), dm.group(2)
        before = txt[:dm.start()].rstrip(); ds = before[before.rfind("/--"):]
        sig = txt[dm.start():txt.find(":=", dm.start())]
        sha = re.search(r"sha256 ([0-9a-f]{64})", ds).group(1)
        anchor = re.search(r'Anchor: "(.*?)"\s*$', ds, flags=re.M).group(1)
        dnc = re.search(r"Does not claim: (.*?) -/\s*$", ds, flags=re.S).group(1).strip().replace("\n", " ")
        if kind == "structure": k = "structure"
        elif sig.rstrip().endswith(": Prop"): k = "def_prop"
        elif kind == "abbrev": k = "abbrev"
        else: k = "def"
        ids = SPECIAL.get(name, SHARED_IDS)
        if mod == "Lifetime" and name not in SPECIAL:
            dnc = "(re-declaration of Specifications/Field.lean's " + name + " in this namespace; same text) " + dnc
        out.append({"name": name, "kind": k, "module": f"Specifications/{mod}.lean (namespace UniversalLaw.Spec.{mod})",
                    "source_id": SRC[sha], "anchor": anchor, "does_not_claim": dnc,
                    "formal_progress": "specified", "claim_ids": ids})
print(len(out), "declarations")
json.dump(out, open("decls.json", "w"), ensure_ascii=False, indent=1)
for o in out:
    if "Lifetime" in o["module"] and o["name"] in SPECIAL or "Field" in o["module"]:
        pass
print(json.dumps([{k: o[k] for k in ("name","kind","module","source_id")} for o in out if o["name"] in SPECIAL], ensure_ascii=False, indent=0)[:3000])
