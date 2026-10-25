import json, sys
OLD = "https://github.com/d6g8k5htny-coder/Math-/blob/79cb314d5b9f6b8b410bfd0df200414314f6b8c7/frontiers/p15_tail_load_20260928/README.md"
NEW = OLD[:-len("README.md")] + "PROOF.md"
ops = []
for base in ("/records/99", "/records/100", "/records/101"):
    ops += [{"op": "test", "path": base + "/id", "value": {"/records/99": "P15-T1", "/records/100": "P15-T2", "/records/101": "P15-T3"}[base]},
            {"op": "test", "path": base + "/source", "value": OLD},
            {"op": "replace", "path": base + "/source", "value": NEW}]
for base, oid in (("/obligations/99", "OBL:P15-T1"), ("/obligations/100", "OBL:P15-T2"), ("/obligations/101", "OBL:P15-T3")):
    ops += [{"op": "test", "path": base + "/id", "value": oid},
            {"op": "test", "path": base + "/proof_bundle/source", "value": OLD},
            {"op": "replace", "path": base + "/proof_bundle/source", "value": NEW}]
ops += [{"op": "test", "path": "/obligations/99/premises", "value": ["P15-E1", "P15-E2", "P15-T3"]},
        {"op": "replace", "path": "/obligations/99/premises", "value": ["P15-E1", "P15-E2", "P15-T2", "P15-T3"]}]
json.dump(ops, open("P15_APPLY_READY.patch.json", "w"), indent=1, ensure_ascii=False)
print(len(ops), "ops")
