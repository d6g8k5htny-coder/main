# Minimal RFC 6902 test/replace applier + full diff of the result against the input.
import json, copy, sys
src = json.load(open(sys.argv[1])); ops = json.load(open(sys.argv[2]))
def ptr(doc, path):
    parts = [p.replace("~1", "/").replace("~0", "~") for p in path.split("/")[1:]]
    for p in parts[:-1]: doc = doc[int(p)] if isinstance(doc, list) else doc[p]
    k = parts[-1]; return doc, (int(k) if isinstance(doc, list) else k)
out = copy.deepcopy(src)
for op in ops:
    d, k = ptr(out, op["path"])
    if op["op"] == "test":
        assert d[k] == op["value"], ("TEST FAILED", op["path"])
    elif op["op"] == "replace":
        assert k in d if isinstance(d, dict) else k < len(d); d[k] = op["value"]
    else: raise SystemExit("unsupported op")
diffs = []
def walk(a, b, p=""):
    if type(a) != type(b): diffs.append(p); return
    if isinstance(a, dict):
        if set(a) != set(b): diffs.append(p + " (keys)")
        for k in a:
            if k in b: walk(a[k], b[k], p + "/" + k)
    elif isinstance(a, list):
        if len(a) != len(b): diffs.append(p); return
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{p}/{i}")
    elif a != b: diffs.append(p)
walk(src, out)
print("all tests passed; changed leaves:", len(diffs)); [print(" ", d) for d in diffs]
print("records:", len(src["records"]), "->", len(out["records"]), " obligations:", len(src["obligations"]), "->", len(out["obligations"]))
print("original_known_required_links:", out["metadata"].get("original_known_required_links"))
print("fingerprints unchanged (not recomputed):", all(src["obligations"][i]["fingerprint"] == out["obligations"][i]["fingerprint"] for i in range(len(src["obligations"]))))
json.dump(out, open("canonical_dataset_patched_preview.json", "w"), indent=1, ensure_ascii=False)
