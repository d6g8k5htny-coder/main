import json, hashlib, sys
d = json.load(open(sys.argv[1]))
cur = json.dumps(d, indent=2, sort_keys=True)
for nl in ("", "\n"):
    b = (cur+nl).encode()
    print("current reserialize", nl!="", len(b), hashlib.sha256(b).hexdigest()[:16])
def strip(d, drop_keys, drop_prefixes):
    e = json.loads(json.dumps(d))
    for k in drop_keys: e.pop(k, None)
    e["sha256"] = {k:v for k,v in e["sha256"].items() if not any(k.startswith(p) for p in drop_prefixes)}
    return e
v2 = strip(d, ["band_ceil"], ["ceil_", "h3_band_ceil", "H3_BAND_CEIL"])
v1 = strip(d, ["band_ceil","band_floor"], ["ceil_", "h3_band_ceil", "H3_BAND_CEIL","band_","h3_band_floor","H3_BAND_FLOOR"])
v0 = strip(v1, ["rung_floor"], ["rung_", "h3_rung", "H3_RUNG"])
for name, e in (("v2(no ceil)",v2),("v1(no band)",v1),("v0(no band,no rung)",v0)):
    s = json.dumps(e, indent=2, sort_keys=True)
    for nl in ("", "\n"):
        b=(s+nl).encode(); print(name, "nl" if nl else "", len(b), hashlib.sha256(b).hexdigest())
