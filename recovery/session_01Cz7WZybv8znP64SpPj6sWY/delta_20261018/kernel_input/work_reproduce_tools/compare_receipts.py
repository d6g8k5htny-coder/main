import json, hashlib, sys, os
from pathlib import Path
deliv = Path(sys.argv[1]); mine = Path(sys.argv[2])
DBASE = "<DELIVERED_BASE>/"  # set to the cwd recorded in the delivered receipts
MBASE = str(mine) + "/"
def h(p):
    b = Path(p).read_bytes(); return len(b), hashlib.sha256(b).hexdigest()
VOLATILE = {"utc", "wall_seconds"}
mism = []
names = ["point-produce","point-verify","interval-produce","interval-verify","work-hold","cutoff-hold","real_cutoff-hold","tail-understatement-reject","rounding-narrowing-reject"]
for n in names:
    d = json.loads((deliv/"evidence/cli-fixtures"/(n+".json")).read_text())
    m = json.loads((mine/"evidence/cli-fixtures"/(n+".json")).read_text())
    if set(d) != set(m): mism.append((n, "keyset", sorted(set(d)^set(m))))
    for k in d:
        if k in VOLATILE: continue
        dv, mv = d[k], m.get(k)
        if k == "cwd":
            if dv.replace(DBASE.rstrip("/"), "<BASE>") != mv.replace(MBASE.rstrip("/"), "<BASE>"): mism.append((n,k,dv,mv))
            continue
        if k in ("sources","inputs","outputs") or k == "raw_log":
            dl = dv if isinstance(dv, list) else [dv]; ml = mv if isinstance(mv, list) else [mv]
            if len(dl) != len(ml): mism.append((n,k,"len",len(dl),len(ml))); continue
            for a, b in zip(dl, ml):
                ra, rb = a["path"].replace(DBASE,""), b["path"].replace(MBASE,"")
                if ra != rb or a["bytes"] != b["bytes"] or a["sha256"] != b["sha256"]:
                    mism.append((n,k,ra,a["bytes"],a["sha256"][:16],rb,b["bytes"],b["sha256"][:16]))
            continue
        if dv != mv: mism.append((n,k,dv,mv))
    print(f"{n}: exit delivered={d['actual_exit']} mine={m['actual_exit']} expected={d['expected_exit']} wall delivered={d['wall_seconds']:.4f} mine={m['wall_seconds']:.4f}")
# summary
ds = json.loads((deliv/"evidence/cli-fixtures/SUMMARY.json").read_text()); ms = json.loads((mine/"evidence/cli-fixtures/SUMMARY.json").read_text())
for k in ds:
    if k in ("total_wall_seconds","max_run_wall_seconds","receipts"): continue
    if ds[k] != ms[k]: mism.append(("SUMMARY",k,ds[k],ms[k]))
for a,b in zip(ds["receipts"], ms["receipts"]):
    if a["path"].replace(DBASE,"") != b["path"].replace(MBASE,""): mism.append(("SUMMARY","receipt order",a["path"],b["path"]))
# delivered summary self-consistency: delivered receipts hash to summary values
for a in ds["receipts"]:
    rel = a["path"].replace(DBASE,""); bl, sh = h(deliv/rel)
    if (bl, sh) != (a["bytes"], a["sha256"]): mism.append(("SUMMARY-delivered-binding", rel))
tot = sum(json.loads((deliv/"evidence/cli-fixtures"/(n+".json")).read_text())["wall_seconds"] for n in names)
mx = max(json.loads((deliv/"evidence/cli-fixtures"/(n+".json")).read_text())["wall_seconds"] for n in names)
print("delivered summary total/max", ds["total_wall_seconds"], ds["max_run_wall_seconds"], "recomputed", tot, mx)
print("MISMATCHES:", len(mism))
for x in mism: print("  ", x)
