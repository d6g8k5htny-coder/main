"""Spot-check Evidence Lineage F counts and the F validation element in each frozen export (stdlib only)."""
import hashlib, re, sys, zipfile
import xml.etree.ElementTree as ET
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
OPTS = None
for path in sys.argv[1:]:
    z = zipfile.ZipFile(path)
    ss = [("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"]) if True)) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
    raw = z.read("xl/worksheets/sheet22.xml")
    root = ET.fromstring(raw)
    vals = {}
    for c in root.iter("{%s}c" % NS["m"]):
        r = c.get("r")
        m = re.fullmatch(r"F(\d+)", r)
        if not m or int(m.group(1)) < 2: continue
        v = c.find("m:v", NS)
        if v is None: continue
        s = ss[int(v.text)] if c.get("t") == "s" else v.text
        if s and s.strip(): vals[int(m.group(1))] = s.strip()
    dv = re.findall(rb"<dataValidation [^>]*>.*?</dataValidation>", raw)
    opts = re.search(rb"<formula1>&quot;(.*?)&quot;</formula1>", dv[0]).group(1).decode().split(",")
    inl = sum(1 for v in vals.values() if v in opts)
    print(path.rsplit("/", 1)[1], hashlib.sha256(open(path, "rb").read()).hexdigest()[:12],
          "populated", len(vals), "in", inl, "out", len(vals) - inl, "distinct", len(set(vals.values())), "maxrow", max(vals),
          "n_dv", len(dv), "dv_sha", hashlib.sha256(dv[0]).hexdigest()[:12], "unused", [o for o in opts if o not in set(vals.values())])
