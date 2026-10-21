import sys, zipfile, hashlib, json, re
import xml.etree.ElementTree as ET
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
OPTS = ["PRIMARY_DERIVATION","EXECUTABLE_RECEIPT","RAW_DATA","FORMAL_PROOF","INDEPENDENT_RECONSTRUCTION",
        "SUMMARY","NAVIGATION","BALLOT","REGISTER_COPY","REVIEW_OF_REVIEW","GOVERNANCE_PRIMARY"]
def main(xlsx, jsonpath=None):
    raw = open(xlsx, "rb").read()
    print("xlsx", xlsx.rsplit("/",1)[-1], len(raw), hashlib.sha256(raw).hexdigest())
    z = zipfile.ZipFile(xlsx)
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    tg = {r.get("Id"): r.get("Target") for r in rels.findall("pr:Relationship", NS)}
    part = None
    for s in wb.find("m:sheets", NS).findall("m:sheet", NS):
        if s.get("name") == "Evidence Lineage":
            t = tg[s.get(RID)]; t = t[1:] if t.startswith("/") else ("xl/"+t if not t.startswith("xl/") else t)
            part = t
    b = z.read(part)
    print("part", part, len(b), hashlib.sha256(b).hexdigest())
    ws = ET.fromstring(b)
    dvs = ws.find("m:dataValidations", NS)
    if dvs is None:
        print("no dataValidations"); return
    for dv in dvs.findall("m:dataValidation", NS):
        f1 = dv.find("m:formula1", NS)
        print("dv", {k: v for k, v in dv.attrib.items()}, "formula1=", (f1.text if f1 is not None else None))
    if jsonpath:
        j = json.load(open(jsonpath, encoding="utf-8"))
        print("json keys", sorted(j.keys()), "header[5]", j["header"][5], "rows", len(j["rows"]))
        pop = [r[5] for r in j["rows"] if len(r) > 5 and r[5] != ""]
        inl = [v for v in pop if v in OPTS]
        print("populated", len(pop), "in_list", len(inl), "outside", len(pop) - len(inl))
main(*sys.argv[1:])
