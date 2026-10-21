#!/usr/bin/env python3
"""parser-A: citation helpers -- byte offsets of F1 cell and dataValidations in the
Evidence Lineage member, F1 shared-string index, physical line count of the member,
and a scan of ALL worksheets for validations whose formula mentions PRIMARY_DERIVATION.
Usage: python3 -I cite_offsets.py XLSX [XLSX ...]
"""
import re
import sys
import zipfile


def main():
    for p in sys.argv[1:]:
        z = zipfile.ZipFile(p)
        name = p.rsplit("/", 1)[-1]
        b = z.read("xl/worksheets/sheet22.xml")
        s = b.decode("utf-8")
        f1 = re.search(r'<c r="F1"[^>]*>.*?</c>', s)
        dv = s.find("<dataValidations")
        dve = s.find("</dataValidations>")
        print("==", name, "sheet22 lines", b.count(b"\n") + 1,
              "F1 at byte", f1.start(), f1.group(0),
              "dataValidations bytes %d..%d" % (dv, dve + len("</dataValidations>")))
        print("   extLst present:", "<extLst" in s, "x14:dataValidation present:", "x14:dataValidation" in s,
              "mc:AlternateContent present:", "AlternateContent" in s)
        hits = []
        for m in z.namelist():
            if m.startswith("xl/worksheets/sheet") and m.endswith(".xml"):
                t = z.read(m).decode("utf-8")
                n_dv = len(re.findall(r"<(?:x14:)?dataValidation[ >]", t))
                if "PRIMARY_DERIVATION" in "".join(re.findall(r"<(?:x14:)?dataValidations.*?</(?:x14:)?dataValidations>", t, re.S)):
                    hits.append((m, n_dv))
        print("   worksheets whose validations mention PRIMARY_DERIVATION:", hits)


if __name__ == "__main__":
    main()
