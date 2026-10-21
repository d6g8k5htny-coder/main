#!/usr/bin/env python3
"""parser-A: independent Evidence Role (column F) replay for GP-REG-032 xlsx exports.

Standard library only (zipfile + xml.etree.ElementTree). Run with python3 -I.
Usage: python3 -I evrole_xlsx.py OUT_JSON XLSX [XLSX ...]

Whitespace rule (stated once, applied everywhere):
  * a cell's text value is the resolved string (shared string / inline string / <v>);
  * a value is "empty" iff value.strip() == "" (Python str.strip(), Unicode whitespace);
  * list membership is an exact, case-sensitive comparison of value.strip() with an
    option string (options are also .strip()ped after splitting the inline list on ',').
  Diagnostics also report how many populated F values differ between raw and stripped
  form, and how many outside-list values would match case-insensitively.
"""
import hashlib
import json
import posixpath
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_X14 = "http://schemas.microsoft.com/office/spreadsheetml/2009/9/main"
NS_XM = "http://schemas.microsoft.com/office/excel/2006/main"
M = "{%s}" % NS_MAIN
X14 = "{%s}" % NS_X14
XM = "{%s}" % NS_XM

TARGET_SHEET = "Evidence Lineage"
XESC = re.compile(r"_x([0-9A-Fa-f]{4})_")


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def git_blob(b):
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def decode_xesc(s):
    return XESC.sub(lambda m: chr(int(m.group(1), 16)), s)


def col2num(col):
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch.upper()) - 64)
    return n


def num2col(n):
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


REF = re.compile(r"^\$?([A-Za-z]{1,3})?\$?(\d+)?$")


def parse_ref(ref):
    m = REF.match(ref)
    if not m:
        raise ValueError("bad ref %r" % ref)
    c = col2num(m.group(1)) if m.group(1) else None
    r = int(m.group(2)) if m.group(2) else None
    return c, r


def parse_sqref(sqref):
    """Return list of (c1, r1, c2, r2) with None for unbounded."""
    out = []
    for tok in sqref.split():
        if ":" in tok:
            a, b = tok.split(":", 1)
        else:
            a = b = tok
        c1, r1 = parse_ref(a)
        c2, r2 = parse_ref(b)
        out.append((c1, r1, c2, r2))
    return out


def covers(rect, col, row):
    c1, r1, c2, r2 = rect
    if c1 is not None and c2 is not None and not (min(c1, c2) <= col <= max(c1, c2)):
        return False
    if r1 is not None and r2 is not None and not (min(r1, r2) <= row <= max(r1, r2)):
        return False
    return True


def touches_col(rect, col):
    c1, _, c2, _ = rect
    if c1 is None or c2 is None:
        return True
    return min(c1, c2) <= col <= max(c1, c2)


def text_of_si(si):
    """Concatenate <t> of an <si>/<is>, including rich runs <r><t>, excluding <rPh>."""
    parts = []
    for child in si:
        if child.tag == M + "t":
            parts.append(child.text or "")
        elif child.tag == M + "r":
            for t in child.findall(M + "t"):
                parts.append(t.text or "")
        # rPh (phonetic) and phoneticPr are ignored deliberately
    return "".join(parts)


def load_shared_strings(z):
    try:
        data = z.read("xl/sharedStrings.xml")
    except KeyError:
        return [], None
    root = ET.fromstring(data)
    sst = [text_of_si(si) for si in root.findall(M + "si")]
    return sst, {"sha256": sha256(data), "count_attr": root.get("count"),
                 "uniqueCount_attr": root.get("uniqueCount"), "si_parsed": len(sst)}


def resolve_target(base_dir, target):
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(base_dir, target))


def workbook_sheets(z):
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rmap = {}
    for rel in rels.findall("{%s}Relationship" % NS_PKG_REL):
        rmap[rel.get("Id")] = (rel.get("Type"), rel.get("Target"))
    sheets = []
    for s in wb.find(M + "sheets").findall(M + "sheet"):
        rid = s.get("{%s}id" % NS_R)
        typ, tgt = rmap.get(rid, (None, None))
        sheets.append({"name": s.get("name"), "sheetId": s.get("sheetId"),
                       "state": s.get("state"), "rid": rid, "rel_type": typ,
                       "target": tgt,
                       "member": resolve_target("xl", tgt) if tgt else None})
    defined = []
    dn = wb.find(M + "definedNames")
    if dn is not None:
        for d in dn.findall(M + "definedName"):
            defined.append({"name": d.get("name"), "localSheetId": d.get("localSheetId"),
                            "hidden": d.get("hidden"), "text": d.text})
    return sheets, defined


def cell_value(c, sst, stats):
    t = c.get("t")
    v = c.find(M + "v")
    if t == "s":
        if v is None or v.text is None:
            return ""
        idx = int(v.text)
        if idx < 0 or idx >= len(sst):
            stats["bad_sst_index"] += 1
            return ""
        stats["t_s"] += 1
        return sst[idx]
    if t == "inlineStr":
        stats["t_inlineStr"] += 1
        is_ = c.find(M + "is")
        return text_of_si(is_) if is_ is not None else ""
    if t == "b":
        stats["t_b"] += 1
        if v is None or v.text is None:
            return ""
        return "TRUE" if v.text.strip() == "1" else "FALSE"
    if t == "str":
        stats["t_str"] += 1
    elif t == "e":
        stats["t_e"] += 1
    else:
        stats["t_n_or_none"] += 1
    if c.find(M + "f") is not None:
        stats["has_formula"] += 1
    return "" if v is None or v.text is None else v.text


def read_sheet(z, member, sst, max_col=None):
    """Return dict row -> dict col -> raw string, plus metadata and the parsed root."""
    data = z.read(member)
    root = ET.fromstring(data)
    stats = Counter()
    cells = {}
    rows_meta = {}
    sd = root.find(M + "sheetData")
    last_r = 0
    for row in sd.findall(M + "row"):
        r = int(row.get("r")) if row.get("r") else last_r + 1
        last_r = r
        rows_meta[r] = {"hidden": row.get("hidden")}
        last_c = 0
        for c in row.findall(M + "c"):
            ref = c.get("r")
            if ref:
                col, rr = parse_ref(ref)
                if rr != r:
                    stats["cell_row_mismatch"] += 1
            else:
                col = last_c + 1
                stats["cell_without_ref"] += 1
            last_c = col
            if max_col is not None and col > max_col:
                continue
            val = cell_value(c, sst, stats)
            if XESC.search(val):
                stats["xesc_decoded"] += 1
                val = decode_xesc(val)
            rowd = cells.setdefault(r, {})
            if col in rowd:
                stats["duplicate_cell_ref"] += 1
            rowd[col] = val
    return cells, rows_meta, root, data, stats


def collect_validations(root):
    vals = []
    dvs = root.find(M + "dataValidations")
    if dvs is not None:
        for i, dv in enumerate(dvs.findall(M + "dataValidation")):
            f1 = dv.find(M + "formula1")
            f2 = dv.find(M + "formula2")
            vals.append({"source": "main:dataValidations[%d]" % i,
                         "dataValidations_count_attr": dvs.get("count"),
                         "attrs": dict(dv.attrib),
                         "sqref": dv.get("sqref"),
                         "formula1": None if f1 is None else (f1.text or ""),
                         "formula2": None if f2 is None else (f2.text or "")})
    # x14 extension validations (extLst/ext/x14:dataValidations/x14:dataValidation)
    ext = root.find(M + "extLst")
    if ext is not None:
        for e in ext.findall(M + "ext"):
            x = e.find(X14 + "dataValidations")
            if x is None:
                continue
            for i, dv in enumerate(x.findall(X14 + "dataValidation")):
                f1 = dv.find(X14 + "formula1")
                f2 = dv.find(X14 + "formula2")
                sq = dv.find(XM + "sqref")
                f1t = None
                if f1 is not None:
                    xf = f1.find(XM + "f")
                    f1t = (xf.text or "") if xf is not None else (f1.text or "")
                f2t = None
                if f2 is not None:
                    xf = f2.find(XM + "f")
                    f2t = (xf.text or "") if xf is not None else (f2.text or "")
                vals.append({"source": "x14:ext uri=%s [%d]" % (e.get("uri"), i),
                             "attrs": dict(dv.attrib),
                             "sqref": sq.text if sq is not None else None,
                             "formula1": f1t, "formula2": f2t})
    # any other dataValidation elements anywhere (e.g. under mc:AlternateContent)
    seen = 0
    for el in root.iter():
        if el.tag.endswith("}dataValidation"):
            seen += 1
    return vals, seen


def split_inline_list(f1):
    s = f1
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].replace('""', '"')
    return [p for p in s.split(",")]


SHEETREF = re.compile(r"^=?(?:'((?:[^']|'')+)'|([^!]+))!(.+)$")


def resolve_range_formula(z, f1, sheets, defined, sst):
    """Resolve a range reference formula to list values. Returns dict."""
    f = f1.strip().lstrip("=")
    m = SHEETREF.match(f)
    info = {"kind": "range_reference", "formula": f1}
    if not m:
        # maybe a defined name
        for d in defined:
            if d["name"] == f:
                info["defined_name"] = d
                return dict(info, **resolve_range_formula(z, d["text"], sheets, defined, sst))
        info["kind"] = "unresolved_formula"
        return info
    sname = (m.group(1) or "").replace("''", "'") if m.group(1) else m.group(2)
    rng = m.group(3)
    sh = [s for s in sheets if s["name"] == sname]
    if not sh:
        info["error"] = "sheet %r not found" % sname
        return info
    cells, _, _, data, _ = read_sheet(z, sh[0]["member"], sst)
    rects = parse_sqref(rng.replace("$", ""))
    values = []
    for rect in rects:
        c1, r1, c2, r2 = rect
        maxr = max(cells) if cells else 0
        r1 = r1 or 1
        r2 = r2 or maxr
        for r in range(min(r1, r2), max(r1, r2) + 1):
            for c in range(min(c1, c2), max(c1, c2) + 1):
                values.append({"cell": "%s%d" % (num2col(c), r),
                               "value": cells.get(r, {}).get(c, "")})
    info.update({"sheet": sname, "member": sh[0]["member"],
                 "member_sha256": sha256(data), "range": rng, "values": values})
    return info


def analyse(path):
    raw = open(path, "rb").read()
    out = {"file": path, "bytes": len(raw), "sha256": sha256(raw), "git_blob": git_blob(raw)}
    z = zipfile.ZipFile(path)
    sst, sst_meta = load_shared_strings(z)
    out["sharedStrings"] = sst_meta
    out["workbook_xml_sha256"] = sha256(z.read("xl/workbook.xml"))
    out["workbook_rels_sha256"] = sha256(z.read("xl/_rels/workbook.xml.rels"))
    sheets, defined = workbook_sheets(z)
    out["sheet_count"] = len(sheets)
    out["defined_names"] = defined
    exact = [s for s in sheets if s["name"] == TARGET_SHEET]
    near = [s["name"] for s in sheets if "lineage" in s["name"].lower() or "evidence" in s["name"].lower()]
    out["sheets_matching_evidence_or_lineage"] = near
    if len(exact) != 1:
        out["error"] = "expected exactly one sheet named %r, found %d" % (TARGET_SHEET, len(exact))
        return out
    sh = exact[0]
    out["sheet"] = sh
    cells, rows_meta, root, data, stats = read_sheet(z, sh["member"], sst)
    out["sheet_member"] = sh["member"]
    out["sheet_member_sha256"] = sha256(data)
    out["sheet_member_bytes"] = len(data)
    out["cell_type_stats"] = dict(stats)
    dim = root.find(M + "dimension")
    out["dimension"] = dim.get("ref") if dim is not None else None
    af = root.find(M + "autoFilter")
    out["autoFilter"] = af.get("ref") if af is not None else None
    tparts = root.find(M + "tableParts")
    out["tableParts_count"] = int(tparts.get("count")) if tparts is not None else 0
    out["row_elements"] = len(rows_meta)
    out["max_row_element"] = max(rows_meta) if rows_meta else 0

    def nonempty(v):
        return v is not None and v.strip() != ""

    header = {c: cells.get(1, {}).get(c, "") for c in range(1, 17)}
    out["header_A_to_P"] = {num2col(c): header[c] for c in header}
    out["header_F"] = header[6]

    rows_any_AF = [r for r in sorted(cells) if r >= 2 and any(nonempty(cells[r].get(c)) for c in range(1, 7))]
    rows_any_any = [r for r in sorted(cells) if r >= 2 and any(nonempty(v) for v in cells[r].values())]
    N = rows_any_AF[-1] if rows_any_AF else 1
    out["last_row_with_any_A_to_F"] = N
    out["last_row_with_any_cell"] = rows_any_any[-1] if rows_any_any else 1
    out["data_rows_with_any_A_to_F"] = len(rows_any_AF)
    out["rows_2_to_N_without_any_A_to_F"] = [r for r in range(2, N + 1) if r not in set(rows_any_AF)]
    out["rows_with_content_only_outside_A_to_F"] = [r for r in rows_any_any if r not in set(rows_any_AF)]

    # validations
    vals, dv_elements_seen = collect_validations(root)
    out["all_dataValidation_elements_in_sheet"] = dv_elements_seen
    out["all_validations_parsed"] = len(vals)
    fvals = []
    options = None
    for v in vals:
        rects = parse_sqref(v["sqref"] or "")
        v["touches_F"] = any(touches_col(rc, 6) for rc in rects)
        v["rects"] = rects
        f1 = v["formula1"]
        if f1 is not None and f1.strip().startswith('"'):
            v["formula1_kind"] = "inline_list"
            v["formula1_list_raw"] = split_inline_list(f1.strip())
        elif f1 is not None:
            v["formula1_kind"] = "range_or_name"
            v["formula1_resolved"] = resolve_range_formula(z, f1, sheets, defined, sst)
        if v["touches_F"]:
            fvals.append(v)
    out["validations_touching_F"] = [
        {k: v[k] for k in v if k != "rects"} for v in fvals]
    out["validations_not_touching_F"] = [
        {"source": v["source"], "sqref": v["sqref"], "type": v["attrs"].get("type")} for v in vals if not v["touches_F"]]
    list_vals = [v for v in fvals if v["attrs"].get("type") == "list"]
    opt_sets = []
    for v in list_vals:
        if v.get("formula1_kind") == "inline_list":
            opt_sets.append(tuple(o.strip() for o in v["formula1_list_raw"]))
        elif v.get("formula1_resolved", {}).get("values") is not None:
            opt_sets.append(tuple(x["value"].strip() for x in v["formula1_resolved"]["values"] if x["value"].strip()))
    out["distinct_option_sets_on_F"] = [list(s) for s in sorted(set(opt_sets))]
    options = set(opt_sets[0]) if len(set(opt_sets)) == 1 else (set().union(*opt_sets) if opt_sets else set())
    out["option_count"] = len(options)
    out["options_have_internal_whitespace_or_case_dupes"] = len({o.upper() for o in options}) != len(options)

    # F column counts over rows 2..N
    populated = []
    blankF_other = []
    raw_vs_strip = 0
    for r in range(2, N + 1):
        rowd = cells.get(r, {})
        f = rowd.get(6)
        if nonempty(f):
            populated.append((r, f))
            if f != f.strip():
                raw_vs_strip += 1
        elif any(nonempty(rowd.get(c)) for c in range(1, 6)):
            blankF_other.append(r)
    inlist = [(r, f) for r, f in populated if f.strip() in options]
    outside = [(r, f) for r, f in populated if f.strip() not in options]
    upper_opts = {o.upper() for o in options}
    outside_ci = [(r, f) for r, f in outside if f.strip().upper() in upper_opts]
    cnt_all = Counter(f.strip() for _, f in populated)
    cnt_out = Counter(f.strip() for _, f in outside)
    cnt_in = Counter(f.strip() for _, f in inlist)
    out["populated_roles"] = len(populated)
    out["in_list"] = len(inlist)
    out["outside_list"] = len(outside)
    out["distinct_nonempty_labels"] = len(cnt_all)
    out["distinct_outside_labels"] = len(cnt_out)
    out["in_list_by_option"] = {o: cnt_in.get(o, 0) for o in (opt_sets[0] if opt_sets else [])}
    out["outside_case_insensitive_matches"] = len(outside_ci)
    out["populated_F_raw_differs_from_stripped"] = raw_vs_strip
    out["blank_role_but_other_A_to_F_populated"] = len(blankF_other)
    out["blank_role_rows"] = blankF_other
    out["top_outside_labels"] = cnt_out.most_common(10)
    out["all_label_counts"] = sorted(cnt_all.items(), key=lambda kv: (-kv[1], kv[0]))
    out["outside_label_shape"] = dict(Counter(
        ("semicolon_or_slash_compound" if (";" in k or "/" in k) else
         "contains_space" if " " in k else
         "lowercase_present" if k != k.upper() else
         "UPPER_SNAKE_token") for k in cnt_out.elements()))
    # F row/value fingerprint for cross-parser comparison
    fp = "\n".join("%d\t%s" % (r, f) for r, f in populated).encode("utf-8")
    out["F_rowvalue_fingerprint_sha256"] = sha256(fp)

    # coverage gaps: F cells in rows 2..N not covered by any F validation; populated F anywhere uncovered
    frects = [rc for v in fvals for rc in v["rects"]]
    uncovered_range = [r for r in range(2, N + 1) if not any(covers(rc, 6, r) for rc in frects)]
    out["F_rows_2_to_N_not_covered"] = uncovered_range
    out["F_populated_not_covered"] = [r for r, _ in populated if not any(covers(rc, 6, r) for rc in frects)]
    allF = [r for r in sorted(cells) if r >= 2 and nonempty(cells[r].get(6))]
    out["F_populated_anywhere_not_covered"] = [r for r in allF if not any(covers(rc, 6, r) for rc in frects)]
    # per-list-validation coverage of populated rows
    out["hidden_rows_in_2_to_N"] = [r for r in range(2, N + 1) if rows_meta.get(r, {}).get("hidden") in ("1", "true")]
    # sample cell evidence for citation
    out["sample_cells"] = {("F%d" % r): f for r, f in populated[:3]}
    out["sample_outside_cells"] = {("F%d" % r): f for r, f in outside[:5]}
    out["first_last_populated_F"] = [populated[0][0], populated[-1][0]] if populated else None
    return out


def main():
    outp = sys.argv[1]
    res = [analyse(p) for p in sys.argv[2:]]
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    for r in res:
        print("==", r["file"].rsplit("/", 1)[-1], r["bytes"], r["sha256"])
        for k in ["git_blob", "sheet_member", "sheet_member_sha256", "dimension", "autoFilter",
                  "row_elements", "max_row_element", "header_F", "last_row_with_any_A_to_F",
                  "last_row_with_any_cell", "data_rows_with_any_A_to_F", "rows_2_to_N_without_any_A_to_F",
                  "rows_with_content_only_outside_A_to_F",
                  "populated_roles", "in_list", "outside_list", "distinct_nonempty_labels",
                  "distinct_outside_labels", "outside_case_insensitive_matches",
                  "populated_F_raw_differs_from_stripped", "blank_role_but_other_A_to_F_populated",
                  "blank_role_rows", "all_dataValidation_elements_in_sheet", "all_validations_parsed",
                  "validations_not_touching_F", "F_rows_2_to_N_not_covered", "F_populated_not_covered",
                  "F_populated_anywhere_not_covered", "hidden_rows_in_2_to_N", "option_count",
                  "in_list_by_option", "top_outside_labels", "outside_label_shape",
                  "F_rowvalue_fingerprint_sha256", "cell_type_stats", "sharedStrings", "sheet",
                  "first_last_populated_F", "sample_cells", "sample_outside_cells"]:
            print("  %s: %s" % (k, json.dumps(r.get(k), ensure_ascii=False)))
        for v in r.get("validations_touching_F", []):
            vv = dict(v)
            print("  F-VALIDATION:", json.dumps(vv, ensure_ascii=False))


if __name__ == "__main__":
    main()
