import sys, zipfile, hashlib, re, json, collections
import xml.etree.ElementTree as ET

M = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
PR = '{http://schemas.openxmlformats.org/package/2006/relationships}'
OPTS = ["PRIMARY_DERIVATION","EXECUTABLE_RECEIPT","RAW_DATA","FORMAL_PROOF","INDEPENDENT_RECONSTRUCTION","SUMMARY","NAVIGATION","BALLOT","REGISTER_COPY","REVIEW_OF_REVIEW","GOVERNANCE_PRIMARY"]

def col_of(ref):
    return re.match(r'([A-Z]+)', ref).group(1)

def row_of(ref):
    return int(re.match(r'[A-Z]+(\d+)', ref).group(1))

def si_text(si):
    out = []
    for ch in si:
        if ch.tag == M+'t':
            out.append(ch.text or '')
        elif ch.tag == M+'r':
            for t in ch.findall(M+'t'):
                out.append(t.text or '')
    return ''.join(out)

def main(path):
    raw = open(path,'rb').read()
    res = {'file': path, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    z = zipfile.ZipFile(path)
    names = z.namelist()
    res['has_docProps'] = any(n.startswith('docProps/') for n in names)
    res['n_worksheets'] = sum(1 for n in names if re.match(r'xl/worksheets/sheet\d+\.xml$', n))
    wbx = z.read('xl/workbook.xml')
    wb = ET.fromstring(wbx)
    rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    rmap = {r.get('Id'): r.get('Target') for r in rels.findall(PR+'Relationship')}
    sheets = [(s.get('name'), s.get('sheetId'), s.get(R+'id')) for s in wb.iter(M+'sheet')]
    target = None
    for nm, sid, rid in sheets:
        if nm == 'Evidence Lineage':
            target = 'xl/' + rmap[rid].lstrip('/').replace('xl/','')
            res['sheet_decl'] = {'name': nm, 'sheetId': sid, 'rid': rid, 'target': rmap[rid]}
    dn = wb.find(M+'definedNames')
    dnl = [] if dn is None else [(d.get('name'), d.get('localSheetId'), d.text) for d in dn.findall(M+'definedName')]
    res['definedNames_count'] = len(dnl)
    res['definedNames_raw_regex_count'] = len(re.findall(rb'<definedName[ >]', wbx))
    res['definedNames_on_EL'] = [d for d in dnl if d[2] and 'Evidence Lineage' in d[2]]
    res['definedNames_name_set'] = sorted(collections.Counter(d[0] for d in dnl).items())
    sx = z.read(target)
    res['member'] = target
    res['member_bytes'] = len(sx)
    res['member_sha256'] = hashlib.sha256(sx).hexdigest()
    # raw structural features
    for tag in [b'<dataValidations', b'<dataValidation ', b'<extLst', b'x14:dataValidation', b'<conditionalFormatting', b'<sheetProtection', b'<autoFilter', b'<dimension', b'<hyperlinks', b'<legacyDrawing', b'<drawing', b'<tableParts']:
        res['count_' + tag.decode().strip('<').strip()] = sx.count(tag)
    m = re.search(rb'<dataValidations[^>]*>.*?</dataValidations>', sx, re.S)
    res['dvs_span'] = [m.start(), m.end()] if m else None
    res['dvs_open_tag'] = re.search(rb'<dataValidations[^>]*>', sx).group(0).decode() if m else None
    dvs = [(mm.start(), mm.end(), mm.group(0).decode()) for mm in re.finditer(rb'<dataValidation [^>]*>.*?</dataValidation>', sx, re.S)]
    res['dv_raw'] = dvs
    af = re.search(rb'<autoFilter[^>]*>', sx)
    res['autoFilter'] = af.group(0).decode() if af else None
    # sheet rels
    relp = 'xl/worksheets/_rels/' + target.split('/')[-1] + '.rels'
    if relp in names:
        rr = ET.fromstring(z.read(relp))
        res['sheet_rels_types'] = sorted(collections.Counter(r.get('Type').split('/')[-1] for r in rr.findall(PR+'Relationship')).items())
    else:
        res['sheet_rels_types'] = None
    # whole-workbook validations
    tot_dv = 0; tot_x14 = 0; dv_with_opts = []
    for n in names:
        if re.match(r'xl/worksheets/sheet\d+\.xml$', n):
            b = z.read(n)
            tot_dv += len(re.findall(rb'<dataValidation[ >]', b))
            tot_x14 += len(re.findall(rb'<x14:dataValidation[ >]', b))
            if b'PRIMARY_DERIVATION' in b and b'<dataValidation' in b:
                for mm in re.finditer(rb'<dataValidation [^>]*>.*?</dataValidation>', b, re.S):
                    if b'PRIMARY_DERIVATION' in mm.group(0):
                        dv_with_opts.append(n)
    res['workbook_dataValidation_elems'] = tot_dv
    res['workbook_x14_elems'] = tot_x14
    res['sheets_with_role_list_dv'] = dv_with_opts
    # shared strings
    sst = ET.fromstring(z.read('xl/sharedStrings.xml'))
    sis = [si_text(si) for si in sst.findall(M+'si')]
    res['sst_count'] = len(sis); res['sst_uniqueCount'] = sst.get('uniqueCount')
    res['sst_contains'] = {o: (o in set(sis)) for o in OPTS}
    sh = ET.fromstring(sx)
    cells = collections.defaultdict(dict)
    formulas = 0; types = collections.Counter()
    for c in sh.iter(M+'c'):
        ref = c.get('r'); t = c.get('t', 'n')
        if c.find(M+'f') is not None: formulas += 1
        v = c.find(M+'v'); val = None
        if t == 's' and v is not None: val = sis[int(v.text)]
        elif t == 'inlineStr':
            is_ = c.find(M+'is'); val = si_text(is_) if is_ is not None else None
        elif v is not None: val = v.text
        if val is not None:
            types[t] += 1
            cells[row_of(ref)][col_of(ref)] = val
    res['formula_cells'] = formulas
    res['header'] = [cells[1].get(c) for c in 'ABCDEFGHIJKLMNOP']
    def ne(x): return x is not None and x.strip() != ''
    data_rows = [r for r in sorted(cells) if r >= 2 and any(ne(cells[r].get(c)) for c in 'ABCDEF')]
    res['last_data_row'] = max(data_rows)
    res['rows_any_AF'] = len(data_rows)
    allrows = [r for r in sorted(cells) if r >= 2 and any(ne(v) for v in cells[r].values())]
    res['rows_any_any'] = len(allrows)
    empties = [r for r in range(2, res['last_data_row']+1) if not any(ne(v) for v in cells.get(r, {}).values())]
    res['empty_rows_in_range'] = empties
    F = [(r, cells[r].get('F')) for r in data_rows]
    pop = [(r, v) for r, v in F if ne(v)]
    res['populated'] = len(pop)
    res['blank_F_with_AF'] = len(F) - len(pop)
    res['ws_diff'] = sum(1 for r, v in pop if v != v.strip())
    inl = [(r, v) for r, v in pop if v.strip() in OPTS]
    res['in_list'] = len(inl); res['outside'] = len(pop) - len(inl)
    res['in_list_rows'] = [r for r, v in inl]
    res['per_option'] = {o: sum(1 for r, v in inl if v.strip() == o) for o in OPTS}
    res['distinct'] = len(set(v.strip() for r, v in pop))
    out = collections.Counter(v.strip() for r, v in pop if v.strip() not in OPTS)
    res['distinct_outside'] = len(out)
    res['outside_freq_hist'] = sorted(collections.Counter(out.values()).items(), reverse=True)
    res['top_outside_ge3'] = sorted([(k, n) for k, n in out.items() if n >= 3], key=lambda x: (-x[1], x[0]))
    res['ci_matches_outside'] = sum(1 for r, v in pop if v.strip() not in OPTS and v.strip().upper() in OPTS)
    res['F_beyond_2001_or_lt2'] = [r for r, v in pop if r < 2 or r > 2001]
    gp = [(r, cells[r].get('A'), cells[r].get('B'), cells[r].get('G')) for r, v in pop if v.strip() == 'GOVERNANCE_PRIMARY']
    res['GOVERNANCE_PRIMARY_rows'] = gp
    res['EXACT_DERIVATION_rows'] = [(r, cells[r].get('A'), cells[r].get('G')) for r, v in pop if v.strip() == 'EXACT_DERIVATION']
    res['PRIMARY_NUMERICAL_SOURCE_rows'] = [r for r, v in pop if v.strip() == 'PRIMARY_NUMERICAL_SOURCE']
    res['sample'] = {r: (cells[r].get('A'), cells[r].get('F')) for r in [2, 3, 9, 11, 12, 52, 155, 156, 188, 349, 355, 385, 386, 464, 517, 518, 528, 529] if r in cells}
    res['F594'] = cells.get(594, {}).get('F')
    acount = collections.Counter(cells[r].get('A') for r in data_rows if ne(cells[r].get('A')))
    res['dup_A'] = [(k, n) for k, n in acount.items() if n > 1]
    fp_lines = '\n'.join('%d\t%s' % (r, v) for r, v in pop)
    res['fp_pop'] = hashlib.sha256(fp_lines.encode()).hexdigest()
    fp2 = '\n'.join('%d\t%s' % (r, cells.get(r, {}).get('F') or '') for r in range(2, res['last_data_row']+1))
    res['fp_all'] = hashlib.sha256(fp2.encode()).hexdigest()
    res['_rowsAP'] = {r: [cells[r].get(c) for c in 'ABCDEFGHIJKLMNOP'] for r in data_rows}
    return res

if __name__ == '__main__':
    outs = [main(p) for p in sys.argv[1:-1]]
    json.dump(outs, open(sys.argv[-1], 'w'), indent=1, ensure_ascii=False)
    for o in outs:
        d = {k: v for k, v in o.items() if k not in ('_rowsAP', 'dv_raw')}
        print(json.dumps(d, ensure_ascii=False)[:6000])
        for s, e, t in o['dv_raw']:
            if 'PRIMARY_DERIVATION' in t:
                print('DV', s, e, t)
        print('----')
