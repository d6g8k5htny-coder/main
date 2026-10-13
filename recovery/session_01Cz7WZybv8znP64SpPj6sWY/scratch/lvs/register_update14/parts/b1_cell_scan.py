def cell(s):
    return str(s).replace('|', '\\|').replace('\n', ' ')


scan = []
for m in MODULES:
    for d in check.scan_module((PK / m).read_text(encoding='utf-8'), PurePosixPath(m).stem, sources):
        scan.append({'module': m, **d})
order = {(r['module'], r['name']): (MODULES.index(r['module']), r['line']) for r in scan}
src_of = {(r['module'], r['name']): r['source_id'] for r in scan if r['kind']}
