import copy, importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location('rc', root / 'tools/retrofit_record_check.py'); rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
base = rc.load_strict((root / 'reviews/retrofit_20261006/CONTRACT_EXAMPLE.json').read_text(encoding='utf-8'))
assert rc.validate(base) == []
def check(name, change):
    d = copy.deepcopy(base); change(d); e = rc.validate(d)
    print('%-44s %s' % (name, 'REJECTED' if e else 'ACCEPTED'), (e[0][:90] if e else ''))
def delta(d, i=0):
    d['records'][i]['delta'] = True; d['records'][i]['subject']['commit'] = 'b' * 40
check('V9  record id with trailing "!"', lambda d: d['records'][0].update(id='EX-001!'))
check('V11 unknown repository on a delta record', lambda d: (delta(d), d['records'][0]['subject'].update(repository='elsewhere')))
check('V12b 41-char commit on a delta record', lambda d: (delta(d), d['records'][0]['subject'].update(commit='b' * 41)))
check('V15 backslash in subject path', lambda d: d['records'][0]['subject'].update(path='reviews\\x.md'))
check('V23 not_applicable with evidence', lambda d: d['records'][0].update(state='not_applicable'))
check('V25 performer with an extra key', lambda d: d['records'][0]['performer'].update(extra='x'))
check('V26 whitespace-only exposure', lambda d: d['records'][0].update(exposure='   '))
check('V27 empty exposure', lambda d: d['records'][0].update(exposure=''))
check('V26b whitespace-only author session', lambda d: d['author'].update(session=' '))
