"""Run the packet's own check.py scan_module (lexer, refusal list, allowed commands, labels,
Source line parse, verbatim anchors) on the two modules, plus an exact-once anchor count."""
import sys, json, pathlib, importlib.util
ROOT = pathlib.Path('<SCRATCH>/repos/math/specifications/lean_statements_20261010')
REPO = pathlib.Path('<SCRATCH>/repos/math')
spec = importlib.util.spec_from_file_location('pkt_check', ROOT / 'check.py')
chk = importlib.util.module_from_spec(spec); spec.loader.exec_module(chk)
reg = json.loads((ROOT / 'REGISTER.json').read_text(encoding='utf-8'))
sources = chk.load_sources(ROOT, REPO, reg)
if isinstance(sources, tuple): sources = sources[0]
print('sources loaded:', len(sources))
total = 0
for stem in ('Field', 'Lifetime'):
    text = (ROOT / 'Specifications' / f'{stem}.lean').read_text(encoding='utf-8')
    decls = chk.scan_module(text, stem, sources)
    kinds = {}
    for d in decls: kinds[d['kind']] = kinds.get(d['kind'], 0) + 1
    print(f'{stem}: scan_module OK, {len(decls)} declarations, kinds {kinds}')
    bad = 0
    for d in decls:
        if d['kind'] is None:
            continue
        src = sources[d['source_id']]['text']
        c = src.count(d['doc_anchor'])
        if c != 1:
            bad += 1; print('   anchor count', c, d['name'])
    print(f'   registered-kind declarations with exactly-once verbatim anchors: {sum(1 for d in decls if d["kind"])-bad}/{sum(1 for d in decls if d["kind"])}')
    total += len(decls)
print('TOTAL', total)
