import sys, json
from pathlib import Path
P = Path('<SCRATCH>/repos/math/specifications/lean_statement_proofs_20261011')
sys.path.insert(0, str(P))
import check
registered, _, _ = check.load_register((P / check.STATEMENT_DIR).resolve())
W = Path('<SCRATCH>/lvs/round4/work')
for stem in ('Side24Analysis', 'Side24Interfaces', 'RNCluster'):
    text = (W / (stem + '.lean')).read_text(encoding='utf-8')
    try:
        s = check.scan_module(text, stem, registered)
    except ValueError as e:
        print(stem, 'SCAN_FAIL', e); continue
    print(stem, 'SCAN_OK imports', s['imports'], 'aux_theorems', s['aux_theorems'], 'aux_defs', s['auxiliary_definitions'], 'set_options', s['set_options'])
    for t in s['theorems']:
        print('   ', t['line'], t['name'], '->', t['proves'], 'args', t['arguments'])
