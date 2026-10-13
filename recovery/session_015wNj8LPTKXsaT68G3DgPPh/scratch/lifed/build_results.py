"""Build RESULTS.json from the packet script's core computation and the stored refined cap sweeps (which --check-full replays)."""
import json, sys, time, importlib.util, os
from fractions import Fraction as Fr
PKT = sys.argv[1]
spec = importlib.util.spec_from_file_location('tb', os.path.join(PKT, 'theorem_b.py'))
tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)
SCR = os.path.dirname(os.path.abspath(__file__))
res = {'schema': 1, 'object': 'CL-C8-LIFETIME-DIFFERENCE-CONSTANT-20261001-v1', 'scientific_effect': 'NONE',
       'certified': True, 'mutant': None, 'parameters': tb.params(), 'dims': {}}
for d in (2, 3):
    t = time.time()
    summ, rate, ref = tb.compute_core(d)
    cap = json.load(open(os.path.join(SCR, 'cap_sweep_d%d.json' % d)))
    boxes = [[str(bb[0]), str(bb[1]), str(kb[0]), str(kb[1])] for bb, kb in tb.CAP_BOXES]
    assert cap['boxes'] == boxes, 'cap boxes order'
    bl = tb.ENG[d].bands()
    assert [rec['r'] for rec in cap['bands']] == [[str(a), str(b)] for a, b in bl], 'band order'
    res['dims'][str(d)] = {'tm_law': summ, 'rate': rate, 'reference': ref, 'table': tb.assemble(d, rate, ref, cap['bands']),
                           'cap_boxes': boxes, 'cap_bands': cap['bands']}
    print('d=%d %.1fs' % (d, time.time() - t), flush=True)
with open(os.path.join(PKT, 'RESULTS.json'), 'w') as fh:
    fh.write(tb.results_text(res))
print(json.dumps({d: {R: (row['C_BK'], row['rel2_up'], row['rel2_dn']) for R, row in res['dims'][d]['table'].items()} for d in res['dims']}, indent=1))
