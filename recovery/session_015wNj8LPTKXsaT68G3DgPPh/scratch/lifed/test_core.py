import sys, time, json
sys.path.insert(0, '/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/lifed/pkt')
import importlib.util
spec = importlib.util.spec_from_file_location('tb', '/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/lifed/pkt/theorem_b.py')
tb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tb)
for d in (2, 3):
    t = time.time()
    summ, rate, ref = tb.compute_core(d)
    print('d=%d core %.1fs' % (d, time.time() - t))
    for R, row in rate.items():
        print('  R=%s a_up=%s eps_up=%s a_dn=%s eps_dn=%s' % (R, row['a_up'], row['eps_up'], row['a_dn'], row['eps_dn']))
    print('  c_ref', ref['c_ref'])
    json.dump({'summ': summ, 'rate': rate, 'ref': ref}, open('/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/lifed/core_d%d.json' % d, 'w'))
