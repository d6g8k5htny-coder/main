import os, sys, pprint
pop = sys.argv[1]
sys.path.insert(0, pop)
import test_global_h0_runner as T
T.MainLifecycle.setUpClass()
case = T.MainLifecycle('test_main_orders_freeze_raw_verification_lookup_normalization_and_exclusive_writes')
result, emitted, events, budgets = case.run_synthetic(fault='height_disagree')
c = result['certificate']
row = c['fixtures'][0]
print('top status:', c['status']); print('failure:', c['failure'])
print('nested:', sorted(set(T.nested_statuses(c))))
print('expected read:', ('expected', 'TEST_ONLY') in events)
print('emitted:', None if emitted is None else sorted(emitted))
print('row status:', row['status'])
print('comparison:', row.get('comparison'))
print('field_checks:'); pprint.pprint(row['field_checks'])
print('soundness:'); pprint.pprint(row['soundness'])
print('root_box evals:'); pprint.pprint([e for e in row['field_evaluations'] if e['route'] == 'root_box'])
print('work field_calls/poly:', c['work']['field_calls'], c['work']['polynomial_evaluations'])
