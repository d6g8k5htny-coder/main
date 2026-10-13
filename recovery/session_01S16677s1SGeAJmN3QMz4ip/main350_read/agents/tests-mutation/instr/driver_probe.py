import os, sys
driver, fault = sys.argv[1], sys.argv[2]
os.environ['GLOBAL_SELECTOR_RUNNER_PATH'] = driver
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_runner_test as T
T.MainLifecycle.setUpClass()
case = T.MainLifecycle('test_main_orders_freeze_raw_verification_lookup_normalization_and_exclusive_writes')
result, emitted, events, budgets = case.run_synthetic(fault=fault)
c = result['certificate']
nested = sorted(set(T.nested_statuses(c)))
row = c['fixtures'][0] if c['fixtures'] else {}
print(f"driver={os.path.basename(os.path.dirname(driver))} fault={fault}: top={c['status']} failure={c.get('failure',{}).get('detail')!r}")
print(f"   nested statuses={nested}")
if row:
    print(f"   fixture.selection.status={row.get('selection',{}).get('status')} barcode={row.get('selection',{}).get('actual_barcode',{}).get('status')} field_checks={row.get('field_checks',{}).get('checks',{}).get('core_identity_overlap')} soundness.core_identity_overlap={row.get('soundness',{}).get('core_identity_overlap')}")
