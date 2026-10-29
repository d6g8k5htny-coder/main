"""Finite source-mutant guards for the output-only sequential Math adapter.
Runs disposable stdlib fixtures only. A kill requires a real assertion failure,
not syntax/import/harness error; the unmutated copied harness must pass first.
"""
from pathlib import Path
import importlib.util
import io
import json
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'run_math_sequential_gate.py'
TEST = HERE / 'test_math_sequential_driver.py'


def load_tests():
    spec = importlib.util.spec_from_file_location('math_driver_mutation_tests', TEST)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def main():
    source = DRIVER.read_text()
    mutations = [
        ('direct_lean_launch', "if command[:3]!=['lake','env','lean']:return list(command),ENV,None", 'if True:return list(command),ENV,None', 'test_lake_environment_launch_preserves_pinned_binary_args_and_values'),
        ('environment_unknown_fields', "require(isinstance(captured,dict)and set(captured)==set(keys),'Lake environment field set')", "require(True,'disabled environment field set')", 'test_lake_environment_unknown_fields_fail_closed'),
        ('environment_duplicates', "require(key not in result,'duplicate Lake environment key')", "require(True,'disabled duplicate key guard')", 'test_lake_environment_duplicate_keys_fail_closed'),
        ('environment_capture_failure', "require(code==0,'Lake environment capture failed')", "require(True,'disabled capture failure')", 'test_lake_environment_capture_failure_cannot_launch'),
        ('environment_toolchain', "require(captured['LEAN_SYSROOT']==str(T.parent)and captured['LAKE']==str(T/'lake'),'Lake environment toolchain')", "require(True,'disabled toolchain identity')", 'test_lake_environment_sysroot_must_match_pinned_toolchain'),
        ('environment_values', 'else:env[key]=value', "else:env[key]='wrong'", 'test_lake_environment_launch_preserves_pinned_binary_args_and_values'),
        ('environment_cwd_cache', 'if cache_key not in LAKE_ENVIRONMENTS:', 'if True:', 'test_lake_environment_capture_cached_by_working_directory'),
        ('monitor_exception_cleanup', 'if reason:\n    try:stop_group(p)', "if reason and reason!='MONITOR_EXCEPTION':\n    try:stop_group(p)", 'test_monitor_error_cleans_direct_child_before_reraise'),
        ('refused_inner_receipt', "if live_receipt.exists():shutil.move(str(live_receipt),OUT/'refused-inner-receipt.json')", "if False:shutil.move(str(live_receipt),OUT/'refused-inner-receipt.json')", 'test_refused_final_seal_retires_inner_success_receipt'),
        ('orphan_guard', "if live:reason='DESCENDANTS_AFTER_EXIT'", "if False:reason='DESCENDANTS_AFTER_EXIT'", 'test_direct_child_exit_with_live_descendant_is_refused_and_cleaned'),
        ('rss_guard', 'if rss>RSS_LIMIT:', 'if False:', 'test_rss_guard_is_observed'),
        ('timeout_guard', 'if time.monotonic()-start>TIME_LIMIT:', 'if False:', 'test_timeout_refuses_and_records_only_own_process_group'),
        ('resource_refusal', "require(reason is None,'resource boundary: '+label+' '+str(reason))", "require(True,'disabled resource refusal')", 'test_rss_guard_is_observed'),
        ('module_build_failure', "require(code==0,'registered module build failed: '+module)", "require(True,'disabled module failure')", 'test_failed_module_build_cannot_be_passing'),
        ('kernel_failure', "require(code==0,'exact named-module kernel replay failed: '+module)", "require(True,'disabled kernel failure')", 'test_failed_exact_kernel_cannot_be_passing'),
        ('phase_expectation', "require((code==0)==expect_success,'unexpected original gate phase outcome: '+label)", "require(True,'disabled original expectation')", 'test_original_negative_phase_expectation_is_preserved'),
        ('module_validation', "require(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*',module),'unsafe module name')", "require(True,'disabled module validation')", 'test_exact_module_name_validation_and_source'),
        ('kernel_command', "['lake','env','lean',str(source)]", "['lake','env','leanchecker',module]", 'test_exact_module_name_validation_and_source'),
        ('log_identity', "hashlib.sha256(path.read_bytes()).hexdigest()", "'0'*64", 'test_success_is_observed_and_log_hash_bound'),
    ]
    baseline = load_tests()
    baseline_log = io.StringIO()
    baseline_result = unittest.TextTestRunner(stream=baseline_log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(baseline.DriverGuardTests))
    (HERE / 'sequential-driver-mutation-baseline.log').write_text(baseline_log.getvalue())
    if not baseline_result.wasSuccessful() or baseline_result.testsRun != 21 or baseline_result.skipped:
        raise RuntimeError('unmutated harness did not pass all21 cases')
    rows = []
    for name, old, new, test in mutations:
        if source.count(old) != 1:
            raise RuntimeError('missing/ambiguous source mutation: ' + name)
        with tempfile.TemporaryDirectory(prefix='cbz-math-source-mutant-') as temp:
            path = Path(temp) / 'mutant.py'
            path.write_text(source.replace(old, new))
            tests = load_tests()
            tests.DRIVER = path
            log = io.StringIO()
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.TestSuite([tests.DriverGuardTests(test)]))
            (HERE / ('sequential-driver-mutation-' + name + '.log')).write_text(log.getvalue())
            rows.append({'name':name,'test':test,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'killed':result.testsRun==1 and len(result.failures)==1 and not result.errors and not result.skipped})
    report={'scope':'finite engineering guard mutations only;not theorem or fullLayerA certification','baseline_tests':baseline_result.testsRun,'baseline_pass':baseline_result.wasSuccessful(),'declared_threshold_numerator':19,'declared_threshold_denominator':19,'mutants':rows}
    (HERE / 'sequential-driver-mutations.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if all(r['killed'] for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
