"""Bounded source mutations graded by the existing live-traceback recorder."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / 'tools/cbz_site_receipt_recovery.py'
TEST = ROOT / 'tests/test_cbz_site_recovery.py'
FIXTURE = ROOT / 'tests/test_current_site_publication.py'
PUBLICATION = ROOT / 'tools/current_site_publication.py'
PUBLICATION_DATA = ROOT / 'tools/current_site_data.py'
RECORDER = ROOT / 'tools/cbz/reference_crawler_mutation_check.py'
WORKFLOW = ROOT / '.github/workflows/cbz-site-recovery.yml'
WORKFLOW_TEST = ROOT / 'tests/test_cbz_site_recovery_workflow.py'
RECORDER_SHA256 = '14b770828aa566b529be64214a0d99e5cfa4b762be792f07af04b39985ce0481'

MUTANTS = [
    ('cas-before-oid', 'test_atomic_ref_request_binds_expected_head_and_nonforce',
     "'beforeOid': before", "'beforeOid': after", 1),
    ('cas-force', 'test_atomic_ref_request_binds_expected_head_and_nonforce',
     "'afterOid': after, 'force': False", "'afterOid': after, 'force': True", 1),
    ('cas-repository', 'test_atomic_ref_request_binds_expected_head_and_nonforce',
     "'repositoryId': self.REPOSITORY_ID", "'repositoryId': 'wrong-repository'", 1),
    ('cas-response-error', 'test_atomic_ref_request_rejects_errors_and_identity_mismatch',
     "'errors' in obj", 'False', 1),
    ('cas-response-origin', 'test_atomic_ref_request_rejects_errors_and_identity_mismatch',
     'response.geturl() != self.ORIGIN', 'False', 1),
    ('cas-response-status', 'test_atomic_ref_request_rejects_errors_and_identity_mismatch',
     'response.status != 200', 'False', 1),
    ('cas-invalid-head', 'test_atomic_ref_request_rejects_invalid_heads_before_network',
     "before == '0' * 40", 'False', 1),
    ('cas-adapter-bypass', 'test_ref_rollback_after_guard_does_not_accept_fast_forward',
     "self.ref_advance.advance(self.expected, body['sha'])", 'self.api.request(method, path, body)', 1),
    ('cas-second-parent', 'test_success_uses_new_expected_head_for_each_append',
     "GuardedRefAPI(api, ref_advance, intermediate['head']), observed", "GuardedRefAPI(api, ref_advance, failed['head']), observed", 1),
    ('expected-head', 'test_wrong_expected_heads_refuse_before_any_write',
     "failed['head'] != expected_failed_head or failed['receipt']['phase'] != 'prepared'",
     "False or failed['receipt']['phase'] != 'prepared'", 1),
    ('workflow-status', 'test_running_or_successful_workflow_refuses',
     "run.get('status') != 'completed'", 'False', 1),
    ('workflow-outcome', 'test_running_or_successful_workflow_refuses',
     "run.get('conclusion') not in ('failure', 'cancelled')", 'False', 1),
    ('workflow-source', 'test_wrong_workflow_source_or_boolean_attempt_refuses',
     "run.get('head_sha') != candidate['source']['commit']", 'False', 1),
    ('attempt-type', 'test_wrong_workflow_source_or_boolean_attempt_refuses',
     "type(run.get('run_attempt')) is not int", 'False', 1),
    ('deployment-status', 'test_nonterminal_or_unknown_deployment_refuses',
     "deployment.get('status') != 'deployment_cancelled'", 'False', 1),
    ('main-head', 'test_current_main_mismatch_refuses',
     "main_head != context['commit']", 'False', 1),
    ('snapshot-retention', 'test_lost_prepared_snapshot_refuses',
     "not set(failed_candidate['snapshot_ids']) <= set(prior_candidate['snapshot_ids'])", 'False', 1),
    ('prior-prepared-binding', 'test_prior_observed_candidate_must_match_its_prepared_parent',
     "publication.candidate_of(old_prepared['receipt']) != publication.candidate_of(previous['receipt'])", 'False', 1),
    ('first-public-readback', 'test_public_mismatch_refuses_without_writing',
     "first_public = publication.public_readback(previous['receipt'], reader=reader, attempts=attempts)", 'first_public = {}', 1),
    ('second-public-readback', 'test_second_public_mismatch_leaves_only_prepared_prior_candidate',
     "second_public = publication.public_readback(previous['receipt'], reader=reader, attempts=attempts)", 'second_public = {}', 1),
    ('second-platform-check', 'test_change_to_deployment_before_write_refuses',
     'platform_before_write = check_platform(platform, failed_candidate, context)', 'platform_before_write = None', 1),
    ('route-allowlist', 'test_read_only_platform_routes_reject_unrelated_paths',
     "if not (path == '/git/ref/heads/main'", "if False and not (path == '/git/ref/heads/main'", 1),
    ('response-url', 'test_platform_read_rejects_redirect_status_and_malformed_response',
     'response.geturl() != url', 'False', 1),
    ('response-status', 'test_platform_read_rejects_redirect_status_and_malformed_response',
     'response.status != 200', 'False', 2),
    ('prepared-event', 'test_success_records_actual_recovery_evidence_in_order',
     "emit({'stage': 'prepared', 'head': intermediate['head'], 'parent': failed['head']})", 'None', 1),
    ('public-digest', 'test_success_records_actual_recovery_evidence_in_order',
     'hashlib.sha256(data).hexdigest()', "'0' * 64", 1),
    ('dispatch-guard', 'test_cli_refuses_wrong_event_before_runtime_access',
     "os.environ.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'", 'False', 1),
]
WORKFLOW_MUTANTS = [
    ('extra-trigger', 'test_only_pr_controls_and_explicit_dispatch_are_triggered'),
    ('missing-trigger-path', 'test_only_pr_controls_and_explicit_dispatch_are_triggered'),
    ('cancel-publish', 'test_dispatch_serializes_with_the_normal_pages_publication_group'),
    ('write-permissions', 'test_write_job_requires_successful_controls_and_exact_dispatch_branch'),
    ('missing-main-guard', 'test_write_job_requires_successful_controls_and_exact_dispatch_branch'),
    ('missing-controls', 'test_write_job_requires_successful_controls_and_exact_dispatch_branch'),
    ('unpinned-action', 'test_actions_are_pinned_and_cannot_continue_on_error'),
    ('ignored-failure', 'test_actions_are_pinned_and_cannot_continue_on_error'),
    ('shell-input', 'test_dispatch_inputs_are_environment_values_not_shell_expressions'),
    ('hidden-evidence', 'test_recovery_logs_are_retained_even_on_failure'),
]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_occurrence(source, old, new, number):
    parts = source.split(old)
    need(0 < number < len(parts), 'mutation source not found')
    return old.join(parts[:number]) + new + old.join(parts[number:])


def load_recorder():
    need(digest(RECORDER.read_bytes()) == RECORDER_SHA256, 'origin recorder source changed')
    spec = importlib.util.spec_from_file_location('cbz_recovery_origin_recorder', RECORDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(recorder, source, test_source, name, case_dir):
    with tempfile.TemporaryDirectory(prefix='cbz-site-recovery-mutation-') as temp:
        work = Path(temp)
        (work / 'tools').mkdir()
        (work / 'tests').mkdir()
        (work / 'tests/__init__.py').write_bytes(b'')
        (work / 'tools/cbz_site_receipt_recovery.py').write_text(source)
        (work / 'tests/test_cbz_site_recovery.py').write_text(test_source)
        shutil.copyfile(PUBLICATION, work / 'tools/current_site_publication.py')
        shutil.copyfile(PUBLICATION_DATA, work / 'tools/current_site_data.py')
        shutil.copyfile(FIXTURE, work / 'tests/test_current_site_publication.py')
        selector = 'tests.test_cbz_site_recovery.SiteRecoveryTests.' + name
        outcome, detail, failed, errored = recorder.run_suite(work, [name], selector=selector, copy_test=False)
        report = recorder.LAST_ORIGIN_REPORT
        need(isinstance(report, dict), 'missing live provenance report')
        (case_dir / 'origin.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        return outcome, detail, failed, errored, report


def execute_workflow(recorder, workflow, tests, name, case_dir):
    with tempfile.TemporaryDirectory(prefix='cbz-site-recovery-workflow-') as temp:
        work = Path(temp)
        (work / 'tests').mkdir()
        (work / 'tests/__init__.py').write_bytes(b'')
        (work / 'tests/test_cbz_site_recovery_workflow.py').write_text(tests)
        (work / '.github/workflows').mkdir(parents=True)
        (work / '.github/workflows/cbz-site-recovery.yml').write_text(workflow)
        selector = 'tests.test_cbz_site_recovery_workflow.RecoveryWorkflowTests.' + name
        outcome, detail, failed, errored = recorder.run_suite(work, [name], selector=selector, copy_test=False)
        report = recorder.LAST_ORIGIN_REPORT
        need(isinstance(report, dict), 'missing workflow provenance report')
        (case_dir / 'origin.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        return outcome, detail, failed, errored, report


def mutate_workflow(original, label):
    workflow = copy.deepcopy(original)
    controls, recovery = workflow['jobs']['controls'], workflow['jobs']['recover']
    if label == 'extra-trigger':
        workflow['on']['push'] = {}
    elif label == 'missing-trigger-path':
        workflow['on']['pull_request']['paths'].remove('tests/test_current_site_publication.py')
    elif label == 'cancel-publish':
        workflow['concurrency']['cancel-in-progress'] = True
    elif label == 'write-permissions':
        recovery['permissions']['pages'] = 'write'
    elif label == 'missing-main-guard':
        recovery['if'] = recovery['if'].replace("github.ref == 'refs/heads/main'", 'true')
    elif label == 'missing-controls':
        recovery['needs'] = []
    elif label == 'unpinned-action':
        controls['steps'][0]['uses'] = 'actions/checkout@v4'
    elif label == 'ignored-failure':
        recovery['steps'][1]['continue-on-error'] = True
    elif label == 'shell-input':
        recovery['steps'][1]['run'] += '\necho ${{ inputs.expected_failed_head }}\n'
    elif label == 'hidden-evidence':
        recovery['steps'][-1]['with']['include-hidden-files'] = False
    else:
        raise ValueError('unknown workflow mutation')
    return json.dumps(workflow, indent=2) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    recorder = load_recorder()
    subject, tests = SUBJECT.read_text(), TEST.read_text()
    results = []
    for label, name, old, new, occurrence in MUTANTS:
        directory = out / label
        directory.mkdir()
        baseline_dir = directory / 'baseline'; baseline_dir.mkdir()
        result = execute(recorder, subject, tests, name, baseline_dir)
        need(result[0] == 'pass' and not result[2] and not result[3], 'baseline failed ' + label)
        mutant = replace_occurrence(subject, old, new, occurrence)
        need(mutant != subject, 'mutation unchanged ' + label)
        (directory / 'mutant.py').write_text(mutant)
        mutant_dir = directory / 'mutant'; mutant_dir.mkdir()
        result = execute(recorder, mutant, tests, name, mutant_dir)
        need(result[0] == 'fail' and result[2] == {name} and not result[3], 'mutation not killed by selected oracle ' + label)
        results.append({'mutation': label, 'selected': name, 'mutant_sha256': digest(mutant.encode()),
                        'baseline': 'pass', 'mutant': 'authorized-selected-failure'})
    workflow_raw, workflow_tests = WORKFLOW.read_text(), WORKFLOW_TEST.read_text()
    workflow = json.loads(workflow_raw)
    for label, name in WORKFLOW_MUTANTS:
        directory = out / ('workflow-' + label); directory.mkdir()
        baseline_dir = directory / 'baseline'; baseline_dir.mkdir()
        result = execute_workflow(recorder, workflow_raw, workflow_tests, name, baseline_dir)
        need(result[0] == 'pass' and not result[2] and not result[3], 'workflow baseline failed ' + label)
        mutant = mutate_workflow(workflow, label)
        need(mutant != workflow_raw, 'workflow mutation unchanged ' + label)
        (directory / 'mutant.yml').write_text(mutant)
        mutant_dir = directory / 'mutant'; mutant_dir.mkdir()
        result = execute_workflow(recorder, mutant, workflow_tests, name, mutant_dir)
        need(result[0] == 'fail' and result[2] == {name} and not result[3],
             'workflow mutation not killed by selected oracle ' + label)
        results.append({'mutation': 'workflow-' + label, 'selected': name, 'mutant_sha256': digest(mutant.encode()),
                        'baseline': 'pass', 'mutant': 'authorized-selected-failure'})
    # Actual subject/helper AssertionErrors must be rejected as origins, not kills.
    adverse = [
        ('subject-assertion', '    check_context(context)\n',
         '    raise AssertionError("subject-origin adverse probe")\n    check_context(context)\n'),
        ('helper-assertion', '    main_head = platform.main_head()\n',
         '    raise AssertionError("helper-origin adverse probe")\n    main_head = platform.main_head()\n'),
    ]
    selected = 'test_current_main_mismatch_refuses'
    for label, old, new in adverse:
        directory = out / label; directory.mkdir()
        mutant = replace_occurrence(subject, old, new, 1)
        (directory / 'mutant.py').write_text(mutant)
        child = directory / 'child'; child.mkdir()
        result = execute(recorder, mutant, tests, selected, child)
        report = result[4]
        need(result[0] == 'invalid' and report['failure_count'] == 1 and report['error_count'] == 0
             and report['unauthorized_failures'] == [selected] and not report['authorized_failures'],
             'subject assertion was not rejected ' + label)
        results.append({'mutation': label, 'selected': selected, 'mutant_sha256': digest(mutant.encode()),
                        'mutant': 'rejected-unauthorized-origin'})
    label = 'workflow-setup-assertion'
    directory = out / label; directory.mkdir()
    test_mutant = replace_occurrence(workflow_tests, "        self.assertTrue(WORKFLOW.is_file(), 'recovery workflow is not implemented')\n",
        "        raise AssertionError('lifecycle-origin adverse probe')\n", 1)
    (directory / 'test-mutant.py').write_text(test_mutant)
    child = directory / 'child'; child.mkdir()
    selected = 'test_only_pr_controls_and_explicit_dispatch_are_triggered'
    result = execute_workflow(recorder, workflow_raw, test_mutant, selected, child)
    report = result[4]
    need(result[0] == 'invalid' and report['failure_count'] == 1 and report['error_count'] == 0
         and report['unauthorized_failures'] == [selected] and not report['authorized_failures'],
         'lifecycle assertion was not rejected')
    results.append({'mutation': label, 'selected': selected, 'mutant_sha256': digest(test_mutant.encode()),
                    'mutant': 'rejected-unauthorized-origin'})
    need(len(results) == len(MUTANTS) + len(WORKFLOW_MUTANTS) + len(adverse) + 1, 'incomplete campaign')
    (out / 'results.json').write_text(json.dumps({'optimize': sys.flags.optimize,
        'subject_sha256': digest(subject.encode()), 'test_sha256': digest(tests.encode()),
        'workflow_sha256': digest(workflow_raw.encode()), 'workflow_test_sha256': digest(workflow_tests.encode()),
        'recorder_sha256': RECORDER_SHA256, 'results': results}, indent=2) + '\n')
    print(len(MUTANTS), 'source and', len(WORKFLOW_MUTANTS), 'workflow clean baselines and selected-oracle mutations;',
          len(adverse) + 1, 'subject/helper/lifecycle assertion origins rejected; optimize', sys.flags.optimize)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
