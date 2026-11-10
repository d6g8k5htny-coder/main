"""Structural security contract for the narrow Pages recovery workflow."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/cbz-site-recovery.yml'


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate workflow key')
        result[key] = value
    return result


class RecoveryWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(WORKFLOW.is_file(), 'recovery workflow is not implemented')
        self.workflow = json.loads(WORKFLOW.read_text(), object_pairs_hook=unique_pairs)

    def test_only_pr_controls_and_explicit_dispatch_are_triggered(self):
        trigger = self.workflow['on']
        self.assertEqual(set(trigger), {'pull_request', 'workflow_dispatch'})
        self.assertEqual(set(trigger['workflow_dispatch']['inputs']),
                         {'expected_failed_head', 'expected_previous_head'})
        self.assertTrue(all(value['required'] is True and value['type'] == 'string'
                            for value in trigger['workflow_dispatch']['inputs'].values()))
        self.assertEqual(self.workflow['permissions'], {})
        self.assertEqual(set(trigger['pull_request']['paths']), {
            'tools/cbz_site_receipt_recovery.py', 'tools/cbz_site_recovery_mutations.py',
            'tools/cbz/reference_crawler_mutation_check.py', 'tools/current_site_publication.py',
            'tools/current_site_data.py', 'tests/test_cbz_site_recovery.py',
            'tests/test_cbz_site_recovery_workflow.py', 'tests/test_current_site_publication.py',
            '.github/workflows/cbz-site-recovery.yml'})

    def test_dispatch_serializes_with_the_normal_pages_publication_group(self):
        group = self.workflow['concurrency']
        self.assertFalse(group['cancel-in-progress'])
        self.assertIn('current-research-pages-publish', group['group'])
        self.assertIn("github.event_name == 'pull_request'", group['group'])
        self.assertIn('github.event.pull_request.number', group['group'])

    def test_write_job_requires_successful_controls_and_exact_dispatch_branch(self):
        jobs = self.workflow['jobs']
        self.assertEqual(set(jobs), {'controls', 'recover'})
        recovery = jobs['recover']
        self.assertEqual(recovery['needs'], 'controls')
        self.assertEqual(recovery['permissions'], {'actions': 'read', 'contents': 'write', 'pages': 'read'})
        self.assertIn("github.event_name == 'workflow_dispatch'", recovery['if'])
        self.assertIn("github.ref == 'refs/heads/main'", recovery['if'])
        self.assertIn("github.repository == 'd6g8k5htny-coder/main'", recovery['if'])
        self.assertEqual(jobs['controls']['permissions'], {'contents': 'read'})

    def test_actions_are_pinned_and_cannot_continue_on_error(self):
        for job in self.workflow['jobs'].values():
            self.assertNotIn('continue-on-error', job)
            for step in job['steps']:
                self.assertNotIn('continue-on-error', step)
                if 'uses' in step:
                    self.assertRegex(step['uses'], r'^[\w.-]+/[\w.-]+@[0-9a-f]{40}$')

    def test_dispatch_inputs_are_environment_values_not_shell_expressions(self):
        recovery = self.workflow['jobs']['recover']
        step = next(step for step in recovery['steps'] if step.get('name') == 'Reconcile canceled publication')
        self.assertEqual(step['env']['EXPECTED_FAILED_HEAD'], '${{ inputs.expected_failed_head }}')
        self.assertEqual(step['env']['EXPECTED_PREVIOUS_HEAD'], '${{ inputs.expected_previous_head }}')
        self.assertNotIn('${{ inputs.', step['run'])
        self.assertIn('"$EXPECTED_FAILED_HEAD"', step['run'])
        self.assertIn('"$EXPECTED_PREVIOUS_HEAD"', step['run'])

    def test_controls_run_both_modes_without_skips_and_retain_failures(self):
        steps = self.workflow['jobs']['controls']['steps']
        run = next(step['run'] for step in steps if step.get('name') == 'Run recovery controls')
        self.assertIn('tests.test_cbz_site_recovery.SiteRecoveryTests', run)
        self.assertIn('tests.test_cbz_site_recovery_workflow.RecoveryWorkflowTests', run)
        self.assertIn('python3 -B -O -S', run)
        self.assertIn('cbz_site_recovery_mutations.py', run)
        self.assertIn('skipped', run)
        artifacts = [step for step in steps if 'uses' in step and step['uses'].startswith('actions/upload-artifact@')]
        self.assertEqual(len(artifacts), 1)
        self.assertIn('always()', artifacts[0]['if'])
        self.assertIs(artifacts[0]['with']['include-hidden-files'], True)

    def test_recovery_logs_are_retained_even_on_failure(self):
        steps = self.workflow['jobs']['recover']['steps']
        command = next(step['run'] for step in steps if step.get('name') == 'Reconcile canceled publication')
        self.assertIn('recovery.stdout', command)
        self.assertIn('recovery.stderr', command)
        self.assertIn('exit "$status"', command)
        artifact = next(step for step in steps if 'uses' in step and step['uses'].startswith('actions/upload-artifact@'))
        self.assertIn('always()', artifact['if'])
        self.assertIs(artifact['with']['include-hidden-files'], True)


if __name__ == '__main__':
    unittest.main()
