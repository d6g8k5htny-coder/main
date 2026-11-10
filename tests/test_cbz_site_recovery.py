"""Fail-closed tests for recovering a canceled prepared Pages publication."""
import copy
import importlib.util
import inspect
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

from tests import test_current_site_publication as fixtures

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/cbz_site_receipt_recovery.py'


class Platform:
    def __init__(self):
        self.main = 'c' * 40
        self.run = {'id': 43, 'status': 'completed', 'conclusion': 'failure',
                    'run_attempt': 2, 'head_sha': 'b' * 40, 'head_branch': 'main',
                    'path': '.github/workflows/current-site.yml', 'event': 'push',
                    'repository': {'full_name': 'd6g8k5htny-coder/main'},
                    'head_repository': {'full_name': 'd6g8k5htny-coder/main'}}
        self.deployment = {'status': 'deployment_cancelled'}
        self.main_reads = 0
        self.change_main_on = None
        self.deployment_reads = 0
        self.change_deployment_on = None

    def main_head(self):
        self.main_reads += 1
        if self.main_reads == self.change_main_on:
            return 'd' * 40
        return self.main

    def workflow_run(self, run_id):
        if run_id != 43:
            raise ValueError('unexpected run')
        return copy.deepcopy(self.run)

    def pages_deployment(self, deployment_id):
        if deployment_id != 'b' * 40:
            raise ValueError('unexpected deployment')
        self.deployment_reads += 1
        if self.deployment_reads == self.change_deployment_on:
            return {'status': 'syncing_files'}
        return copy.deepcopy(self.deployment)


class SwitchReader(fixtures.PublicReader):
    def get_bytes(self, path):
        if len(self.calls) >= 3:
            self.calls.append(path)
            return b'changed public data'
        return super().get_bytes(path)


class Response:
    def __init__(self, url, body, status=200):
        self.url, self.body, self.status = url, body, status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self):
        return self.url

    def read(self, limit):
        return self.body[:limit]


class Opener:
    def __init__(self, body, *, url=None, status=200):
        self.body, self.url, self.status = body, url, status
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        return Response(self.url or request.full_url, self.body, self.status)


class SiteRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.is_file(), 'site recovery helper has not been implemented')
        spec = importlib.util.spec_from_file_location('cbz_site_receipt_recovery', TOOL)
        self.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tool)
        self.fixture = fixtures.PublicationTests(methodName='runTest')
        self.fixture.setUp()
        f = self.fixture
        first = f.prepare()
        self.prior = f.module.observe_publication(f.candidate, first, api=f.api, reader=f.reader,
            context=f.context, now='2026-10-13T15:01:00Z', attempts=1)
        failed_candidate = copy.deepcopy(f.candidate)
        failed_candidate['source']['commit'] = 'b' * 40
        failed_candidate['run']['id'] = 43
        failed_context = dict(f.context, commit='b' * 40, run_id=43)
        self.failed = f.module.prepare_publication(failed_candidate, api=f.api, context=failed_context,
            now='2026-10-13T15:02:00Z', attempts=1)
        self.platform = Platform()
        self.context = dict(f.context, commit='c' * 40, run_id=99)
        self.reader = fixtures.PublicReader(f.raw)
        self.before_calls = len(f.api.calls)

    def recover(self, **overrides):
        kwargs = {'api': self.fixture.api, 'platform': self.platform, 'reader': self.reader,
                  'context': self.context, 'expected_failed_head': self.failed['head'],
                  'expected_previous_head': self.prior['head'], 'now': '2026-10-13T15:03:00Z',
                  'attempts': 1}
        kwargs.update(overrides)
        return self.tool.recover_cancelled(**kwargs)

    def refusal_observation(self, **overrides):
        try:
            self.recover(**overrides)
        except ValueError as error:
            reason = str(error)
        else:
            reason = None
        writes = [method for method, _, _ in self.fixture.api.calls[self.before_calls:]
                  if method in ('POST', 'PATCH')]
        return reason, self.fixture.api.head, writes

    def test_recovers_only_the_previous_observed_candidate_with_failed_head_retained(self):
        result = self.recover()
        m, api = self.fixture.module, self.fixture.api
        self.assertEqual(result['receipt']['phase'], 'observed')
        self.assertEqual(m.candidate_of(result['receipt']), m.candidate_of(self.prior['receipt']))
        self.assertEqual(result['receipt']['recorded_by'], m.observer(self.context))
        intermediate = m.read_anchor_at(api, result['receipt']['previous_receipt_commit'])
        self.assertEqual(intermediate['receipt']['phase'], 'prepared')
        self.assertEqual(m.candidate_of(intermediate['receipt']), m.candidate_of(self.prior['receipt']))
        self.assertEqual(intermediate['receipt']['previous_receipt_commit'], self.failed['head'])
        self.assertEqual(len(self.reader.calls), 6)
        self.assertGreaterEqual(self.platform.deployment_reads, 2)
        self.assertTrue(all(body['force'] is False for method, _, body in api.calls[self.before_calls:]
                            if method == 'PATCH'))

    def test_wrong_expected_heads_refuse_before_any_write(self):
        expected = ('expected prepared receipt or parent changed', self.failed['head'], [])
        self.assertEqual(self.refusal_observation(expected_failed_head='f' * 40), expected)
        self.assertEqual(self.refusal_observation(expected_previous_head='e' * 40), expected)

    def test_running_or_successful_workflow_refuses(self):
        self.platform.run['status'] = 'in_progress'
        self.assertEqual(self.refusal_observation(), ('prepared source run is not a matching completed failure', self.failed['head'], []))
        self.platform.run.update(status='completed', conclusion='success')
        self.assertEqual(self.refusal_observation(), ('prepared source run is not a matching completed failure', self.failed['head'], []))

    def test_wrong_workflow_source_or_boolean_attempt_refuses(self):
        self.platform.run['head_sha'] = 'e' * 40
        self.assertEqual(self.refusal_observation(), ('prepared source run is not a matching completed failure', self.failed['head'], []))
        self.platform.run['head_sha'] = 'b' * 40
        self.platform.run['run_attempt'] = True
        self.assertEqual(self.refusal_observation(), ('prepared source run is not a matching completed failure', self.failed['head'], []))

    def test_nonterminal_or_unknown_deployment_refuses(self):
        self.platform.deployment['status'] = 'syncing_files'
        self.assertEqual(self.refusal_observation(), ('prepared Pages deployment is not confirmed canceled', self.failed['head'], []))
        self.platform.deployment['status'] = 'deployment_lost'
        self.assertEqual(self.refusal_observation(), ('prepared Pages deployment is not confirmed canceled', self.failed['head'], []))

    def test_current_main_mismatch_refuses(self):
        self.platform.main = 'e' * 40
        self.assertEqual(self.refusal_observation(), ('Main advanced from recovery source', self.failed['head'], []))

    def test_public_mismatch_refuses_without_writing(self):
        self.reader = fixtures.PublicReader(dict(self.fixture.raw, **{'current.json': b'wrong'}))
        self.assertEqual(self.refusal_observation(), ('public readback failed; prepared publication recovery needed, anchor retained', self.failed['head'], []))
        self.assertTrue(self.reader.calls)

    def test_change_to_main_before_write_refuses(self):
        self.platform.change_main_on = 2
        self.assertEqual(self.refusal_observation(), ('Main advanced from recovery source', self.failed['head'], []))

    def test_change_to_deployment_before_write_refuses(self):
        self.platform.change_deployment_on = 2
        self.assertEqual(self.refusal_observation(), ('prepared Pages deployment is not confirmed canceled', self.failed['head'], []))

    def test_lost_prepared_snapshot_refuses(self):
        m, api = self.fixture.module, self.fixture.api
        candidate = copy.deepcopy(m.candidate_of(self.failed['receipt']))
        candidate['snapshot_ids'] = sorted(candidate['snapshot_ids'] + ['e' * 64])
        receipt = dict(self.failed['receipt'], **candidate)
        api.head = self.prior['head']
        self.failed = m.append_receipt(api, receipt, self.prior['head'])
        self.before_calls = len(api.calls)
        self.assertEqual(self.refusal_observation(), ('recovery would lose a proposed historical snapshot', self.failed['head'], []))

    def test_second_public_mismatch_leaves_only_prepared_prior_candidate(self):
        self.reader = SwitchReader(self.fixture.raw)
        with self.assertRaisesRegex(ValueError, 'public readback failed'):
            self.recover()
        m, api = self.fixture.module, self.fixture.api
        current = m.read_anchor(api)
        self.assertEqual(current['receipt']['phase'], 'prepared')
        self.assertEqual(m.candidate_of(current['receipt']), m.candidate_of(self.prior['receipt']))
        self.assertEqual(current['receipt']['previous_receipt_commit'], self.failed['head'])

    def test_malformed_actual_context_refuses(self):
        self.context['ref'] = 'refs/heads/other'
        self.assertEqual(self.refusal_observation(), ('recovery requires the main repository and branch', self.failed['head'], []))

    def test_prior_observed_candidate_must_match_its_prepared_parent(self):
        m, api = self.fixture.module, self.fixture.api
        first_head = self.prior['receipt']['previous_receipt_commit']
        api.head = first_head
        altered = copy.deepcopy(self.prior['receipt'])
        altered['artifact']['sha256'] = 'e' * 64
        self.prior = m.append_receipt(api, altered, first_head)
        failed = dict(self.failed['receipt'], previous_receipt_commit=self.prior['head'])
        self.failed = m.append_receipt(api, failed, self.prior['head'])
        self.before_calls = len(api.calls)
        self.assertEqual(self.refusal_observation(), ('previous observation lacks its matching prepared parent', self.failed['head'], []))

    def test_non_fast_forward_receipt_race_is_not_forced(self):
        self.fixture.api.race = True
        with self.assertRaisesRegex(ValueError, 'non-fast-forward branch race'):
            self.recover()
        self.assertEqual(self.fixture.api.head, self.failed['head'])
        patches = [body for method, _, body in self.fixture.api.calls[self.before_calls:] if method == 'PATCH']
        self.assertEqual(len(patches), 1)
        self.assertIs(patches[0]['force'], False)

    def test_read_only_platform_routes_reject_unrelated_paths(self):
        platform = self.tool.PlatformRead(token='synthetic-token', opener=Opener(b'{}'))
        for path in ('/git/refs/heads/main', '/actions/runs/0', '/actions/runs/43?x=1',
                     '/pages/deployments/not-a-sha', '/user', 'https://example.invalid/'):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, 'non-recovery platform route refused'):
                    platform.get(path)

    def test_success_records_actual_recovery_evidence_in_order(self):
        self.assertIn('on_event', inspect.signature(self.tool.recover_cancelled).parameters)
        events = []
        result = self.recover(on_event=events.append)
        self.assertEqual([event['stage'] for event in events],
                         ['ready', 'prepared', 'verified_before_observed', 'observed'])
        self.assertEqual(events[0]['failed_head'], self.failed['head'])
        self.assertEqual(events[0]['previous_head'], self.prior['head'])
        self.assertEqual(events[0]['platform']['deployment_status'], 'deployment_cancelled')
        self.assertEqual(events[0]['public']['current.json']['sha256'],
                         self.prior['receipt']['payload']['sha256'])
        self.assertEqual(events[1]['head'], result['receipt']['previous_receipt_commit'])
        self.assertEqual(events[-1]['head'], result['head'])

    def test_interrupted_recovery_records_prepared_but_not_observed(self):
        self.assertIn('on_event', inspect.signature(self.tool.recover_cancelled).parameters)
        events = []
        with self.assertRaisesRegex(ValueError, 'public readback failed'):
            self.recover(reader=SwitchReader(self.fixture.raw), on_event=events.append)
        self.assertEqual([event['stage'] for event in events], ['ready', 'prepared'])
        self.assertEqual(events[-1]['head'], self.fixture.api.head)

    def test_platform_read_uses_fixed_get_route_and_checks_main_ref(self):
        body = fixtures.encoded({'ref': 'refs/heads/main', 'object': {'type': 'commit', 'sha': 'c' * 40}})
        opener = Opener(body)
        platform = self.tool.PlatformRead(token='synthetic-token', opener=opener)
        self.assertEqual(platform.main_head(), 'c' * 40)
        self.assertEqual(len(opener.requests), 1)
        request, timeout = opener.requests[0]
        self.assertEqual(request.get_method(), 'GET')
        self.assertEqual(request.full_url, self.fixture.module.API_ORIGIN + '/git/ref/heads/main')
        self.assertEqual(timeout, 20)

    def test_platform_read_rejects_redirect_status_and_malformed_response(self):
        for opener in (Opener(b'{}', url='https://example.invalid/'), Opener(b'{}', status=201)):
            with self.subTest(opener=opener):
                with self.assertRaisesRegex(ValueError, 'platform response identity mismatch'):
                    self.tool.PlatformRead(token='synthetic-token', opener=opener).main_head()
        for body in (b'[]', b'{"x": 1, "x": 2}', b' ' * (self.fixture.module.MAX_RECEIPT_BYTES + 1)):
            with self.subTest(bytes=len(body)):
                with self.assertRaises(ValueError):
                    self.tool.PlatformRead(token='synthetic-token', opener=Opener(body)).main_head()

    def test_platform_read_requires_token_and_strict_run_identifier(self):
        with self.assertRaisesRegex(ValueError, 'token required'):
            self.tool.PlatformRead(token='')
        platform = self.tool.PlatformRead(token='synthetic-token', opener=Opener(b'{}'))
        with self.assertRaises(ValueError):
            platform.workflow_run(True)
        self.assertEqual(platform.opener.requests, [])

    def test_cli_refuses_wrong_event_before_runtime_access(self):
        argv = ['recovery', '--expected-failed-head', 'a' * 40,
                '--expected-previous-head', 'b' * 40]
        with mock.patch.object(sys, 'argv', argv), mock.patch.dict('os.environ', {
                'GITHUB_EVENT_NAME': 'push', 'GITHUB_REPOSITORY': 'd6g8k5htny-coder/main',
                'GITHUB_REF': 'refs/heads/main', 'GITHUB_SHA': 'c' * 40,
                'GITHUB_RUN_ID': '99', 'GITHUB_RUN_ATTEMPT': '1',
                'GITHUB_WORKFLOW_REF': 'd6g8k5htny-coder/main/.github/workflows/cbz-site-recovery.yml@refs/heads/main'}, clear=True), \
                mock.patch.object(self.tool.publication, 'execution_context', return_value=self.context) as context, \
                mock.patch.object(self.tool, 'recover_cancelled') as recover, \
                mock.patch('sys.stderr', new_callable=io.StringIO) as stderr:
            recover.return_value = {'head': 'd' * 40, 'receipt': {'phase': 'observed',
                'source': {'commit': 'a' * 40}, 'recorded_by': {'run_id': 99}}}
            with self.assertRaises(SystemExit) as exited:
                self.tool.main()
        self.assertEqual(exited.exception.code, 1)
        self.assertIn('explicit main-branch workflow dispatch', stderr.getvalue())
        context.assert_not_called()
        recover.assert_not_called()


if __name__ == '__main__':
    unittest.main()
