"""Narrow append-only recovery of a canceled prepared Pages publication.

Only a canceled prepared candidate whose previous observed candidate is still
public and retains every proposed snapshot can be superseded. Both new
receipts describe that previous candidate and identify the actual recovery
run as observer; the failed prepared receipt remains in Git ancestry.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import urllib.request

import current_site_publication as publication


class PlatformRead:
    """Three bounded read-only GitHub routes; all writes use the existing guard."""
    def __init__(self, token=None, opener=None):
        self.token = os.environ.get('GITHUB_TOKEN') if token is None else token
        if not isinstance(self.token, str) or not self.token:
            raise ValueError('GitHub token required for platform status')
        self.opener = opener if opener is not None else urllib.request.build_opener(publication.NoRedirect())

    def get(self, path):
        if not (path == '/git/ref/heads/main'
                or re.fullmatch(r'/actions/runs/[1-9][0-9]{0,18}', path)
                or re.fullmatch(r'/pages/deployments/[0-9a-f]{40}', path)):
            raise ValueError('non-recovery platform route refused')
        url = publication.API_ORIGIN + path
        request = urllib.request.Request(url, headers={
            'Accept': 'application/vnd.github+json', 'User-Agent': 'cbz-site-receipt-recovery',
            'X-GitHub-Api-Version': '2022-11-28', 'Authorization': 'Bearer ' + self.token})
        with self.opener.open(request, timeout=20) as response:
            if response.geturl() != url or response.status != 200:
                raise ValueError('platform response identity mismatch')
            raw = response.read(publication.MAX_RECEIPT_BYTES + 1)
        if len(raw) > publication.MAX_RECEIPT_BYTES:
            raise ValueError('platform response too large')
        obj = publication.parse_json(raw)
        if not isinstance(obj, dict):
            raise ValueError('platform response must be an object')
        return obj

    def main_head(self):
        ref = self.get('/git/ref/heads/main')
        if (ref.get('ref') != 'refs/heads/main' or not isinstance(ref.get('object'), dict)
                or ref['object'].get('type') != 'commit'):
            raise ValueError('invalid Main ref')
        return publication.hex_value(ref['object'].get('sha'), publication.HEX40, 'Main head')

    def workflow_run(self, run_id):
        publication.integer(run_id, 1, 2**63 - 1, 'failed run id')
        return self.get('/actions/runs/' + str(run_id))

    def pages_deployment(self, deployment_id):
        publication.hex_value(deployment_id, publication.HEX40, 'deployment id')
        return self.get('/pages/deployments/' + deployment_id)


def check_context(context):
    publication.closed(context, ('repository', 'ref', 'commit', 'run_id', 'run_attempt'), 'recovery context')
    publication.hex_value(context['commit'], publication.HEX40, 'recovery source')
    publication.integer(context['run_id'], 1, 2**63 - 1, 'recovery run')
    publication.integer(context['run_attempt'], 1, 100_000, 'recovery attempt')
    if context['repository'] != publication.REPOSITORY or context['ref'] != 'refs/heads/main':
        raise ValueError('recovery requires the main repository and branch')


def check_platform(platform, candidate, context):
    main_head = platform.main_head()
    if main_head != context['commit']:
        raise ValueError('Main advanced from recovery source')
    run = platform.workflow_run(candidate['run']['id'])
    if (not isinstance(run, dict) or type(run.get('id')) is not int or run['id'] != candidate['run']['id']
            or run.get('status') != 'completed' or run.get('conclusion') not in ('failure', 'cancelled')
            or type(run.get('run_attempt')) is not int or run['run_attempt'] < candidate['run']['attempt']
            or run.get('head_sha') != candidate['source']['commit'] or run.get('head_branch') != 'main'
            or run.get('path') != publication.WORKFLOW or run.get('event') != 'push'
            or not isinstance(run.get('repository'), dict) or run['repository'].get('full_name') != publication.REPOSITORY
            or not isinstance(run.get('head_repository'), dict) or run['head_repository'].get('full_name') != publication.REPOSITORY):
        raise ValueError('prepared source run is not a matching completed failure')
    deployment = platform.pages_deployment(candidate['source']['commit'])
    if not isinstance(deployment, dict) or deployment.get('status') != 'deployment_cancelled':
        raise ValueError('prepared Pages deployment is not confirmed canceled')
    return {'main_head': main_head, 'run_id': run['id'], 'run_attempt': run['run_attempt'],
            'run_status': run['status'], 'run_conclusion': run['conclusion'],
            'deployment_id': candidate['source']['commit'], 'deployment_status': deployment['status']}


def public_evidence(raw):
    return {path: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            for path, data in sorted(raw.items())}


def recover_cancelled(*, expected_failed_head, expected_previous_head, context,
                      api=None, platform=None, reader=None, now=None, attempts=2, on_event=None):
    check_context(context)
    publication.hex_value(expected_failed_head, publication.HEX40, 'expected failed head')
    publication.hex_value(expected_previous_head, publication.HEX40, 'expected previous head')
    api = api if api is not None else publication.GitHubAPI()
    platform = platform if platform is not None else PlatformRead()
    emit = on_event if on_event is not None else lambda event: None
    failed = publication.read_anchor(api)
    if (failed['head'] != expected_failed_head or failed['receipt']['phase'] != 'prepared'
            or failed['receipt']['previous_receipt_commit'] != expected_previous_head):
        raise ValueError('expected prepared receipt or parent changed')
    previous = publication.read_anchor_at(api, expected_previous_head)
    if previous['receipt']['phase'] != 'observed':
        raise ValueError('previous receipt is not observed')
    old_prepared_head = previous['receipt']['previous_receipt_commit']
    old_prepared = publication.read_anchor_at(api, old_prepared_head)
    if (old_prepared['receipt']['phase'] != 'prepared'
            or publication.candidate_of(old_prepared['receipt']) != publication.candidate_of(previous['receipt'])):
        raise ValueError('previous observation lacks its matching prepared parent')
    failed_candidate = publication.candidate_of(failed['receipt'])
    prior_candidate = publication.candidate_of(previous['receipt'])
    if failed_candidate == prior_candidate:
        raise ValueError('failed and previous candidates are identical')
    if not set(failed_candidate['snapshot_ids']) <= set(prior_candidate['snapshot_ids']):
        raise ValueError('recovery would lose a proposed historical snapshot')

    check_platform(platform, failed_candidate, context)
    first_public = publication.public_readback(previous['receipt'], reader=reader, attempts=attempts)
    if publication.read_anchor(api) != failed:
        raise ValueError('receipt changed before recovery')
    platform_before_write = check_platform(platform, failed_candidate, context)
    emit({'stage': 'ready', 'failed_head': failed['head'], 'previous_head': previous['head'],
          'platform': platform_before_write, 'public': public_evidence(first_public)})
    stamp = now or datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    prepared = {**prior_candidate, 'phase': 'prepared', 'recorded_at': stamp,
                'recorded_by': publication.observer(context), 'previous_receipt_commit': failed['head']}
    intermediate = publication.append_receipt(api, prepared, failed['head'])
    emit({'stage': 'prepared', 'head': intermediate['head'], 'parent': failed['head']})

    second_public = publication.public_readback(previous['receipt'], reader=reader, attempts=attempts)
    if publication.read_anchor(api) != intermediate:
        raise ValueError('receipt changed during recovery')
    platform_before_observed = check_platform(platform, failed_candidate, context)
    emit({'stage': 'verified_before_observed', 'prepared_head': intermediate['head'],
          'platform': platform_before_observed, 'public': public_evidence(second_public)})
    stamp = now or datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    observed = {**prior_candidate, 'phase': 'observed', 'recorded_at': stamp,
                'recorded_by': publication.observer(context), 'previous_receipt_commit': intermediate['head']}
    result = publication.append_receipt(api, observed, intermediate['head'])
    emit({'stage': 'observed', 'head': result['head'], 'parent': intermediate['head'],
          'recorded_by': result['receipt']['recorded_by']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-failed-head', required=True)
    parser.add_argument('--expected-previous-head', required=True)
    args = parser.parse_args()
    try:
        if (os.environ.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'
                or os.environ.get('GITHUB_WORKFLOW_REF') !=
                publication.REPOSITORY + '/.github/workflows/cbz-site-recovery.yml@refs/heads/main'):
            raise ValueError('recovery requires its explicit main-branch workflow dispatch')
        context = publication.execution_context()
        def emit(event):
            print(json.dumps(event, sort_keys=True), flush=True)
        result = recover_cancelled(expected_failed_head=args.expected_failed_head,
                                   expected_previous_head=args.expected_previous_head, context=context, on_event=emit)
        print(json.dumps({'phase': result['receipt']['phase'], 'receipt_commit': result['head'],
                          'candidate_source': result['receipt']['source']['commit'],
                          'recorded_by': result['receipt']['recorded_by'], 'scientific_effect': 'NONE'}))
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(1, 'site receipt recovery refused: ' + str(error) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
