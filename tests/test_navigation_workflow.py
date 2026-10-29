"""Restricted contract for navigation workflow event eligibility."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github' / 'workflows' / 'navigation.yml'
DEFAULT_PR_ACTIONS = {'opened', 'synchronize', 'reopened'}
PRESERVED_JOB_SHA256 = (
    '01d33dfa8150cc9d6d0c90ea65880db525a5bd68c37692cd1bd02332f18c051a'
)


def _parse_flow_list(value: str) -> list[str]:
    if not value.startswith('[') or not value.endswith(']'):
        raise AssertionError('expected a single-line flow list')
    inner = value[1:-1].strip()
    if not inner:
        return []
    items = []
    for raw in inner.split(','):
        item = raw.strip()
        if len(item) >= 2 and item[0] == item[-1] and item[0] in "'\"":
            item = item[1:-1]
        if not item or any(character in item for character in '[]{}'):
            raise AssertionError('unsupported flow-list item')
        items.append(item)
    return items


def parse_event_contract(workflow: str) -> dict[str, dict[str, list[str]]]:
    lines = workflow.splitlines()
    try:
        start = lines.index("'on':") + 1
    except ValueError as exc:
        raise AssertionError('missing quoted top-level on key') from exc
    events: dict[str, dict[str, list[str]]] = {}
    current_event = None
    for line in lines[start:]:
        if line and not line.startswith(' '):
            break
        if not line:
            continue
        event_match = re.fullmatch(r'  ([a-z_][a-z0-9_-]*):(.*)', line)
        if event_match:
            event, remainder = event_match.groups()
            if remainder.strip():
                raise AssertionError('event mappings may not use scalar values')
            if event in events:
                raise AssertionError(f'duplicate event key: {event}')
            events[event] = {}
            current_event = event
            continue
        option_match = re.fullmatch(
            r'    ([a-z_][a-z0-9_-]*): (\[[^\n]*\])', line
        )
        if option_match and current_event is not None:
            option, value = option_match.groups()
            if option in events[current_event]:
                raise AssertionError(
                    f'duplicate {current_event} option: {option}'
                )
            events[current_event][option] = _parse_flow_list(value)
            continue
        raise AssertionError(f'unsupported event grammar: {line!r}')
    return events


def assert_unfiltered_event_contract(
    workflow: str,
) -> dict[str, dict[str, list[str]]]:
    events = parse_event_contract(workflow)
    if set(events) != {'push', 'pull_request', 'workflow_dispatch'}:
        raise AssertionError(f'unexpected event keys: {sorted(events)}')
    if events['push'] != {'branches': ['main']}:
        raise AssertionError(f"unexpected push contract: {events['push']}")
    if events['pull_request']:
        raise AssertionError(
            'pull_request must retain default activities and all bases: '
            f"{events['pull_request']}"
        )
    if events['workflow_dispatch']:
        raise AssertionError('workflow_dispatch must remain a bare event')
    return events


def automatic_event_selected(
    workflow: str,
    *,
    event: str,
    ref: str = '',
    action: str = '',
    changed_paths: tuple[str, ...] = (),
) -> bool:
    assert_unfiltered_event_contract(workflow)
    del changed_paths
    if event == 'push':
        return ref == 'refs/heads/main'
    if event == 'pull_request':
        return action in DEFAULT_PR_ACTIONS
    return False


class NavigationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = WORKFLOW.read_text(encoding='utf-8')

    def test_unfiltered_event_contract(self):
        self.assertEqual(assert_unfiltered_event_contract(self.workflow), {
            'push': {'branches': ['main']},
            'pull_request': {},
            'workflow_dispatch': {},
        })

    def test_all_repository_paths_are_positive_on_eligible_events(self):
        paths = (
            'formal/manifest.json',
            'tools/atlas_formal_evidence_build.py',
            'tests/fixtures/atlas_formal_build_3b39/manifest.json',
            'architecture/graph-artifact-bindings.v2.json',
            'records/node_status/example.json',
            'experiments/st_contract_pilot/run.py',
            'governance/OWNER_MANDATE_20261015.md',
            '.github/workflows/formal-verification.yml',
            'LICENSE',
            'tests/nested/test_unrelated.py',
        )
        for path in paths:
            with self.subTest(path=path, event='push'):
                self.assertTrue(automatic_event_selected(
                    self.workflow,
                    event='push',
                    ref='refs/heads/main',
                    changed_paths=(path,),
                ))
            for action in sorted(DEFAULT_PR_ACTIONS):
                with self.subTest(path=path, event='pull_request', action=action):
                    self.assertTrue(automatic_event_selected(
                        self.workflow,
                        event='pull_request',
                        action=action,
                        changed_paths=(path,),
                    ))

    def test_empty_diff_has_no_path_based_rejection(self):
        self.assertTrue(automatic_event_selected(
            self.workflow, event='push', ref='refs/heads/main'))
        self.assertTrue(automatic_event_selected(
            self.workflow, event='pull_request', action='opened'))

    def test_event_negatives_remain_negative(self):
        for ref in ('refs/heads/feature', 'refs/tags/v1.0.0'):
            with self.subTest(ref=ref):
                self.assertFalse(automatic_event_selected(
                    self.workflow,
                    event='push',
                    ref=ref,
                    changed_paths=('docs/README.md',),
                ))
        for action in ('edited', 'labeled', 'closed', 'ready_for_review'):
            with self.subTest(action=action):
                self.assertFalse(automatic_event_selected(
                    self.workflow,
                    event='pull_request',
                    action=action,
                    changed_paths=('docs/README.md',),
                ))

    def test_manual_subscription_remains_separate(self):
        events = assert_unfiltered_event_contract(self.workflow)
        self.assertEqual(events['workflow_dispatch'], {})

    def test_filter_and_event_mutations_fail_closed(self):
        mutations = {
            'push paths': self.workflow.replace(
                '    branches: [main]\n',
                "    branches: [main]\n    paths: ['README.md']\n",
                1,
            ),
            'pull-request paths-ignore': self.workflow.replace(
                '  pull_request:\n',
                "  pull_request:\n    paths-ignore: ['docs/**']\n",
                1,
            ),
            'wrong push branch': self.workflow.replace(
                '    branches: [main]', '    branches: [develop]', 1),
            'explicit PR activities': self.workflow.replace(
                '  pull_request:\n',
                '  pull_request:\n    types: [opened, synchronize]\n',
                1,
            ),
            'PR base restriction': self.workflow.replace(
                '  pull_request:\n',
                '  pull_request:\n    branches: [main]\n',
                1,
            ),
            'missing manual event': self.workflow.replace(
                '  workflow_dispatch:\n', '', 1),
            'duplicate PR event': self.workflow.replace(
                '  workflow_dispatch:\n',
                '  pull_request:\n  workflow_dispatch:\n',
                1,
            ),
            'malformed nested event value': self.workflow.replace(
                '  pull_request:\n',
                '  pull_request:\n    types:\n      - opened\n',
                1,
            ),
        }
        for name, mutated in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(AssertionError):
                    assert_unfiltered_event_contract(mutated)

    def test_entire_job_and_permissions_are_byte_preserved(self):
        suffix = self.workflow[self.workflow.index('permissions:'):]
        self.assertEqual(
            hashlib.sha256(suffix.encode('utf-8')).hexdigest(),
            PRESERVED_JOB_SHA256,
        )

    def test_normal_discovery_command_remains_exactly_once(self):
        command = (
            "python -B -S -m unittest discover -s tests -p 'test_*.py' -v "
            "> navigation-evidence/tests-normal.log 2>&1"
        )
        self.assertEqual(self.workflow.count(command), 1)


if __name__ == '__main__':
    unittest.main()
