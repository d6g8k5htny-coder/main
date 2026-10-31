"""PR admission contract: arbitrary future topology inputs must reach the crawl.

This reuses the existing restricted event-contract reader. It is a source
subscription guard, not a general YAML parser or proof of hosted execution.
"""
import importlib.util
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/cbz-federation-drift.yml'
SPEC = importlib.util.spec_from_file_location(
    'navigation_event_contract', ROOT / 'tests/test_navigation_workflow.py')
EVENTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVENTS)


def require_unfiltered_pr(text):
    """Read only the actual PR subscription, rejecting unsupported syntax."""
    # YAML full-line comments do not close the on/event mappings. Strip them
    # before either boundary reader can mistake one for a new top-level key.
    lines = [line for line in text.splitlines()
             if not line.lstrip().startswith('#')]
    start = lines.index("'on':") + 1
    end = next((i for i in range(start, len(lines))
                if lines[i] and not lines[i].startswith(' ')), len(lines))
    starts = [i for i in range(start, end)
              if re.fullmatch(r'  pull_request:\s*', lines[i])]
    if len(starts) != 1:
        raise AssertionError('exactly one bare pull_request subscription required')
    begin = starts[0]
    stop = next((i for i in range(begin + 1, end)
                 if re.match(r'  \S', lines[i])), end)
    event = EVENTS.parse_event_contract(
        "'on':\n" + '\n'.join(lines[begin:stop]) + '\n')['pull_request']
    if event:
        raise AssertionError('PR eligibility must not depend on paths, bases or actions')


class DriftPullRequestAdmissionTests(unittest.TestCase):
    def test_future_topology_inputs_cannot_be_excluded_before_discovery(self):
        # An unconditional subscription is the admission contract: today's
        # registry filenames cannot enumerate tomorrow's federation inputs.
        require_unfiltered_pr(WORKFLOW.read_text(encoding='utf-8'))

    def test_guard_rejects_path_base_and_activity_restrictions(self):
        restrictions = (
            "paths: ['formal/**']",
            "paths-ignore: ['**/*.json']",
            'branches: [main]',
            'types: [opened]',
        )
        for restriction in restrictions:
            with self.subTest(restriction=restriction):
                with self.assertRaises(AssertionError):
                    require_unfiltered_pr(
                        "'on':\n  pull_request:\n    " + restriction + '\n')

    def test_comments_cannot_hide_a_restriction(self):
        for restriction in ("paths: ['formal/**']", "paths-ignore: ['**/*.json']",
                            'branches: [main]', 'types: [opened]'):
            with self.subTest(restriction=restriction):
                with self.assertRaises(AssertionError):
                    require_unfiltered_pr(
                        "'on':\n  pull_request:\n# a YAML comment is not a mapping boundary\n    "
                        + restriction + '\n')

    def test_comments_preserve_an_unfiltered_subscription(self):
        require_unfiltered_pr(
            "'on':\n# event documentation\n  pull_request:\n# still inside on\n"
            "  push:\n    branches: [main]\n")


if __name__ == '__main__':
    unittest.main()
