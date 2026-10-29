"""Persistent-reading semantics and exact old-source/history preservation."""
import ast
import copy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import current_site_build as build
import current_site_history as history
from tests.test_current_site_build import fixture


class Paragraphs(HTMLParser):
    def __init__(self, text):
        super().__init__(); self.stack = []; self.statuses = []; self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'p' and attrs.get('class') == 'current-state':
            self.statuses.append((attrs, self.stack[-1] if self.stack else None))
        if tag not in {'meta', 'link', 'input', 'img', 'br', 'hr'}:
            self.stack.append((tag, attrs))

    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]; break


class PersistentStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()
        cls.old = json.loads((ROOT/'tests/fixtures/persistent_status_legacy.json').read_text())
        cls.meta = cls.old['meta']
        raw = json.dumps(cls.data, sort_keys=True).encode()
        assert hashlib.sha256(raw).hexdigest() == cls.old['fixture_sha256']

    def assert_readable(self, text):
        paragraphs = Paragraphs(text).statuses
        self.assertEqual(len(paragraphs), 1)
        attrs, parent = paragraphs[0]
        self.assertNotIn('role', attrs, 'the paragraph must retain ordinary text semantics')
        self.assertEqual(parent, ('div', {'role': 'status'}))

    def test_current_panel_retains_paragraph_inside_status_region(self):
        for compact in (False, True):
            with self.subTest(compact=compact):
                self.assert_readable(build.current_section(self.data, self.meta, compact=compact, readable_status=True))

    def test_all_new_current_static_routes_have_readable_status(self):
        for name, raw in history.render_live_current_v2(self.data, self.meta).items():
            with self.subTest(path=name): self.assert_readable(raw.decode())

    def test_original_archived_section_and_frozen_renderer_sources_do_not_change(self):
        self.assertEqual(hashlib.sha256(build.current_section(self.data, self.meta, archived=True).encode()).hexdigest(),
                         self.old['archived_section_sha256'])
        self.assertEqual(hashlib.sha256(build.current_section(self.data, self.meta, archived=True, readable_status=True).encode()).hexdigest(),
                         self.old['archived_section_sha256'])
        source = (ROOT/'tools/current_site_history.py').read_text()
        funcs = {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef)}
        for name, digest in self.old['frozen_renderer_sources'].items():
            with self.subTest(function=name):
                self.assertEqual(hashlib.sha256(funcs[name].encode()).hexdigest(), digest)

    def test_previous_full_live_descriptors_restore_byte_exactly(self):
        expected = self.old['legacy_live_descriptors']
        restored = history._restore_v2_live_pages(self.data, self.meta, expected)
        self.assertEqual([history.descriptor(k, v) for k, v in sorted(restored.items())], expected)

    def test_both_explicit_live_presentations_restore_and_keep_same_source(self):
        for readable in (True, False):
            pages = history.render_live_current_v2(self.data, self.meta, readable_status=readable)
            expected = [history.descriptor(k, v) for k, v in sorted(pages.items())]
            self.assertEqual(history._restore_v2_live_pages(self.data, self.meta, expected), pages)
            if not readable: self.assertEqual(expected, self.old['legacy_live_descriptors'])

    def test_complete_descriptors_do_not_admit_changed_source_text_or_role(self):
        pages = history.render_live_current_v2(self.data, self.meta)
        name = 'nodes/index.html'
        changes = [(b'Pinned source observation:', b'Accepted scientific observation:'),
                   (self.data['repositories'][0]['commit'].encode(), b'c'*40),
                   (b'role="status"', b'role="alert"')]
        for before, after in changes:
            tampered = dict(pages)
            self.assertIn(before, tampered[name])
            tampered[name] = tampered[name].replace(before, after, 1)
            descriptors = [history.descriptor(k, v) for k, v in sorted(tampered.items())]
            with self.subTest(change=before), self.assertRaisesRegex(ValueError, 'exact observation'):
                history._restore_v2_live_pages(self.data, self.meta, descriptors)

    def test_legacy50_restoration_qualifies_acquisitions_once(self):
        expected = self.old['legacy_live_descriptors']
        with patch.object(history, 'project_current_acquisitions',
                          wraps=history.project_current_acquisitions) as project:
            restored = history._restore_v2_live_pages(self.data, self.meta, expected)
        self.assertEqual([history.descriptor(k, v) for k, v in sorted(restored.items())], expected)
        self.assertEqual(len(restored), 50)
        self.assertEqual(project.call_count, 1, 'presentation retries share the same acquisition input')

    def test_all_24_presentations_restore_exactly_with_task_table_sensitive_input(self):
        data = copy.deepcopy(self.data)
        data['graph']['data']['nodes']['math.lifetime-remainder'] = data['graph']['data']['nodes'].pop('node-0')
        for task_records in (history.RESUME_TASK_RECORDS, history._RESUME_TASK_RECORDS_V1):
            for annotations, style_first, resume in history.LIVE_PRESENTATIONS:
                for readable in (True, False):
                    pages = history.render_live_current_v2(
                        data, self.meta, annotations=annotations, style_first=style_first,
                        resume=resume, task_records=task_records, readable_status=readable)
                    expected = [history.descriptor(k, v) for k, v in sorted(pages.items())]
                    with self.subTest(current_tasks=task_records is history.RESUME_TASK_RECORDS,
                                      annotations=annotations, style_first=style_first,
                                      resume=resume, readable=readable):
                        self.assertEqual(history._restore_v2_live_pages(data, self.meta, expected), pages)

    def test_restore_still_requires_complete_ordered_descriptor_membership(self):
        pages = history.render_live_current_v2(self.data, self.meta)
        expected = [history.descriptor(k, v) for k, v in sorted(pages.items())]
        wrong_path = copy.deepcopy(expected); wrong_path[0]['path'] = 'nodes/other.html'
        wrong_size = copy.deepcopy(expected); wrong_size[0]['bytes'] += 1
        mixed = copy.deepcopy(expected); mixed[0] = copy.deepcopy(self.old['legacy_live_descriptors'][0])
        controls = {'missing': expected[:-1], 'duplicate': expected + [expected[0]],
                    'reordered': list(reversed(expected)), 'wrong_path': wrong_path,
                    'wrong_size': wrong_size, 'mixed': mixed}
        for label, descriptors in controls.items():
            with self.subTest(control=label), self.assertRaisesRegex(ValueError, 'exact observation'):
                history._restore_v2_live_pages(self.data, self.meta, descriptors)

    def test_each_restoration_qualifies_changed_input_without_mutation(self):
        first = copy.deepcopy(self.data)
        second = copy.deepcopy(self.data)
        second['sources'].append(copy.deepcopy(second['sources'][0]))
        self.assertNotEqual(history.project_current_acquisitions(first)['reasons'],
                            history.project_current_acquisitions(second)['reasons'])
        before = copy.deepcopy(second)
        cases = []
        for data in (first, second):
            pages = history.render_live_current_v2(data, self.meta, readable_status=False)
            expected = [history.descriptor(k, v) for k, v in sorted(pages.items())]
            cases.append((data, pages, expected))
        with patch.object(history, 'project_current_acquisitions',
                          wraps=history.project_current_acquisitions) as project:
            for data, pages, expected in cases:
                self.assertEqual(history._restore_v2_live_pages(data, self.meta, expected), pages)
        self.assertEqual(project.call_count, 2, 'each restoration must qualify its own input')
        self.assertEqual(second, before)


if __name__ == '__main__': unittest.main()
