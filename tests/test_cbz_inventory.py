"""Tests for the CBZ-0 inventory renderer: unavailable data is shown, never dropped."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cbz_inventory_under_test', ROOT / 'tools' / 'cbz_inventory.py')
inv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inv)


def repo(**over):
    base = {'name': 'r', 'clone': '/x', 'default_branch': 'main', 'default_head': 'a' * 40,
            'remote_branches': [{'name': 'origin/main', 'last_commit': 't', 'merged_into_default': True},
                                {'name': 'origin/feat', 'last_commit': 't', 'merged_into_default': False}],
            'tags': ['t1'], 'stashes': 0, 'worktrees': ['/x'], 'working_tree_dirty_entries': 0,
            'status_surfaces': {s: False for s in inv.SURFACES}, 'file_count': 3, 'lean_file_count': 1,
            'workflow_count': 0,
            'layer_a_default_tree': {'files_scanned': 1, 'hits': [{'path': 'A.lean', 'line': 2, 'token': 'sorry'}],
                                     'errors': []},
            'default_head_checks': [{'name': 'ci', 'status': 'completed', 'conclusion': 'failure'}],
            'open_prs': [{'number': 7, 'title': 'T|x', 'draft': True, 'head_ref': 'h', 'head_sha': 'b' * 40,
                          'base_ref': 'main', 'created_at': '2026-10-01T00:00:00Z', 'updated_at': '2026-10-02T00:00:00Z',
                          'changed_lean_files': 1,
                          'layer_a_changed_lean': {'files_scanned': 1, 'hits': [{'path': 'B.lean', 'line': 3, 'token': 'axiom'}],
                                                   'errors': [], 'deleted': []}}]}
    base.update(over)
    return base


def doc(repos, inaccessible=()):
    return {'generated_at': 'now', 'repos': repos, 'gates': inv.gate_matrix(),
            'inaccessible': [{'name': n, 'reason': r} for n, r in inaccessible]}


class RenderFailsClosed(unittest.TestCase):
    def test_hits_failures_and_counts_are_shown(self):
        md = inv.render_md(doc([repo()]))
        self.assertIn('1 hits, 0 errors, 1 Lean files', md)
        self.assertIn('2 (1)', md)                      # two remote branches, one unmerged
        self.assertIn('ci=failure', md)
        self.assertIn('B.lean:3 axiom', md)
        self.assertIn('T/x', md)                        # table pipes are escaped

    def test_unavailable_repo_checks_and_prs_are_reported(self):
        md = inv.render_md(doc([{'name': 'gone', 'unavailable': 'clone missing'},
                                repo(default_head_checks={'unavailable': 'HTTP 403'},
                                     open_prs={'unavailable': 'HTTP 404'})],
                               inaccessible=[('secret', 'no access')]))
        self.assertIn('unavailable: clone missing', md)
        self.assertIn('unavailable: HTTP 403', md)
        self.assertIn('unavailable: HTTP 404', md)
        self.assertIn('secret: no access', md)

    def test_gate_matrix_covers_g0_to_g11(self):
        gates = [g['gate'] for g in inv.gate_matrix()]
        self.assertEqual(gates, [f'G{i}' for i in range(12)])
        for g in inv.gate_matrix():
            self.assertTrue(g.get('tool') or g.get('gap'), g['gate'])

    def test_scope_note_is_printed_first(self):
        d = doc([repo()]); d['scope_note'] = 'PARTIAL: 6 of 12 repositories'
        md = inv.render_md(d)
        self.assertLess(md.index('PARTIAL: 6 of 12'), md.index('## Repositories'))

    def test_gate_with_tool_and_gap_shows_both(self):
        md = inv.render_md(doc([repo()]))
        self.assertIn('verify_pins.py', md)
        self.assertIn('gap: no current coherent federation cut', md)


if __name__ == '__main__':
    unittest.main()
