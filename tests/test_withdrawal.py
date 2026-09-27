"""Synthetic structural tests only. No research theorem is adjudicated."""
from __future__ import annotations
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'withdrawal_check.py'
spec = importlib.util.spec_from_file_location('withdrawal_pilot', SCRIPT)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
H = 'a' * 64


def fixture(alternative=False):
    def node():
        return {'kind': 'CLAIM', 'contract_sha256': H, 'scope_sha256': H,
                'evidential_status': 'UNASSESSED', 'admissibility': 'ELIGIBLE',
                'scheduling': 'ACTIVE'}
    def warrant(n):
        return {'conclusion': n, 'artifact_sha256': H, 'scope_sha256': H,
                'admissibility': 'ELIGIBLE'}
    before = {'schema_version': 1, 'snapshot_id': 'SYNTHETIC-BEFORE',
              'coverage': {'scope_id': 'SYNTHETIC-ONLY', 'status': 'COMPLETE', 'review_sha256': H},
              'nodes': {x: node() for x in 'ABCD'},
              'grounds': {'GA': warrant('A'), 'GB': warrant('B'), 'GD': warrant('D')},
              'routes': {'ABC': dict(warrant('C'), requires=['A', 'B'])},
              'dashboard_population': list('ABCD')}
    if alternative:
        before['routes']['DC'] = dict(warrant('C'), requires=['D'])
    after = copy.deepcopy(before)
    after['snapshot_id'] = 'SYNTHETIC-AFTER'
    after['grounds']['GA']['admissibility'] = 'SUSPENDED'
    ticket = {'schema_version': 1, 'transition_id': 'SYNTHETIC-WITHDRAWAL-001',
              'operation': 'WITHDRAW_SUPPORT', 'author_session': 'SYNTHETIC-AUTHOR',
              'before_sha256': '', 'after_sha256': '',
              'reason': {'kind': 'PROOF_DEFECT', 'description': 'Synthetic defective warrant.', 'evidence_sha256': H},
              'changed_objects': ['grounds/GA'], 'expected_lost_support': [],
              'expected_supported_nodes': [],
              'publication': {'phase': 'PREPARED', 'drive': None, 'github': None}}
    seal(before, after, ticket)
    return before, after, ticket


def seal(b, a, t):
    t['before_sha256'] = w.canonical_sha(b)
    t['after_sha256'] = w.canonical_sha(a)
    sb, _ = w.support(b)
    sa, _ = w.support(a)
    t['expected_lost_support'] = sorted(sb - sa)
    t['expected_supported_nodes'] = sorted(sa)


class WithdrawalTests(unittest.TestCase):
    def reject(self, mutate):
        b, a, t = fixture()
        mutate(b, a, t)
        with self.assertRaises((w.Invalid, TypeError, KeyError)):
            w.validate_transition(b, a, t)

    def test_required_premise_loss(self):
        b, a, t = fixture()
        out = w.validate_transition(b, a, t)
        self.assertEqual(out['lost_support'], ['A', 'C'])
        self.assertEqual(out['disabled_routes']['ABC'], ['A', 'B'])
        self.assertFalse(out['scientific_acceptance'])

    def test_alternative_route_survives(self):
        b, a, t = fixture(True)
        out = w.validate_transition(b, a, t)
        self.assertEqual(out['lost_support'], ['A'])
        self.assertEqual(out['surviving_witnesses']['C'], ['routes/DC'])

    def test_premise_evaporation(self):
        def mutate(b, a, t):
            a['routes']['ABC']['requires'] = ['B']; seal(b, a, t)
        self.reject(mutate)

    def test_survivor_laundering(self):
        self.reject(lambda b, a, t: t['expected_supported_nodes'].append('C'))

    def test_impact_omission(self):
        self.reject(lambda b, a, t: t.update(expected_lost_support=['A']))

    def test_no_deletion(self):
        def mutate(b, a, t):
            del a['grounds']['GA']; seal(b, a, t)
        self.reject(mutate)

    def test_no_new_warrant(self):
        def mutate(b, a, t):
            a['grounds']['NEW'] = dict(a['grounds']['GB'], conclusion='C'); seal(b, a, t)
        self.reject(mutate)

    def test_no_scope_change(self):
        def mutate(b, a, t):
            a['nodes']['C']['scope_sha256'] = 'b' * 64; seal(b, a, t)
        self.reject(mutate)

    def test_no_contract_change(self):
        def mutate(b, a, t):
            a['nodes']['C']['contract_sha256'] = 'b' * 64; seal(b, a, t)
        self.reject(mutate)

    def test_no_falsehood_from_proof_failure(self):
        def mutate(b, a, t):
            a['nodes']['A']['evidential_status'] = 'REFUTED'
            a['nodes']['A']['admissibility'] = 'INELIGIBLE'; seal(b, a, t)
        self.reject(mutate)

    def test_no_dashboard_denominator_shrink(self):
        def mutate(b, a, t):
            a['dashboard_population'].remove('A'); seal(b, a, t)
        self.reject(mutate)

    def test_scheduling_is_not_truth(self):
        b, a, t = fixture()
        a = copy.deepcopy(b); a['snapshot_id'] = 'SYNTHETIC-PAUSED'
        a['nodes']['A']['scheduling'] = 'CLOSED'
        t.update(operation='CHANGE_SCHEDULE', changed_objects=['nodes/A'])
        t['reason'].update(kind='RESOURCE_ALLOCATION', evidence_sha256=None)
        seal(b, a, t)
        self.assertEqual(w.validate_transition(b, a, t)['lost_support'], [])

    def test_resource_reason_cannot_withdraw(self):
        self.reject(lambda b, a, t: t['reason'].update(kind='RESOURCE_ALLOCATION'))

    def test_stale_binding(self):
        self.reject(lambda b, a, t: t.update(before_sha256='f' * 64))

    def test_uncovered_change(self):
        self.reject(lambda b, a, t: t.update(changed_objects=[]))

    def test_no_restoration(self):
        b, a, t = fixture()
        b['grounds']['GA']['admissibility'] = 'INELIGIBLE'; seal(b, a, t)
        with self.assertRaises(w.Invalid): w.validate_transition(b, a, t)

    def test_partial_publication(self):
        self.reject(lambda b, a, t: t['publication'].update(phase='RECORDED'))

    def test_mixed_transaction(self):
        self.reject(lambda b, a, t: t['publication'].update(drive={
            'transaction_id': 'WRONG', 'snapshot_sha256': t['after_sha256'], 'reference': 'synthetic'}))

    def test_mixed_snapshot(self):
        self.reject(lambda b, a, t: t['publication'].update(github={
            'transaction_id': t['transition_id'], 'snapshot_sha256': H, 'reference': 'synthetic'}))

    def test_recorded_is_not_accepted(self):
        b, a, t = fixture()
        t['publication'] = {'phase': 'RECORDED', **{x: {
            'transaction_id': t['transition_id'], 'snapshot_sha256': t['after_sha256'],
            'reference': f'synthetic:{x}'} for x in ('drive', 'github')}}
        self.assertFalse(w.validate_transition(b, a, t)['scientific_acceptance'])

    def test_missing_coverage_review(self):
        self.reject(lambda b, a, t: a['coverage'].update(review_sha256=None))

    def test_partial_coverage_cannot_be_recorded(self):
        b, a, t = fixture()
        for s in (b, a): s['coverage'].update(status='PARTIAL')
        seal(b, a, t)
        t['publication'] = {'phase': 'RECORDED', **{x: {
            'transaction_id': t['transition_id'], 'snapshot_sha256': t['after_sha256'],
            'reference': 'synthetic'} for x in ('drive', 'github')}}
        with self.assertRaises(w.Invalid): w.validate_transition(b, a, t)

    def test_unknown_fields_fail_closed(self):
        self.reject(lambda b, a, t: t.update(accept=True))

    def test_wrong_version(self):
        self.reject(lambda b, a, t: t.update(schema_version=2))

    def test_boolean_not_integer_version(self):
        self.reject(lambda b, a, t: t.update(schema_version=True))

    def test_duplicate_dependency(self):
        self.reject(lambda b, a, t: a['routes']['ABC'].update(requires=['A', 'A']))

    def test_dangling_dependency(self):
        self.reject(lambda b, a, t: a['routes']['ABC'].update(requires=['MISSING']))

    def test_no_unconditional_empty_route(self):
        self.reject(lambda b, a, t: a['routes']['ABC'].update(requires=[]))

    def test_unseeded_cycle_has_no_support(self):
        b, _, _ = fixture()
        b['grounds'] = {}
        b['routes']['CA'] = dict(b['routes']['ABC'], conclusion='A', requires=['C'])
        self.assertEqual(w.support(b)[0], set())

    def test_legacy_inventory_never_promotes_empty_dependency_list(self):
        result = w.inventory({'claims': {'A': {'depends_on': [], 'grade': 'LIVE_ROOT_THEOREM'}}, 'premises': {}})
        self.assertEqual(result['objects'][0]['support_verdict'], 'NOT_ASSESSED')
        self.assertEqual(result['scientific_status_changes'], [])

    def test_empty_legacy_inventory_fails(self):
        with self.assertRaises(w.Invalid): w.inventory({'claims': {}, 'premises': {}})

    def test_exhaustive_monotonicity_81_assignments(self):
        b, _, _ = fixture(True)
        baseline = w.support(b)[0]
        for values in itertools.product(w.ADM, repeat=4):
            a = copy.deepcopy(b)
            for key, value in zip(('GA', 'GB', 'GD'), values[:3]):
                a['grounds'][key]['admissibility'] = value
            a['routes']['ABC']['admissibility'] = values[3]
            self.assertLessEqual(w.support(a)[0], baseline)

    def test_cli_actual_paths_and_optimized_mode(self):
        b, a, t = fixture()
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d) / name for name in ('before.json', 'after.json', 'ticket.json')]
            for path, obj in zip(paths, (b, a, t)): path.write_text(json.dumps(obj))
            args = ['check', '--before', str(paths[0]), '--after', str(paths[1]), '--ticket', str(paths[2])]
            for mode in ([], ['-O']):
                run = subprocess.run([sys.executable, *mode, str(SCRIPT), *args], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
            t['expected_lost_support'] = []
            paths[2].write_text(json.dumps(t))
            for mode in ([], ['-O']):
                run = subprocess.run([sys.executable, *mode, str(SCRIPT), *args], capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stdout)

    def test_cli_missing_input(self):
        run = subprocess.run([sys.executable, str(SCRIPT), 'inventory', '--legacy-graph', '/nonexistent/withdrawal.json'], capture_output=True)
        self.assertEqual(run.returncode, 2)

    def test_json_duplicate_key_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.json'; p.write_text('{"x": 1, "x": 2}')
            with self.assertRaises(w.Invalid): w.load(p)

    def test_json_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.json'; p.write_text('{"x": NaN}')
            with self.assertRaises(w.Invalid): w.load(p)


if __name__ == '__main__':
    unittest.main()
