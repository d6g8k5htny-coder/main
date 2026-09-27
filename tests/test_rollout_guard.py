"""Synthetic author-side controls; no review/authority is fabricated."""
import copy
import importlib.util
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'rollout_guard.py'
spec = importlib.util.spec_from_file_location('rollout_guard_pilot', SCRIPT)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
H = 'a' * 64


def sample():
    return {'schema_version': 1, 'change_id': 'SYNTHETIC-METADATA', 'change_kind': 'METADATA_REPAIR',
            'from_stage': 'BASELINE', 'to_stage': 'APPLIED_SCOPED', 'policy_sha256': H,
            'authority_ref': 'SYNTHETIC-AUTHORITY-ONLY', 'target_consumers': ['home', 'router'],
            'coverage_complete': True, 'rollback_sha256': H, 'review_sha256': None,
            'effects': ['navigation'], 'scientific_status_unchanged': True,
            'frozen_bytes_preserved': True, 'reactivate_superseded': False, 'blocked_consumers': [],
            'observations': [{'consumer': x, 'mode': 'READBACK', 'policy_sha256': H,
                              'artifact_sha256': H, 'result': 'PASS'} for x in ('home', 'router')]}


class RolloutControls(unittest.TestCase):
    def reject(self, change):
        p = sample(); change(p)
        with self.assertRaises((g.Invalid, ValueError, TypeError)):
            g.assess(p)

    def test_small_repair_does_not_require_six_new_stages(self):
        result = g.assess(sample())
        self.assertEqual(result['result'], 'PLAN_CONSISTENT')
        self.assertFalse(result['live_admission'])
        self.assertFalse(result['scientific_acceptance'])

    def test_start_here_24_misses_two_current_keys(self):
        self.assertEqual(g.entry_coverage(100, 24, {'withdrawal': 31, 'rollout': 32}, True), ['rollout', 'withdrawal'])

    def test_start_here_31_still_misses_audit(self):
        self.assertEqual(g.entry_coverage(100, 31, {'withdrawal': 31, 'rollout': 32}, True), ['rollout'])

    def test_allocated_compact_view_covers_new_rows(self):
        self.assertEqual(g.entry_coverage(100, 100, {'withdrawal': 31, 'rollout': 32, 'future': 99}, True), [])

    def test_truncated_discovery_not_complete(self):
        with self.assertRaises(g.Invalid): g.entry_coverage(100, 100, {'rollout': 32}, False)

    def test_unknown_grid_bound_fails(self):
        with self.assertRaises(g.Invalid): g.entry_coverage(100, 101, {'rollout': 32}, True)

    def test_empty_key_inventory_fails(self):
        with self.assertRaises(g.Invalid): g.entry_coverage(100, 100, {}, True)

    def test_out_of_bounds_required_key_fails(self):
        with self.assertRaises(g.Invalid): g.entry_coverage(100, 100, {'future': 101}, True)

    def test_boolean_row_rejected(self):
        with self.assertRaises(g.Invalid): g.entry_coverage(100, 100, {'future': True}, True)

    def test_binding_letter_a_is_not_globally_typed(self):
        self.assertEqual(g.binding_kind('OP-PROT-013@2026-07-26', 'A'), 'RAW_CONTENT')
        self.assertEqual(g.binding_kind('OP-PROT-014-v1.0', 'A'), 'METADATA_ONLY')

    def test_binding_letter_b_is_not_globally_typed(self):
        self.assertNotEqual(g.binding_kind('OP-PROT-013@2026-07-26', 'B'), g.binding_kind('OP-PROT-014-v1.0', 'B'))

    def test_unqualified_binding_fails(self):
        with self.assertRaises(g.Invalid): g.binding_kind('', 'A')

    def test_new_binding_version_needs_explicit_mapping(self):
        with self.assertRaises(g.Invalid): g.binding_kind('OP-PROT-014-v2.0', 'A')

    def test_current_authorization_not_revoked_by_unobserved_uptake(self):
        g.effective_rule('R17:entry', True, 'R17:entry', ['R16:entry'])

    def test_stricter_old_rule_cannot_resurrect(self):
        with self.assertRaises(g.Invalid): g.effective_rule('R17:entry', True, 'R16:entry', ['R16:entry'])

    def test_no_authority_from_newer_name(self):
        with self.assertRaises(g.Invalid): g.effective_rule('R18:entry', False, 'R18:entry', [])

    def test_no_global_metadata_block(self):
        self.reject(lambda p: p.update(blocked_consumers=['unrelated-math']))

    def test_no_new_local_block_disguised_as_metadata(self):
        self.reject(lambda p: p.update(blocked_consumers=['home']))

    def test_no_scientific_status_change(self):
        self.reject(lambda p: p.update(scientific_status_unchanged=False))

    def test_no_frozen_edit(self):
        self.reject(lambda p: p.update(frozen_bytes_preserved=False))

    def test_no_retired_control_reactivation(self):
        self.reject(lambda p: p.update(reactivate_superseded=True))

    def test_no_semantic_change_called_metadata(self):
        self.reject(lambda p: p.update(effects=['admission']))

    def test_missing_readback_is_not_applied(self):
        self.reject(lambda p: p['observations'].pop())

    def test_wrong_policy_observation(self):
        self.reject(lambda p: p['observations'][0].update(policy_sha256='b' * 64))

    def test_failed_consumer_is_not_pass(self):
        self.reject(lambda p: p['observations'][0].update(result='FAIL'))

    def test_empty_cohort_cannot_pass_vacuously(self):
        self.reject(lambda p: p.update(target_consumers=[], observations=[]))

    def test_duplicate_consumer(self):
        self.reject(lambda p: p['target_consumers'].append('home'))

    def test_duplicate_observation(self):
        self.reject(lambda p: p['observations'].append(p['observations'][0]))

    def test_unknown_schema(self):
        self.reject(lambda p: p.update(schema_version=2))

    def test_boolean_schema_version(self):
        self.reject(lambda p: p.update(schema_version=True))

    def test_unknown_field(self):
        self.reject(lambda p: p.update(auto_approve=True))

    def test_incomplete_coverage(self):
        self.reject(lambda p: p.update(coverage_complete=False))

    def test_material_rollout_cannot_skip_to_global(self):
        self.reject(lambda p: p.update(change_kind='BEHAVIOR_CHANGE', from_stage='PROPOSED', to_stage='ACTIVE_GENERAL'))

    def test_material_pilot_needs_actual_supplied_pilot_observations(self):
        self.reject(lambda p: p.update(change_kind='BEHAVIOR_CHANGE', from_stage='SHADOW', to_stage='PILOT'))

    def test_supplied_shadow_plan_remains_non_authoritative(self):
        p = sample(); p.update(change_kind='BEHAVIOR_CHANGE', from_stage='PROPOSED', to_stage='SHADOW')
        for x in p['observations']: x['mode'] = 'SHADOW'
        self.assertFalse(g.assess(p)['live_admission'])

    def test_review_stage_needs_review_reference(self):
        p = sample(); p.update(change_kind='BEHAVIOR_CHANGE', from_stage='PILOT', to_stage='REVIEWED')
        p['observations'] = [dict(x, mode=m) for x in p['observations'] for m in ('SHADOW', 'PILOT')]
        with self.assertRaises(g.Invalid): g.assess(p)

    def test_containment_not_automatic_global_replacement(self):
        self.reject(lambda p: p.update(change_kind='EMERGENCY_CONTAINMENT'))

    def test_fresh_pinned_execution_observation(self):
        s = {'base_head': 'a'*40, 'candidate_head': 'b'*40, 'policy_sha256': H}
        g.execution_snapshot(s, s, '2026-09-21T22:00:00Z', '2026-09-22T00:00:00Z', '2026-09-21T23:00:00Z')

    def test_changed_base_head_rejected_even_without_text_conflict(self):
        s = {'base_head': 'a'*40, 'candidate_head': 'b'*40, 'policy_sha256': H}
        with self.assertRaises(g.Invalid):
            g.execution_snapshot(s, dict(s, base_head='c'*40), '2026-09-21T22:00:00Z', '2026-09-22T00:00:00Z', '2026-09-21T23:00:00Z')

    def test_expired_lease_cannot_be_renewed_retroactively(self):
        s = {'base_head': 'a'*40, 'candidate_head': 'b'*40, 'policy_sha256': H}
        with self.assertRaises(g.Invalid):
            g.execution_snapshot(s, s, '2026-09-21T17:20:00Z', '2026-09-21T19:20:00Z', '2026-09-21T22:00:00Z')

    def test_timestamp_requires_timezone(self):
        with self.assertRaises(g.Invalid): g.timestamp('2026-09-21T22:00:00')

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'a.json'; p.write_text('{"a":1,"a":2}')
            with self.assertRaises(g.Invalid): g.load(p)

    def test_cli_checks_actual_mutated_file_in_both_modes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'plan.json'
            for valid in (True, False):
                p = sample()
                if not valid: p['scientific_status_unchanged'] = False
                path.write_text(json.dumps(p))
                for mode in ([], ['-O']):
                    r = subprocess.run([sys.executable, *mode, str(SCRIPT), str(path)], capture_output=True)
                    self.assertEqual(r.returncode, 0 if valid else 2, r.stderr)

    def test_real_git_clean_merge_can_break_semantic_interface(self):
        if shutil.which('git') is None:
            self.skipTest('git unavailable; no merge experiment claimed')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def git(*args):
                return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
            git('init', '-q', '-b', 'base')
            git('config', 'user.name', 'Synthetic test')
            git('config', 'user.email', 'synthetic@example.invalid')
            (root/'producer.json').write_text(json.dumps({'interface': 1}))
            (root/'consumer.json').write_text(json.dumps({'expects': 1, 'uses': 'old'}))
            git('add', '.'); git('commit', '-qm', 'synthetic baseline')
            baseline = git('rev-parse', 'HEAD')
            git('checkout', '-qb', 'producer-change')
            (root/'producer.json').write_text(json.dumps({'interface': 2}))
            git('commit', '-qam', 'new producer interface')
            git('checkout', '-qb', 'consumer-change', baseline)
            (root/'consumer.json').write_text(json.dumps({'expects': 1, 'uses': 'new-dependent'}))
            git('commit', '-qam', 'old interface still assumed')
            git('merge', '--no-ff', '-m', 'syntactically clean merge', 'producer-change')
            self.assertNotEqual(json.loads((root/'producer.json').read_text())['interface'],
                                json.loads((root/'consumer.json').read_text())['expects'])
            self.assertEqual(git('status', '--porcelain'), '')

    def test_missing_file_fails_cli(self):
        r = subprocess.run([sys.executable, str(SCRIPT), '/nonexistent/rollout-plan.json'], capture_output=True)
        self.assertEqual(r.returncode, 2)

    def test_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'a.json'; p.write_text('{"a":NaN}')
            with self.assertRaises(g.Invalid): g.load(p)


if __name__ == '__main__': unittest.main()
