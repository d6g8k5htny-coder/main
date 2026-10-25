"""Real journal, process contention, failure-specific decisions and stale guards."""
import concurrent.futures
import copy
import hashlib
import inspect
import json
import multiprocessing
import os
import sys
import types
from unittest.mock import patch
import tempfile
import shutil
import time
import unittest
from pathlib import Path

try:
    from proof_conductor import ProofConductor, DIAGNOSES, STRATEGIES
except ImportError:
    ProofConductor = None
    DIAGNOSES = STRATEGIES = {}


def contend(directory, task_id, agent, queue):
    try:
        queue.put(ProofConductor(directory).claim(task_id, agent))
    except ValueError as exc:
        queue.put(str(exc))


class ProofConductorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ProofConductor, 'global semantic proof conductor is missing')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'frontier'
        self.conductor = ProofConductor(self.directory, None)

    def plan(self, targets=None, budget=20):
        return self.conductor.blueprint(targets or ['D1-KR-BOREL'], budget)

    def ready(self):
        return next(t for t in self.conductor.snapshot()['tasks'] if t['claimable'])

    def test_blueprint_preserves_and_dependencies_and_minimal_missing_frontier(self):
        bp = self.plan()
        state = self.conductor.snapshot()
        record = next(t for t in state['tasks'] if t['kind'] == 'record' and t['subject_id'] == 'D1-KR-BOREL')
        parent_subjects = {t['subject_id'] for t in state['tasks'] if t['task_id'] in record['dependencies']}
        self.assertTrue({'D1-ELDER-MEAS','D1-KR-KERNEL'}.issubset(parent_subjects))
        self.assertFalse(record['claimable'])
        frontier = {t['task_id']:t for t in state['tasks'] if t['task_id'] in bp['blocking_frontier']}
        self.assertTrue(frontier)
        self.assertTrue(all(not t['dependencies'] for t in frontier.values()))
        self.assertTrue({'EXT-C2-KERNEL','EXT-KR-AAL','SRC-D1-ERRATUM'}.issubset({t['subject_id'] for t in frontier.values()}))
        self.assertGreater(max(t['downstream_impact'] for t in frontier.values()), 1)
        self.assertTrue(bp['necessary_not_sufficient'])
        self.assertEqual(bp['target_ids'], ['D1-KR-BOREL'])

    def test_legacy_blueprint_identity_replays_and_migrates_without_new_task_or_budget(self):
        from conductor import Journal
        from frontier_ir import digest
        original = self.plan(['D5d-M'],budget=3)
        legacy_root = Path(self.temp.name)/'legacy-identity'
        journal = Journal(legacy_root)
        for event in self.conductor.journal.events:
            payload = copy.deepcopy(event['payload'])
            if event['type']=='BLUEPRINT_CREATED':
                bp = payload['blueprint']
                bp.pop('identity_version',None)
                bp['blueprint_id'] = 'bp-'+digest({'ir':bp['ir_sha256'],'targets':bp['target_ids']})[:24]
            journal.append(event['type'],payload)
        c = ProofConductor(legacy_root,None)
        before = c.snapshot()
        replacement = c.blueprint(['D5d-M'],budget=3)
        after = c.snapshot()
        self.assertNotEqual(before['blueprints'][0]['blueprint_id'],replacement['blueprint_id'])
        self.assertEqual(replacement['identity_version'],2)
        self.assertEqual(len(after['tasks']),len(before['tasks']))
        self.assertEqual(after['budget']['limit'],3)
        self.assertEqual(after['budget']['remaining'],3)
        task = next(t for t in after['tasks'] if t['claimable'])
        c.claim(task['task_id'],'same-question')
        self.assertEqual(c.snapshot()['budget']['remaining'],2)

    def test_shared_tasks_and_duplicate_blueprints_do_not_grant_extra_budget(self):
        bp = self.plan(budget=3)
        first_count = len(self.conductor.snapshot()['tasks'])
        self.assertEqual(self.plan(budget=3), bp)
        self.plan(['D1-KR-KERNEL'], budget=3)
        state = self.conductor.snapshot()
        self.assertEqual(len(state['tasks']), first_count)
        self.assertEqual(len(state['blueprints']), 2)
        self.assertEqual(state['budget']['limit'], 3)
        self.assertTrue(any(len(t['blueprint_ids']) == 2 for t in state['tasks']))
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.plan(['D5d-M'], budget=4)

    def test_d5_missing_collision_interface_is_explicit_and_not_a_canonical_edge(self):
        bp = self.plan(['d5-second-factorial-upper'])
        state = self.conductor.snapshot()
        interface = next(t for t in state['tasks'] if t['kind'] == 'research_interface')
        self.assertIn('collision', interface['intended_question'].lower())
        self.assertIn('original conditioned and endpoint-tilted law', interface['intended_question'])
        self.assertTrue(interface['claimable'])
        self.assertFalse(any(n['id'] == interface['subject_id'] for n in state['ir']['nodes']))
        self.assertIn(interface['task_id'], bp['blocking_frontier'])

    def test_claim_binds_exact_contract_and_idempotent_owner_does_not_double_charge(self):
        self.plan(budget=2)
        task = self.ready()
        claim = self.conductor.claim(task['task_id'], 'worker')
        self.assertEqual(claim['input_fingerprint'], task['input_fingerprint'])
        self.assertEqual(claim['question_sha256'], task['question_sha256'])
        self.assertEqual(claim['scope_sha256'], task['scope_sha256'])
        self.assertEqual(self.conductor.claim(task['task_id'], 'worker'), claim)
        with self.assertRaisesRegex(ValueError, 'leased|claim'):
            self.conductor.claim(task['task_id'], 'competitor')
        self.assertEqual(self.conductor.snapshot()['budget']['remaining'], 1)

    def test_process_contention_has_one_winner_and_one_reservation(self):
        self.plan(budget=3)
        task = self.ready()
        queue = multiprocessing.Queue()
        processes = [multiprocessing.Process(target=contend, args=(self.directory,task['task_id'],str(i),queue)) for i in range(3)]
        for p in processes: p.start()
        for p in processes: p.join(15)
        self.assertTrue(all(not p.is_alive() for p in processes))
        results = [queue.get(timeout=2) for _ in processes]
        self.assertEqual(sum(isinstance(r,dict) for r in results), 1)
        self.assertEqual(self.conductor.snapshot()['budget']['reserved'], 1)

    def test_global_budget_reservation_survives_restart_and_is_not_refunded(self):
        self.plan(budget=1)
        claim = self.conductor.claim(self.ready()['task_id'], 'worker')
        self.conductor.submit_failure(claim['claim_id'], 'timeout')
        restarted = ProofConductor(self.directory, None)
        self.assertEqual(restarted.snapshot()['budget']['remaining'], 0)
        self.assertEqual(restarted.snapshot()['budget']['spent'], 1)
        with self.assertRaisesRegex(ValueError, 'budget'):
            restarted.claim(self.ready()['task_id'], 'new-worker')

    def test_expired_lease_cannot_report_after_restart(self):
        self.plan(budget=3)
        task = self.ready()
        claim = self.conductor.claim(task['task_id'], 'first', lease_seconds=.01)
        time.sleep(.02)
        restarted = ProofConductor(self.directory, None)
        new = restarted.claim(task['task_id'], 'second')
        self.assertNotEqual(claim['claim_id'], new['claim_id'])
        self.assertNotEqual(claim['strategy'], new['strategy'])
        with self.assertRaisesRegex(ValueError, 'expired|lease'):
            restarted.submit_failure(claim['claim_id'], 'timeout')
        self.assertEqual(len(restarted.snapshot()['claims']), 2)

    def test_each_typed_failure_preserves_question_and_selects_different_strategy(self):
        expected = {'insufficient_guidance':'guide','false_lemma':'correct','invalid_decomposition':'replan',
                    'missing_source':'guide','checker_failure':'replan','timeout':'replan','statement_drift':'correct'}
        self.assertEqual(set(DIAGNOSES), set(expected))
        for diagnosis, action in expected.items():
            with self.subTest(diagnosis=diagnosis), tempfile.TemporaryDirectory() as directory:
                c = ProofConductor(directory, None)
                c.blueprint(['D1-KR-KERNEL'], budget=4)
                task = next(t for t in c.snapshot()['tasks'] if t['claimable'])
                claim = c.claim(task['task_id'], 'worker')
                result = c.submit_failure(claim['claim_id'], diagnosis, {'report':'untrusted observation'})
                decision = result['decision']
                self.assertEqual(decision['action'], action)
                self.assertNotEqual(decision['strategy'], claim['strategy'])
                self.assertIn(decision['strategy'], STRATEGIES)
                next_claim = c.claim(task['task_id'], 'repair-worker')
                self.assertEqual(next_claim['strategy'], decision['strategy'])
                self.assertEqual(next_claim['question_sha256'], claim['question_sha256'])
                self.assertEqual(next_claim['scope_sha256'], claim['scope_sha256'])
                state = ProofConductor(directory, None).snapshot()
                self.assertEqual(len(state['attempts']), 1)
                self.assertEqual(state['attempts'][0]['kind'], 'agent_failure_report')
                self.assertEqual(state['full_theorem_certificates'], 0)
                self.assertEqual(state['ir']['ir_sha256'], c.snapshot()['ir']['ir_sha256'])

    def test_false_lemma_report_opens_investigation_without_refuting_or_promoting(self):
        self.plan()
        claim = self.conductor.claim(self.ready()['task_id'], 'worker')
        before = self.conductor.snapshot()['ir']
        result = self.conductor.submit_failure(claim['claim_id'], 'false_lemma', {'conclusion':'false'})
        self.assertEqual(result['decision']['disposition'], 'investigate_untrusted_false_lemma_report')
        self.assertEqual(self.conductor.snapshot()['ir'], before)
        self.assertFalse(any(t.get('completed') for t in self.conductor.snapshot()['tasks']))

    def test_duplicate_failure_is_idempotent_and_conflicting_redelivery_rejected(self):
        self.plan()
        claim = self.conductor.claim(self.ready()['task_id'], 'worker')
        result = self.conductor.submit_failure(claim['claim_id'], 'missing_source', {'a':'b'})
        self.assertEqual(self.conductor.submit_failure(claim['claim_id'], 'missing_source', {'a':'b'}),result)
        with self.assertRaisesRegex(ValueError, 'different|already'):
            self.conductor.submit_failure(claim['claim_id'], 'false_lemma')
        self.assertEqual(len(self.conductor.snapshot()['decisions']),1)

    def test_attempt_cap_and_repeating_failed_strategy_are_rejected(self):
        self.plan(budget=20)
        task = self.ready()
        used = []
        for _ in range(4):
            claim = self.conductor.claim(task['task_id'], 'worker')
            used.append(claim['strategy'])
            self.conductor.submit_failure(claim['claim_id'], 'timeout')
        self.assertEqual(len(set(used)),4)
        with self.assertRaisesRegex(ValueError, 'attempt|exhausted'):
            self.conductor.claim(task['task_id'], 'worker')

    def test_unknown_strategy_diagnosis_agent_effects_and_unbounded_inputs_rejected(self):
        self.plan()
        task = self.ready()
        for strategy in ('exec:rm','agent.approve','checker.write'):
            with self.subTest(strategy=strategy), self.assertRaises(ValueError):
                self.conductor.claim(task['task_id'], 'worker', strategy)
        claim = self.conductor.claim(task['task_id'], 'worker')
        with self.assertRaises(ValueError):
            self.conductor.submit_failure(claim['claim_id'], 'none')
        for evidence in (float('nan'), {'x':'a'*100001}, object()):
            with self.assertRaises(ValueError):
                self.conductor.submit_failure(claim['claim_id'], 'timeout', evidence)
        with self.assertRaises(ValueError):
            self.conductor.claim(task['task_id'], 'w', lease_seconds=True)
        self.assertEqual(len(self.conductor.snapshot()['claims']),1)

    def test_unknown_and_duplicate_targets_and_invalid_budget_rejected(self):
        for targets in ([], ['unknown'], ['D5d-M','D5d-M'], 'D5d-M'):
            with self.assertRaises(ValueError):
                self.conductor.blueprint(targets)
        for budget in (True,0,1001,1.5):
            with self.assertRaises(ValueError):
                self.conductor.blueprint(['D5d-M'],budget)

    def test_live_coordinator_changes_invalidate_claim_and_old_blueprint(self):
        from conductor import Coordinator
        coordinator = Coordinator(Path(self.temp.name) / 'engine')
        c = ProofConductor(self.directory,coordinator)
        c.blueprint(['D1-KR-KERNEL'], budget=3)
        task = next(t for t in c.snapshot()['tasks'] if t['claimable'])
        claim = c.claim(task['task_id'],'worker')
        record = next(r for r in coordinator.snapshot()['records'] if r['id']=='D1-KR-KERNEL')
        proposal = coordinator.propose('statement',record['id'],record['statement']+' Reviewed branch amendment.')
        coordinator.decide(proposal['proposal_id'],True)
        with self.assertRaisesRegex(ValueError, 'stale'):
            c.submit_failure(claim['claim_id'],'statement_drift')
        with self.assertRaisesRegex(ValueError, 'stale'):
            c.claim(task['task_id'],'worker')
        self.assertTrue(c.snapshot()['blueprints'][0]['stale'])
        c.blueprint(['D1-KR-KERNEL'],budget=3)
        self.assertEqual(len(c.snapshot()['blueprints']),2)
        self.assertEqual(c.snapshot()['budget']['remaining'],2)

    def test_hash_valid_journal_cannot_install_forged_semantics_or_authority(self):
        from conductor import Journal
        self.plan()
        journal = Journal(self.directory)
        journal.append('MATH_APPROVED', {'full_theorem_certificates':1})
        with self.assertRaisesRegex(ValueError, 'event|authority'):
            ProofConductor(self.directory,None)

    def test_hash_valid_blueprint_forgery_is_rejected_by_independent_reconstruction(self):
        from conductor import Journal
        self.plan()
        original = self.conductor.journal.events
        other = Path(self.temp.name) / 'forged'
        journal = Journal(other)
        for event in original:
            payload = copy.deepcopy(event['payload'])
            if event['type'] == 'BLUEPRINT_CREATED':
                payload['tasks'][0]['scope'] = {'kind':'unrestricted'}
            journal.append(event['type'],payload)
        with self.assertRaisesRegex(ValueError, 'blueprint|contract'):
            ProofConductor(other,None)

    def test_shared_correction_priorities_do_not_change_between_target_blueprints(self):
        self.plan(['D1-A'])
        self.plan(['D1-B'])
        state = self.conductor.snapshot()
        corrections = [t for t in state['tasks'] if t['kind']=='correction']
        self.assertTrue(corrections)
        self.assertTrue(all(len(t['blueprint_ids'])==2 for t in corrections))

    def test_unrelated_branch_change_reuses_task_and_preserves_active_claim(self):
        from conductor import Coordinator
        coordinator = Coordinator(Path(self.temp.name)/'engine')
        c = ProofConductor(self.directory,coordinator)
        c.blueprint(['D1-A'],budget=4)
        task = next(t for t in c.snapshot()['tasks'] if t['kind']=='side_condition')
        claim = c.claim(task['task_id'],'worker')
        unrelated = next(r for r in coordinator.snapshot()['records'] if r['id']=='C6-01')
        proposal = coordinator.propose('statement',unrelated['id'],unrelated['statement']+' Unrelated reviewed amendment.')
        coordinator.decide(proposal['proposal_id'],True)
        self.assertFalse(next(t for t in c.snapshot()['tasks'] if t['task_id']==task['task_id'])['stale'])
        c.blueprint(['D1-A'],budget=4)
        self.assertEqual(sum(t['task_id']==task['task_id'] for t in c.snapshot()['tasks']),1)
        c.submit_failure(claim['claim_id'],'insufficient_guidance')
        self.assertEqual(c.snapshot()['budget']['remaining'],3)

    def test_approved_research_attachment_changes_only_its_bound_mathematical_inputs(self):
        from conductor import Coordinator
        from flow import Intake
        coordinator = Coordinator(Path(self.temp.name)/'engine')
        c = ProofConductor(self.directory,coordinator)
        c.blueprint(['D1-A','C6-01'],budget=4)
        state = c.snapshot()
        related = next(t for t in state['tasks'] if t['kind']=='side_condition' and t['bound_record_ids']==['D1-A'])
        unrelated = next(t for t in state['tasks'] if t['kind']=='side_condition' and t['bound_record_ids']==['C6-01'])
        claim = c.claim(related['task_id'],'worker')
        sources = Path(self.temp.name)/'sources'
        sources.mkdir()
        (sources/'finding.md').write_text('Preserved contextual observation; applicability unvalidated.')
        config = {'source_namespace':'frontier-test','patterns':['*.md'],'auto_register_evidence':True,
            'rules':{'finding.md':{'classification':'unreviewed','scope':'Context only.',
                'limitations':['No mathematical applicability established.'],
                'links':[{'record_id':'D1-A','relation':'context_only','reason':'Input custody test.'}]}}}
        intake = Intake(coordinator,sources,config)
        intake.capture('finding.md')
        intake.recover_links()
        state = c.snapshot()
        self.assertTrue(next(t for t in state['tasks'] if t['task_id']==related['task_id'])['stale'])
        self.assertFalse(next(t for t in state['tasks'] if t['task_id']==unrelated['task_id'])['stale'])
        with self.assertRaisesRegex(ValueError,'stale'):
            c.submit_failure(claim['claim_id'],'missing_source')
        c.blueprint(['D1-A','C6-01'],budget=4)
        self.assertEqual(c.snapshot()['full_theorem_certificates'],0)

    def test_scalar_pilot_contract_survives_unrelated_canonical_branch_change(self):
        from conductor import Coordinator
        coordinator = Coordinator(Path(self.temp.name)/'engine')
        c = ProofConductor(self.directory,coordinator)
        c.blueprint(['scalar-variance-bounds'],budget=4)
        task = next(t for t in c.snapshot()['tasks'] if t['kind']=='scalar_pilot')
        claim = c.claim(task['task_id'],'worker')
        record = coordinator.snapshot()['records'][0]
        proposal = coordinator.propose('statement',record['id'],record['statement']+' Separate branch amendment.')
        coordinator.decide(proposal['proposal_id'],True)
        c.blueprint(['scalar-variance-bounds'],budget=4)
        state = c.snapshot()
        self.assertEqual(len(state['tasks']),1)
        self.assertFalse(state['tasks'][0]['stale'])
        self.assertEqual(state['claims'][0]['state'],'active')
        c.submit_failure(claim['claim_id'],'checker_failure')

    def test_scalar_claim_uses_only_fixed_registered_worker_strategy(self):
        self.plan(['scalar-variance-bounds','D5d-M'])
        state = self.conductor.snapshot()
        pilot = next(t for t in state['tasks'] if t['kind']=='scalar_pilot')
        claim = self.conductor.claim(pilot['task_id'],'scalar')
        self.assertEqual(claim['strategy'],'native_scalar_profile')
        ordinary = next(t for t in state['tasks'] if t['claimable'] and t['kind']!='scalar_pilot')
        with self.assertRaisesRegex(ValueError,'strategy|pilot'):
            self.conductor.claim(ordinary['task_id'],'ordinary',strategy='native_scalar_profile')

    def test_hash_valid_path_traversal_claim_is_rejected_on_journal_replay(self):
        from conductor import Journal
        self.plan()
        self.conductor.claim(self.ready()['task_id'],'worker')
        events = self.conductor.journal.events
        journal = Journal(Path(self.temp.name)/'forged-path')
        for event in events:
            payload = copy.deepcopy(event['payload'])
            if event['type']=='TASK_CLAIMED':
                payload['claim_id'] = '../escaped'
                payload['artifact_directory'] = str(journal.workdir.resolve()/'artifacts'/'../escaped')
            journal.append(event['type'],payload)
        with self.assertRaisesRegex(ValueError,'claim id|claim identifier'):
            ProofConductor(journal.workdir,None)

    def test_scalar_pilot_is_exactly_bound_and_cannot_close_actual_field_tasks(self):
        from polynomial_prover import profile_contract
        bp = self.plan(['scalar-variance-bounds'])
        state = self.conductor.snapshot()
        task = next(t for t in state['tasks'] if t['kind']=='scalar_pilot')
        self.assertEqual(task['profile_contract'], profile_contract())
        self.assertFalse(task['scope']['actual_field_applicable'])
        self.assertEqual(len(task['profile_contract']['statements']),2)
        self.assertEqual(task['bound_record_ids'],[])
        self.assertIn('checker.fixed_scalar_replay',task['effects'])
        self.assertEqual(state['capabilities']['registered_checkers']['native_scalar_profile']['target_kind'],'scalar_pilot')
        self.assertTrue(task['claimable'])
        self.assertEqual(len(state['ir']['nodes']),104)
        claim = self.conductor.claim(task['task_id'],'scalar-worker')
        self.assertEqual(Path(claim['artifact_directory']).parent, (self.directory/'artifacts').resolve())
        self.assertEqual(Path(claim['artifact_directory']).name,claim['claim_id'])
        self.assertTrue(Path(claim['artifact_directory']).is_dir())

    def test_scalar_completion_replays_real_native_artifact_and_preserves_failed_search(self):
        from polynomial_prover import prove_profile
        self.plan(['scalar-variance-bounds'])
        task = self.ready()
        claim = self.conductor.claim(task['task_id'],'scalar-worker',lease_seconds=180)
        receipt = prove_profile(claim['artifact_directory'])
        self.assertEqual(receipt['status'],'scoped_kernel_checked',receipt.get('reason'))
        result = self.conductor.complete_claim(claim['claim_id'],receipt['receipt_path'])
        self.assertEqual(result['kind'],'verified_scalar_pilot')
        self.assertTrue(result['verification']['fresh_replay'])
        self.assertEqual(len(result['worker_attempts']),2)
        self.assertNotEqual(result['worker_attempts'][0]['strategy'],result['worker_attempts'][1]['strategy'])
        self.assertNotEqual(result['worker_attempts'][0]['exit_code'],0)
        self.assertEqual(result['worker_attempts'][1]['exit_code'],0)
        state = ProofConductor(self.directory,None).snapshot()
        self.assertTrue(next(t for t in state['tasks'] if t['task_id']==task['task_id'])['completed'])
        self.assertEqual(state['claims'][0]['state'],'completed_scalar_pilot')
        self.assertEqual(state['budget']['spent'],1)
        self.assertEqual(state['full_theorem_certificates'],0)
        self.assertEqual(len(state['ir']['nodes']),104)
        # A second submitted result never becomes a new authority or a budget refund.
        with self.assertRaisesRegex(ValueError,'completed|pending'):
            self.conductor.submit_failure(claim['claim_id'],'timeout')

    def test_scalar_technical_failure_retries_fixed_worker_with_new_reservation(self):
        from polynomial_prover import prove_profile
        self.plan(['scalar-variance-bounds'],budget=3)
        task = self.ready()
        first = self.conductor.claim(task['task_id'],'worker')
        blocked = prove_profile(first['artifact_directory'],toolchain=Path(self.temp.name)/'absent-toolchain')
        self.assertEqual(blocked['status'],'blocked')
        self.conductor.submit_failure(first['claim_id'],'checker_failure',{'reason':blocked['reason']})
        second = self.conductor.claim(task['task_id'],'recovered-worker')
        self.assertNotEqual(first['claim_id'],second['claim_id'])
        self.assertEqual(second['strategy'],'native_scalar_profile')
        self.assertEqual(self.conductor.snapshot()['budget']['remaining'],1)
        self.assertEqual(first['input_fingerprint'],second['input_fingerprint'])

    def test_scalar_expired_process_claim_has_bounded_native_retry_after_restart(self):
        self.plan(['scalar-variance-bounds'],budget=5)
        task = self.ready()
        first = self.conductor.claim(task['task_id'],'dead-process',lease_seconds=.01)
        time.sleep(.02)
        restarted = ProofConductor(self.directory,None)
        second = restarted.claim(task['task_id'],'restarted-process')
        self.assertEqual(second['strategy'],'native_scalar_profile')
        self.assertNotEqual(second['claim_id'],first['claim_id'])
        with self.assertRaisesRegex(ValueError,'expired|lease'):
            restarted.submit_failure(first['claim_id'],'timeout')
        restarted.submit_failure(second['claim_id'],'timeout')
        for _ in range(2):
            attempt = restarted.claim(task['task_id'],'retry')
            self.assertEqual(attempt['strategy'],'native_scalar_profile')
            restarted.submit_failure(attempt['claim_id'],'checker_failure')
        with self.assertRaisesRegex(ValueError,'attempt|exhausted'):
            restarted.claim(task['task_id'],'too-many')
        self.assertEqual(restarted.snapshot()['budget']['remaining'],1)

    def test_scalar_mathematical_failure_requires_different_investigation_strategy(self):
        self.plan(['scalar-variance-bounds'])
        task = self.ready()
        first = self.conductor.claim(task['task_id'],'worker')
        self.conductor.submit_failure(first['claim_id'],'false_lemma')
        with self.assertRaisesRegex(ValueError,'strategy'):
            self.conductor.claim(task['task_id'],'worker',strategy='native_scalar_profile')
        second = self.conductor.claim(task['task_id'],'investigator')
        self.assertNotEqual(first['strategy'],second['strategy'])
        with self.assertRaisesRegex(ValueError,'strategy'):
            self.conductor.complete_claim(second['claim_id'],Path(second['artifact_directory'])/'receipt.json')

    def test_completed_scalar_runtime_relocates_and_replays_without_original_paths(self):
        from polynomial_prover import prove_profile
        self.plan(['scalar-variance-bounds'])
        claim = self.conductor.claim(self.ready()['task_id'],'scalar-worker',lease_seconds=180)
        receipt = prove_profile(claim['artifact_directory'])
        self.assertEqual(receipt['status'],'scoped_kernel_checked',receipt.get('reason'))
        self.conductor.complete_claim(claim['claim_id'],receipt['receipt_path'])
        destination = Path(self.temp.name)/'relocated'/'frontier'
        shutil.copytree(self.directory,destination)
        self.directory.rename(Path(self.temp.name)/'original-hidden')
        state = ProofConductor(destination,None).snapshot()
        self.assertEqual(state['claims'][0]['state'],'completed_scalar_pilot')
        self.assertTrue(Path(state['claims'][0]['artifact_directory']).is_relative_to(destination))
        self.assertTrue(Path(state['completions'][0]['receipt_path']).is_relative_to(destination))
        self.assertTrue(state['completions'][0]['verification']['fresh_replay'])
        self.assertEqual(state['full_theorem_certificates'],0)

    def test_completed_pilot_missing_backend_is_history_until_independent_replay_recovers(self):
        from polynomial_prover import prove_profile
        self.plan(['scalar-variance-bounds'])
        claim = self.conductor.claim(self.ready()['task_id'],'scalar-worker',lease_seconds=180)
        receipt = prove_profile(claim['artifact_directory'])
        self.assertEqual(receipt['status'],'scoped_kernel_checked',receipt.get('reason'))
        self.conductor.complete_claim(claim['claim_id'],receipt['receipt_path'])
        missing_backend = Path(self.temp.name)/'missing-backend'
        blocked_conductor = ProofConductor(self.directory,None,toolchain=missing_backend)
        blocked = blocked_conductor.snapshot()
        self.assertEqual(len(blocked['completions']),1)
        self.assertFalse(blocked['completions'][0]['verification']['valid'])
        self.assertEqual(blocked['completions'][0]['verification']['state'],'blocked')
        self.assertTrue(blocked['completions'][0]['recorded_verification']['valid'])
        self.assertFalse(blocked['tasks'][0]['completed'])
        self.assertFalse(blocked['tasks'][0]['claimable'])
        self.assertEqual(blocked['claims'][0]['state'],'historical_scalar_pilot_replay_blocked')
        missing_backend.symlink_to(Path(__file__).parent/'toolchains'/'lean-4.34.1-linux-x86_64',target_is_directory=True)
        recovered = blocked_conductor.snapshot()
        self.assertTrue(recovered['tasks'][0]['completed'])
        self.assertTrue(recovered['completions'][0]['verification']['fresh_replay'])
        self.assertEqual(recovered['budget']['spent'],1)
        self.assertEqual(recovered['full_theorem_certificates'],0)

    def test_same_conductor_invalidates_native_cache_on_backend_loss_and_changed_target(self):
        from polynomial_prover import prove_profile
        bundled = Path(__file__).parent/'toolchains'/'lean-4.34.1-linux-x86_64'
        selected = Path(self.temp.name)/'selected-backend'
        selected.symlink_to(bundled,target_is_directory=True)
        c = ProofConductor(self.directory,None,toolchain=selected)
        c.blueprint(['scalar-variance-bounds'])
        task = next(t for t in c.snapshot()['tasks'] if t['kind']=='scalar_pilot')
        claim = c.claim(task['task_id'],'native-cache-control',lease_seconds=300)
        receipt = prove_profile(claim['artifact_directory'],toolchain=selected)
        self.assertEqual(receipt['status'],'scoped_kernel_checked',receipt.get('reason'))
        c.complete_claim(claim['claim_id'],receipt['receipt_path'])
        self.assertTrue(c.snapshot()['completions'][0]['verification']['valid'])
        selected.unlink()
        missing = c.snapshot()
        self.assertFalse(missing['completions'][0]['verification']['valid'])
        self.assertEqual(missing['completions'][0]['verification']['state'],'blocked')
        selected.symlink_to(bundled,target_is_directory=True)
        repaired = c.snapshot()
        self.assertTrue(repaired['completions'][0]['verification']['fresh_replay'])
        alternate = Path(self.temp.name)/'different-empty-backend'
        alternate.mkdir()
        selected.unlink()
        selected.symlink_to(alternate,target_is_directory=True)
        changed = c.snapshot()
        self.assertFalse(changed['tasks'][0]['completed'])
        self.assertEqual(changed['completions'][0]['verification']['state'],'blocked')
        self.assertEqual(changed['budget']['spent'],1)

    def test_same_conductor_cache_tracks_environment_override_and_explicit_disable(self):
        from polynomial_prover import prove_profile
        selected = Path(self.temp.name)/'environment-selected'
        selected.symlink_to(Path(__file__).parent/'toolchains'/'lean-4.34.1-linux-x86_64',target_is_directory=True)
        with patch.dict(os.environ,{'UL_LEAN_TOOLCHAIN':str(selected)}):
            c = ProofConductor(self.directory,None)
            c.blueprint(['scalar-variance-bounds'])
            task = next(t for t in c.snapshot()['tasks'] if t['kind']=='scalar_pilot')
            claim = c.claim(task['task_id'],'environment-cache-control',lease_seconds=300)
            receipt = prove_profile(claim['artifact_directory'])
            self.assertEqual(receipt['status'],'scoped_kernel_checked',receipt.get('reason'))
            c.complete_claim(claim['claim_id'],receipt['receipt_path'])
            os.environ['UL_LEAN_TOOLCHAIN'] = ''
            disabled = c.snapshot()
            self.assertFalse(disabled['completions'][0]['verification']['valid'])
            self.assertEqual(disabled['completions'][0]['verification']['state'],'blocked')
            os.environ['UL_LEAN_TOOLCHAIN'] = str(Path(self.temp.name)/'missing-environment-backend')
            self.assertFalse(c.snapshot()['tasks'][0]['completed'])
            os.environ['UL_LEAN_TOOLCHAIN'] = str(selected)
            restored = c.snapshot()
            self.assertTrue(restored['completions'][0]['verification']['fresh_replay'])
            self.assertEqual(restored['budget']['spent'],1)

    def test_backend_identity_binds_all_pinned_metadata_and_native_search_shadows(self):
        import proof_conductor
        identity = getattr(proof_conductor,'native_backend_identity',None)
        self.assertIsNotNone(identity,'native cache identity helper is missing')
        root = Path(self.temp.name)/'metadata-only-backend'
        (root/'bin').mkdir(parents=True)
        (root/'lib'/'lean').mkdir(parents=True)
        for name in ('lean','leanchecker'):
            (root/'bin'/name).write_bytes(b'not executable; metadata inspection only')
        library = root/'lib'/'lean'/'Init.olean'
        library.write_bytes(b'a')
        first = identity(root)['signature']
        old = library.stat()
        library.write_bytes(b'b')
        os.utime(library,ns=(old.st_atime_ns,old.st_mtime_ns))
        second = identity(root)['signature']
        self.assertNotEqual(first,second,'ctime/inode must detect same-size preserved-mtime changes')
        shadow = root/'lib'/'lean'/'glibc-hwcaps'
        shadow.mkdir()
        (shadow/'libc.so.6').symlink_to(Path(self.temp.name)/'missing-shadow')
        self.assertNotEqual(second,identity(root)['signature'])
        with patch.dict(os.environ,{'UL_LEAN_TOOLCHAIN':''}):
            disabled = identity(None)
        self.assertIsNone(disabled['selected_path'])
        self.assertFalse(disabled['present'])

    def test_completed_profile_upgrade_preserves_unverified_history_and_unrelated_tasks(self):
        from polynomial_prover import prove_profile,profile_contract
        original_blueprint = self.plan(['scalar-variance-bounds'],budget=5)
        self.plan(['D5d-M'],budget=5)
        state = self.conductor.snapshot()
        pilot = next(t for t in state['tasks'] if t['kind']=='scalar_pilot')
        claim = self.conductor.claim(pilot['task_id'],'native-worker',lease_seconds=300)
        receipt = prove_profile(claim['artifact_directory'])
        self.assertEqual(receipt['status'],'scoped_kernel_checked',receipt.get('reason'))
        self.conductor.complete_claim(claim['claim_id'],receipt['receipt_path'])
        upgraded = copy.deepcopy(profile_contract())
        upgraded['checker_bindings']['worker_sha256'] = 'a'*64
        with patch('polynomial_prover.profile_contract',return_value=upgraded):
            old = self.conductor.snapshot()
            self.assertEqual(old['completions'][0]['verification']['state'],'stale_profile_contract')
            self.assertFalse(old['completions'][0]['verification']['valid'])
            with self.assertRaisesRegex(ValueError,'stale'):
                self.conductor.complete_claim(claim['claim_id'],receipt['receipt_path'])
            self.assertTrue(old['completions'][0]['recorded_verification']['valid'])
            self.assertTrue(any(t['claimable'] and t['kind']!='scalar_pilot' for t in old['tasks']))
            with self.assertRaisesRegex(ValueError,'budget'):
                self.conductor.blueprint(['scalar-variance-bounds'],budget=6)
            replacement_blueprint = self.conductor.blueprint(['scalar-variance-bounds'],budget=5)
            self.assertNotEqual(replacement_blueprint['blueprint_id'],original_blueprint['blueprint_id'])
            self.assertEqual(self.conductor.snapshot()['budget']['limit'],5)
            self.assertEqual(self.conductor.snapshot()['budget']['spent'],1)
            new = next(t for t in self.conductor.snapshot()['tasks'] if t['kind']=='scalar_pilot' and not t['stale'])
            self.assertNotEqual(new['task_id'],pilot['task_id'])
            fresh = self.conductor.claim(new['task_id'],'new-profile-worker')
            self.assertEqual(fresh['strategy'],'native_scalar_profile')
            self.assertEqual(self.conductor.snapshot()['budget']['remaining'],3)
        recovered = self.conductor.snapshot()
        self.assertTrue(recovered['completions'][0]['verification']['fresh_replay'])
        self.assertEqual(recovered['full_theorem_certificates'],0)

    def test_completion_rejects_nonpilot_false_receipt_outside_path_and_symlink(self):
        self.plan(['D5d-M','scalar-variance-bounds'])
        state = self.conductor.snapshot()
        nonpilot = next(t for t in state['tasks'] if t['claimable'] and t['kind']!='scalar_pilot')
        ordinary = self.conductor.claim(nonpilot['task_id'],'ordinary')
        with self.assertRaisesRegex(ValueError,'pilot'):
            self.conductor.complete_claim(ordinary['claim_id'],str(Path(self.temp.name)/'receipt.json'))
        task = next(t for t in state['tasks'] if t['kind']=='scalar_pilot')
        claim = self.conductor.claim(task['task_id'],'scalar')
        outside = Path(self.temp.name)/'receipt.json'
        outside.write_text('{"valid":true,"fresh_replay":true}')
        with self.assertRaisesRegex(ValueError,'assigned|directory|outside|path'):
            self.conductor.complete_claim(claim['claim_id'],str(outside))
        link = Path(claim['artifact_directory'])/'receipt.json'
        link.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'symlink|path|directory'):
            self.conductor.complete_claim(claim['claim_id'],str(link))
        link.unlink()
        link.write_text('{"valid":true,"fresh_replay":true}')
        with self.assertRaisesRegex(ValueError,'verification|receipt|replay'):
            self.conductor.complete_claim(claim['claim_id'],str(link))
        self.assertFalse(any(t['completed'] for t in self.conductor.snapshot()['tasks']))

    def test_completion_rejects_expired_lease_before_native_checker_work(self):
        self.plan(['scalar-variance-bounds'])
        claim = self.conductor.claim(self.ready()['task_id'],'scalar',lease_seconds=.01)
        time.sleep(.02)
        with self.assertRaisesRegex(ValueError,'expired|lease'):
            self.conductor.complete_claim(claim['claim_id'],str(Path(claim['artifact_directory'])/'receipt.json'))

    def test_offline_baseline_must_validate_before_refreshing_any_derived_metadata(self):
        self.conductor.base['records'][0]['scope'] += ' tampered without stored fingerprint update'
        self.conductor._live_cache.clear()
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            self.conductor.snapshot()

    def test_snapshot_exposes_inspectable_bounded_state_and_capabilities(self):
        self.plan()
        state = self.conductor.snapshot()
        self.assertTrue({'blueprints','tasks','claims','attempts','decisions','budget','version','ir'}.issubset(state))
        self.assertTrue(set(state['ir']['effects']).issubset(state['capabilities']['effects']))
        self.assertFalse(state['capabilities']['mathematical_approval'])
        state['tasks'][0]['scope'] = {'changed':True}
        self.assertNotEqual(state,self.conductor.snapshot())


SEPARATORS = 'A\u2028B\u0085C\u2029D'
ZERO_HASH = '0' * 64


def _pre_fix_canonical(value):
    """Historical canonical JSON bytes. ensure_ascii=False is unchanged."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf8')


def _legacy_journal_events(raw, canonical):
    """Pre-G1-F1 unload: decode the archive, then str.splitlines().

    That split treats U+0085, U+2028 and U+2029 as line boundaries, so one
    canonical record can become several fragments and fail to load.
    """
    if raw and not raw.endswith(b'\n'):
        raise ValueError('incomplete journal tail; preserve and repair explicitly before resuming')
    events = []
    prev = ZERO_HASH
    ids = {}
    for n, line in enumerate(raw.decode('utf8').splitlines(), 1):
        if not line.strip():
            raise ValueError('empty journal line ' + str(n))
        event = json.loads(line)
        if not isinstance(event, dict) or not isinstance(event.get('payload'), dict) or not isinstance(event.get('type'), str):
            raise ValueError('malformed journal event at line ' + str(n))
        if event['seq'] != len(events) + 1 or event['prev_hash'] != prev:
            raise ValueError('event sequence/previous hash corrupted at line ' + str(n))
        body = {k: v for k, v in event.items() if k != 'event_hash'}
        expected = hashlib.sha256(canonical(body)).hexdigest()
        if event.get('event_hash') != expected:
            raise ValueError('tampered event at line ' + str(n))
        eid = event.get('event_id')
        if eid:
            if eid in ids:
                raise ValueError('duplicate event ID in stored journal')
            ids[eid] = event
        events.append(event)
        prev = expected
    return events


class JournalLfFramingTests(unittest.TestCase):
    """G1-F1 reader framing. canonical() and the torn-tail guard stay as they are.

    proof_conductor.Journal is engine.conductor.Journal. It has no second
    decode-and-splitlines reader; ProofConductor(workdir) and _read() both
    call this _load.
    """

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent
        engine = root / 'engine'
        sys.path.insert(0, str(engine))
        try:
            import model  # noqa: F401
        except ImportError:
            stub = types.ModuleType('model')
            stub.load = lambda *args, **kwargs: None
            stub.validate = lambda *args, **kwargs: None
            stub.fingerprints = lambda *args, **kwargs: {}
            stub.digest = lambda *args, **kwargs: ''
            stub.GateError = type('GateError', (Exception,), {})
            sys.modules['model'] = stub
        from conductor import Journal, ZERO, canonical
        cls.Journal = Journal
        cls.ZERO = ZERO
        cls.canonical = staticmethod(canonical)
        cls.root = root
        cls.engine_source = (engine / 'conductor.py').read_text(encoding='utf8')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workdir = Path(self.temp.name) / 'journal'

    def _seal(self, seq, prev, payload, etype='note'):
        event = {'seq': seq, 'when_utc': '2026-10-25T00:00:00+00:00',
                 'type': etype, 'payload': payload, 'prev_hash': prev}
        event['event_hash'] = hashlib.sha256(self.canonical(event)).hexdigest()
        return event

    def _write(self, *events):
        self.workdir.mkdir(parents=True, exist_ok=True)
        raw = b''.join(self.canonical(event) + b'\n' for event in events)
        (self.workdir / 'events.jsonl').write_bytes(raw)
        return raw

    def test_unicode_separators_break_legacy_reader_and_load_on_lf_bytes(self):
        payload = {'note': SEPARATORS, 'plain': 'ok'}
        event = self._seal(1, self.ZERO, payload)
        raw = self._write(event)
        self.assertEqual(raw.count(b'\n'), 1)
        self.assertIn('\u2028'.encode('utf8'), raw)
        self.assertIn('\u0085'.encode('utf8'), raw)
        self.assertIn('\u2029'.encode('utf8'), raw)
        self.assertNotIn(b'\\u2028', raw)
        self.assertNotIn(b'\\u0085', raw)
        self.assertNotIn(b'\\u2029', raw)
        legacy_lines = raw.decode('utf8').splitlines()
        self.assertEqual(len(legacy_lines), 4)
        with self.assertRaises(json.JSONDecodeError):
            _legacy_journal_events(raw, self.canonical)
        journal = self.Journal(self.workdir)
        self.assertEqual(journal.path.read_bytes(), raw)
        journal._load()
        self.assertEqual(journal.path.read_bytes(), raw)
        self.assertEqual(len(journal.events), 1)
        loaded = journal.events[0]
        self.assertEqual(loaded['payload'], payload)
        self.assertEqual(loaded['event_hash'], event['event_hash'])
        body = {k: v for k, v in loaded.items() if k != 'event_hash'}
        self.assertEqual(self.canonical(body), _pre_fix_canonical(body))
        self.assertEqual(loaded['event_hash'], hashlib.sha256(_pre_fix_canonical(body)).hexdigest())

    def test_ascii_lf_records_match_legacy_reader_and_keep_hashes(self):
        first = self._seal(1, self.ZERO, {'msg': 'plain-ascii'})
        second = self._seal(2, first['event_hash'], {'msg': 'second line'})
        raw = self._write(first, second)
        self.assertEqual(raw.decode('utf8').splitlines(), [record.decode('utf8') for record in raw.split(b'\n') if record != b''])
        legacy = _legacy_journal_events(raw, self.canonical)
        journal = self.Journal(self.workdir)
        self.assertEqual(journal.path.read_bytes(), raw)
        self.assertEqual([event['event_hash'] for event in journal.events],
                         [event['event_hash'] for event in legacy])
        self.assertEqual([event['payload'] for event in journal.events],
                         [{'msg': 'plain-ascii'}, {'msg': 'second line'}])
        for event in journal.events:
            body = {k: v for k, v in event.items() if k != 'event_hash'}
            self.assertEqual(self.canonical(body), _pre_fix_canonical(body))
            self.assertEqual(event['event_hash'], hashlib.sha256(_pre_fix_canonical(body)).hexdigest())

    def test_append_round_trip_preserves_canonical_archive_bytes(self):
        journal = self.Journal(self.workdir)
        first = journal.append('ascii', {'msg': 'plain'})
        second = journal.append('unicode', {'msg': SEPARATORS})
        raw = journal.path.read_bytes()
        self.assertEqual(raw, self.canonical(first) + b'\n' + self.canonical(second) + b'\n')
        self.assertTrue(raw.startswith(self.canonical(first) + b'\n'))
        reloaded = self.Journal(self.workdir)
        self.assertEqual(reloaded.path.read_bytes(), raw)
        self.assertEqual([event['event_hash'] for event in reloaded.events],
                         [first['event_hash'], second['event_hash']])
        self.assertEqual(reloaded.events[1]['payload']['msg'], SEPARATORS)
        for event in (first, second):
            body = {k: v for k, v in event.items() if k != 'event_hash'}
            self.assertEqual(self.canonical(body), _pre_fix_canonical(body))
            self.assertEqual(event['event_hash'], hashlib.sha256(_pre_fix_canonical(body)).hexdigest())
        with self.assertRaises(json.JSONDecodeError):
            _legacy_journal_events(raw, self.canonical)

    def test_torn_tail_guard_still_rejects_without_rewriting_bytes(self):
        event = self._seal(1, self.ZERO, {'msg': 'kept'})
        journal = self.Journal(self.workdir)
        blob = self.canonical(event) + b'\n' + b'{"seq":2'
        journal.path.write_bytes(blob)
        with self.assertRaisesRegex(ValueError, 'incomplete journal tail'):
            journal._load()
        self.assertEqual(journal.events, [])
        self.assertEqual(journal.path.read_bytes(), blob)
        journal.path.write_bytes(b'\n')
        with self.assertRaisesRegex(ValueError, 'empty journal line 1'):
            journal._load()
        self.assertEqual(journal.path.read_bytes(), b'\n')

    def test_canonical_and_transport_serialization_stay_ensure_ascii_false(self):
        source = inspect.getsource(self.canonical)
        self.assertIn('ensure_ascii=False', source)
        load_source = inspect.getsource(self.Journal._load)
        self.assertNotIn(".decode('utf8').splitlines()", load_source)
        self.assertIn("records=raw.split(b'\\n')", load_source)
        self.assertEqual(self.engine_source.count('ensure_ascii=False'), 4)
        self.assertNotIn(".decode('utf8').splitlines()", self.engine_source)
        sample = {'note': SEPARATORS}
        self.assertEqual(self.canonical(sample), _pre_fix_canonical(sample))

    def test_proof_conductor_has_no_separate_splitlines_reader(self):
        text = (self.root / 'proof_conductor.py').read_text(encoding='utf8')
        self.assertNotIn('splitlines', text)
        self.assertIn('self.journal = Journal(workdir)', text)
        self.assertIn('self.journal._load()', text)


if __name__ == '__main__':
    unittest.main()
