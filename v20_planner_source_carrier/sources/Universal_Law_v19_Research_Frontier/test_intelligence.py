"""Behavioral director checks: real journals, leases, rational arithmetic."""
import concurrent.futures
import json
import multiprocessing
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

try:
    from intelligence import ResearchDirector
except ImportError:
    ResearchDirector = None


def process_claim(directory, task_id, agent, results):
    try:
        results.put(ResearchDirector(directory).claim(task_id, agent))
    except ValueError as exc:
        results.put(str(exc))


class DirectorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ResearchDirector, "durable research director is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.director = ResearchDirector(self.temp.name)

    def example(self, case="nonnegative"):
        return self.director.create_example_task(case)

    def submit(self, task, claim, outcome="inconclusive", **kwargs):
        return self.director.record_attempt(task["task_id"], claim["agent_id"],
            claim["strategy"], outcome, "sample_coverage", claim_id=claim["claim_id"],
            input_fingerprint=claim["input_fingerprint"], **kwargs)

    def test_seed_binds_all_records_and_target_dependency_priority(self):
        self.director.seed_plan(["C6-01"])
        state = self.director.snapshot()
        tasks = [t for t in state["tasks"] if t["kind"] == "canonical"]
        self.assertEqual(len(tasks), 104)
        task = next(t for t in tasks if t["record_id"] == "C6-01")
        self.assertIn("The same cap has a uniformly finite", task["intended_question"])
        self.assertEqual(task["scope"]["text"], "Fixed d and parameter regime of C6-F; some a>0.")
        self.assertEqual(len(task["dependencies"]), 1)
        priorities = {t["task_id"]: t["priority"] for t in tasks}
        self.assertGreater(priorities[task["dependencies"][0]], task["priority"])
        self.director.seed_plan(["C6-01"])
        self.assertEqual(len(self.director.snapshot()["tasks"]), 104)

    def test_only_one_process_can_claim_and_same_owner_retry_is_idempotent(self):
        task = self.example()
        queue = multiprocessing.Queue()
        workers = [multiprocessing.Process(target=process_claim,
            args=(self.temp.name, task["task_id"], "agent-" + str(i), queue)) for i in range(4)]
        for worker in workers: worker.start()
        for worker in workers: worker.join(10)
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        results = [queue.get(timeout=2) for _ in workers]
        won = [r for r in results if isinstance(r, dict)]
        self.assertEqual(len(won), 1)
        duplicate = self.director.claim(task["task_id"], won[0]["agent_id"])
        self.assertEqual(duplicate["claim_id"], won[0]["claim_id"])

    def test_expired_claim_survives_restart_but_cannot_submit(self):
        task = self.example()
        old = self.director.claim(task["task_id"], "first", lease_seconds=.02)
        time.sleep(.03)
        restarted = ResearchDirector(self.temp.name)
        new = restarted.claim(task["task_id"], "second")
        self.assertNotEqual(old["claim_id"], new["claim_id"])
        with self.assertRaisesRegex(ValueError, "lease|claim"):
            self.submit(task, old)
        self.assertEqual(len(restarted.snapshot()["claims"]), 2)

    def test_scope_and_fingerprint_and_claim_token_are_required(self):
        task = self.example()
        claim = self.director.claim(task["task_id"], "worker")
        with self.assertRaisesRegex(ValueError, "scope"):
            self.submit(task, claim, scope={"domain": "reals"})
        bad = dict(claim, input_fingerprint="0" * 64)
        with self.assertRaisesRegex(ValueError, "stale|fingerprint"):
            self.submit(task, bad)
        with self.assertRaisesRegex(ValueError, "claim"):
            self.director.record_attempt(task["task_id"], "worker", claim["strategy"],
                "inconclusive", "sample_coverage")

    def test_canonical_working_branch_change_invalidates_claim(self):
        from conductor import Coordinator
        coordinator = Coordinator(Path(self.temp.name) / "engine")
        director = ResearchDirector(Path(self.temp.name) / "director", coordinator)
        director.seed_plan()
        task = next(t for t in director.snapshot()["tasks"] if not t["dependencies"])
        claim = director.claim(task["task_id"], "worker")
        record = next(r for r in coordinator.snapshot()["records"] if r["id"] == task["record_id"])
        proposal = coordinator.propose("statement", record["id"], record["statement"] + " Reviewed working change.")
        coordinator.decide(proposal["proposal_id"], True)
        with self.assertRaisesRegex(ValueError, "stale"):
            director.record_attempt(task["task_id"], "worker", claim["strategy"], "failed",
                "missing_evidence", claim_id=claim["claim_id"], input_fingerprint=claim["input_fingerprint"])
        self.assertTrue(next(t for t in director.snapshot()["tasks"] if t["task_id"] == task["task_id"])["stale"])

    def test_owner_cannot_resolve_conflict_against_changed_canonical_inputs(self):
        from conductor import Coordinator
        coordinator = Coordinator(Path(self.temp.name) / "engine")
        director = ResearchDirector(Path(self.temp.name) / "director", coordinator)
        director.seed_plan()
        task = next(t for t in director.snapshot()["tasks"] if not t["dependencies"])
        first = director.add_assertion(task["task_id"], "source:a", "P", True, task["scope"])
        director.add_assertion(task["task_id"], "source:b", "P", False, task["scope"])
        conflict = director.snapshot()["conflicts"][0]
        record = next(r for r in coordinator.snapshot()["records"] if r["id"] == task["record_id"])
        proposal = coordinator.propose("statement", record["id"], record["statement"] + " New reviewed working input.")
        coordinator.decide(proposal["proposal_id"], True)
        with self.assertRaisesRegex(ValueError, "stale"):
            director.resolve_conflict(conflict["conflict_id"], first["assertion_id"], "owner:local", "Compare P", ["a:1"])

    def test_restaged_opposition_uses_current_input_without_poisoning_historical_conflict(self):
        from conductor import Coordinator
        for resolve_old in (False, True):
            with self.subTest(historical_conflict_resolved=resolve_old):
                root = Path(self.temp.name) / str(resolve_old)
                coordinator = Coordinator(root / "engine")
                director = ResearchDirector(root / "director", coordinator)
                director.seed_plan()
                task = next(t for t in director.snapshot()["tasks"] if t.get("record_id") == "C6-F")
                first = director.add_assertion(task["task_id"], "source:a", "P", True, task["scope"])
                director.add_assertion(task["task_id"], "source:b", "P", False, task["scope"])
                old = director.snapshot()["conflicts"][0]
                if resolve_old:
                    director.resolve_conflict(old["conflict_id"], first["assertion_id"], "owner:local", "Old input comparison", ["a:old"])
                record = next(r for r in coordinator.snapshot()["records"] if r["id"] == "C6-F")
                proposal = coordinator.propose("statement", "C6-F", record["statement"] + " Reviewed input revision.")
                coordinator.decide(proposal["proposal_id"], True)
                director.seed_plan()
                current = next(t for t in director.snapshot()["tasks"] if t.get("record_id") == "C6-F" and not t["stale"])
                self.assertEqual(current["scope"], task["scope"])
                accepted = director.add_assertion(current["task_id"], "source:a", "P", True, current["scope"])
                director.add_assertion(current["task_id"], "source:b", "P", False, current["scope"])
                conflicts = director.snapshot()["conflicts"]
                self.assertEqual(len(conflicts), 2, "historical inputs must not merge into the current conflict")
                new = next(c for c in conflicts if accepted["assertion_id"] in c["assertion_ids"])
                self.assertEqual(new["input_fingerprint"], current["input_fingerprint"])
                self.assertEqual(len(new["assertion_ids"]), 2)
                self.assertNotEqual(new["conflict_id"], old["conflict_id"])
                director.resolve_conflict(new["conflict_id"], accepted["assertion_id"], "owner:local", "Current input comparison", ["a:current"])
                replay = ResearchDirector(root / "director", coordinator).snapshot()
                self.assertEqual(len(replay["assertions"]), 4)
                historical = next(c for c in replay["conflicts"] if c["conflict_id"] == old["conflict_id"])
                self.assertEqual(historical["assertion_ids"], old["assertion_ids"])
                self.assertEqual(historical["status"], "resolved_by_owner" if resolve_old else "contested")
                self.assertEqual(next(c for c in replay["conflicts"] if c["conflict_id"] == new["conflict_id"])["status"], "resolved_by_owner")

    def test_legacy_conflict_ids_replay_without_rewriting_history(self):
        from conductor import Journal
        task = self.example()
        first = self.director.add_assertion(task["task_id"], "a", "P", True, task["scope"])
        self.director.add_assertion(task["task_id"], "b", "P", False, task["scope"])
        legacy = Journal(Path(self.temp.name) / "legacy")
        legacy_id = "conflict:legacy-stored-id"
        for event in self.director.journal.events:
            payload = json.loads(json.dumps(event["payload"]))
            if event["type"] == "ASSERTION_ADDED" and payload.get("conflict"):
                payload["conflict"]["conflict_id"] = legacy_id
                payload["conflict"].pop("input_fingerprint", None)
                payload["investigation"]["conflict_id"] = legacy_id
            legacy.append(event["type"], payload)
        director = ResearchDirector(legacy.workdir)
        before = legacy.path.read_bytes()
        director.resolve_conflict(legacy_id, first["assertion_id"], "owner:test", "Legacy comparison", ["legacy:a"])
        director.add_assertion(task["task_id"], "c", "P", False, task["scope"])
        replay = ResearchDirector(legacy.workdir).snapshot()
        self.assertTrue(legacy.path.read_bytes().startswith(before))
        self.assertEqual(len(replay["conflicts"]), 2)
        self.assertTrue(any(c["conflict_id"] == legacy_id and c["status"] == "resolved_by_owner" for c in replay["conflicts"]))
        self.assertEqual(len(replay["assertions"]), 3)

    def test_assertions_cannot_bind_investigation_tasks_and_poison_original_resolution(self):
        task = self.example()
        first = self.director.add_assertion(task["task_id"], "a", "P", True, task["scope"])
        self.director.add_assertion(task["task_id"], "b", "P", False, task["scope"])
        before = self.director.snapshot()
        conflict = before["conflicts"][0]
        with self.assertRaisesRegex(ValueError, "investigation|original"):
            self.director.add_assertion(conflict["investigation_task_id"], "c", "P", True, task["scope"])
        unchanged = self.director.snapshot()
        self.assertEqual(len(unchanged["assertions"]), 2)
        self.assertEqual(unchanged["conflicts"], before["conflicts"])
        self.director.resolve_conflict(conflict["conflict_id"], first["assertion_id"], "owner:local", "Compare original evidence", ["a:1"])
        self.assertEqual(self.director.snapshot()["conflicts"][0]["status"], "resolved_by_owner")

    def test_engine_change_cannot_race_checked_result_publication(self):
        from conductor import Coordinator
        coordinator = Coordinator(Path(self.temp.name) / "engine")
        director = ResearchDirector(Path(self.temp.name) / "director", coordinator)
        director.seed_plan()
        task = next(t for t in director.snapshot()["tasks"] if not t["dependencies"])
        claim = director.claim(task["task_id"], "worker")
        record = next(r for r in coordinator.snapshot()["records"] if r["id"] == task["record_id"])
        proposal = coordinator.propose("statement", record["id"], record["statement"] + " Concurrent working edit.")
        checking, resume, changed, writer_entered = threading.Event(), threading.Event(), threading.Event(), threading.Event()
        failures = []
        actual_execute = director._execute
        # Timing seam only: run the real method after coordinating the real writer.
        def pause_real_execution(task, strategy):
            checking.set()
            if not resume.wait(3): raise AssertionError("execution barrier timeout")
            return actual_execute(task, strategy)
        def publish():
            try:
                director.record_attempt(task["task_id"], "worker", claim["strategy"], "blocked", "missing_evidence",
                    claim_id=claim["claim_id"], input_fingerprint=claim["input_fingerprint"])
            except Exception as exc: failures.append(exc)
        def change():
            try:
                with coordinator._mutex, coordinator.journal.transaction():
                    writer_entered.set()
                    coordinator.decide(proposal["proposal_id"], True)
                    changed.set()
            except Exception as exc: failures.append(exc)
        with patch.object(director, "_execute", side_effect=pause_real_execution):
            publisher = threading.Thread(target=publish)
            writer = threading.Thread(target=change)
            publisher.start()
            self.assertTrue(checking.wait(3))
            writer.start()
            try:
                self.assertFalse(writer_entered.wait(.1), "engine writer entered between checking inputs and appending the result")
            finally:
                resume.set()
                publisher.join(5)
                writer.join(5)
        self.assertEqual(failures, [])
        self.assertTrue(changed.is_set())

    def test_inconclusive_sampling_changes_method_and_exact_recomputation(self):
        task = self.example()
        first = self.director.claim(task["task_id"], "sampler")
        self.assertEqual(first["strategy"], "rational_sampling")
        attempt = self.submit(task, first)
        self.assertEqual(attempt["outcome"], "inconclusive")
        self.assertEqual(attempt["verification"]["values"], ["1", "1"])
        second = self.director.claim(task["task_id"], "algebra")
        self.assertEqual(second["strategy"], "exact_square_completion")
        result = self.submit(task, second, "verified", artifacts={"pretended_minimum": "99"})
        self.assertEqual(result["verification"]["minimum"], "0")
        self.assertEqual(result["verification"]["vertex"], "1")
        self.assertFalse(result["verification"]["full_theorem_certificate"])

    def test_false_inequality_has_recomputed_counterexample(self):
        task = self.example("counterexample")
        sampling = self.director.claim(task["task_id"], "sampler", "rational_sampling")
        self.submit(task, sampling)
        claim = self.director.claim(task["task_id"], "algebra", "exact_square_completion")
        attempt = self.submit(task, claim, "counterexample")
        self.assertEqual(attempt["verification"]["counterexample"], {"x": "1", "value": "-1"})

    def test_fake_success_and_repeating_structurally_failed_method_rejected(self):
        task = self.example()
        claim = self.director.claim(task["task_id"], "worker")
        with self.assertRaisesRegex(ValueError, "recomput|outcome|verified"):
            self.submit(task, claim, "verified", artifacts={"success": True})
        self.submit(task, claim)
        with self.assertRaisesRegex(ValueError, "different|failed|strategy"):
            self.director.claim(task["task_id"], "worker", "rational_sampling")

    def test_strategy_history_changes_contextual_ranking_after_restart(self):
        first = self.example()
        self.submit(first, self.director.claim(first["task_id"], "sampler"))
        self.submit(first, self.director.claim(first["task_id"], "algebra"), "verified")
        restarted = ResearchDirector(self.temp.name)
        second = restarted.create_example_task("counterexample")
        self.assertEqual(restarted.claim(second["task_id"], "new")["strategy"], "exact_square_completion")
        self.assertEqual(restarted.snapshot()["policy"]["learning"], "journal_history_ranking_not_model_training")

    def test_opposed_assertions_are_preserved_contested_and_require_owner_evidence(self):
        task = self.example()
        first = self.director.add_assertion(task["task_id"], "source-A", "p(x)>=0", True, task["scope"], ["A:1"])
        second = self.director.add_assertion(task["task_id"], "source-B", "p(x)>=0", False, task["scope"], ["B:2"])
        state = self.director.snapshot()
        self.assertEqual(len(state["assertions"]), 2)
        self.assertTrue(all(a["status"] == "contested" for a in state["assertions"]))
        self.assertFalse(any(a["established"] for a in state["assertions"]))
        conflict = state["conflicts"][0]
        self.assertTrue(any(t["kind"] == "conflict_investigation" for t in state["tasks"]))
        for actor, evidence in [("agent:bot", ["A:1"]), ("owner:local", [])]:
            with self.assertRaises(ValueError):
                self.director.resolve_conflict(conflict["conflict_id"], first["assertion_id"], actor, "Compare exact domains.", evidence)
        self.director.resolve_conflict(conflict["conflict_id"], first["assertion_id"], "owner:local", "Exact same scope; source A has direct arithmetic.", ["A:1"])
        state = ResearchDirector(self.temp.name).snapshot()
        self.assertEqual(state["conflicts"][0]["status"], "resolved_by_owner")
        self.assertEqual(len(state["assertions"]), 2)
        self.assertFalse(any(a["established"] for a in state["assertions"]))
        self.assertEqual(second["value"], False)

    def test_distinct_propositions_do_not_conflict_and_scope_drift_is_proposal(self):
        task = self.example()
        self.director.add_assertion(task["task_id"], "a", "P", True, task["scope"])
        self.director.add_assertion(task["task_id"], "b", "Q", False, task["scope"])
        self.assertEqual(self.director.snapshot()["conflicts"], [])
        with self.assertRaisesRegex(ValueError, "scope"):
            self.director.add_assertion(task["task_id"], "c", "P", False, {"domain": "integers"})
        proposal = self.director.propose_scope(task["task_id"], {"domain": "integers"}, "agent", "Try a smaller question")
        self.assertEqual(proposal["status"], "awaiting_owner_review")
        self.assertEqual(self.director.snapshot()["tasks"][0]["scope"], task["scope"])

    def test_new_conflict_members_invalidate_old_investigation_lease(self):
        task = self.example()
        self.director.add_assertion(task["task_id"], "a", "P", True, task["scope"])
        self.director.add_assertion(task["task_id"], "b", "P", False, task["scope"])
        conflict = self.director.snapshot()["conflicts"][0]
        investigation = next(t for t in self.director.snapshot()["tasks"] if t["task_id"] == conflict["investigation_task_id"])
        claim = self.director.claim(investigation["task_id"], "investigator")
        self.director.add_assertion(task["task_id"], "c", "P", True, task["scope"])
        with self.assertRaisesRegex(ValueError, "stale|superseded"):
            self.submit(investigation, claim, "inconclusive")
        with self.assertRaisesRegex(ValueError, "stale|superseded"):
            self.director.claim(investigation["task_id"], "someone")

    def test_owner_resolution_invalidates_investigation_lease(self):
        task = self.example()
        first = self.director.add_assertion(task["task_id"], "a", "P", True, task["scope"])
        self.director.add_assertion(task["task_id"], "b", "P", False, task["scope"])
        conflict = self.director.snapshot()["conflicts"][0]
        investigation = next(t for t in self.director.snapshot()["tasks"] if t["task_id"] == conflict["investigation_task_id"])
        claim = self.director.claim(investigation["task_id"], "investigator")
        self.director.resolve_conflict(conflict["conflict_id"], first["assertion_id"], "owner:test", "Compared exact evidence", ["a:1"])
        with self.assertRaisesRegex(ValueError, "stale|resolved"):
            self.submit(investigation, claim)

    def test_canonical_notes_cannot_fake_mathematical_success(self):
        self.director.seed_plan()
        task = self.director.snapshot()["tasks"][0]
        claim = self.director.claim(task["task_id"], "worker")
        with self.assertRaisesRegex(ValueError, "unverified"):
            self.submit(task, claim, "verified", artifacts={"certificate": "forged", "code": "raise Exception()"})
        result = self.submit(task, claim, "blocked")
        self.assertFalse(result["verification"]["recomputed"])
        self.assertFalse(result["established"])

    def test_registered_milestones_prioritize_actual_missing_work_without_new_records(self):
        self.director.seed_milestones()
        state = self.director.snapshot()
        self.assertEqual(len([t for t in state["tasks"] if t["kind"] == "canonical"]), 104)
        milestones = [t for t in state["tasks"] if t["kind"] == "project_milestone"]
        self.assertEqual(len(milestones), 3)
        top = next(t for t in milestones if t["milestone_id"] == "d5-second-factorial-upper")
        self.assertIn("D5d-M", top["bound_record_ids"])
        self.assertIn("lower bound only", top["priority_reason"])
        self.assertEqual(state["recommendations"][0]["task_id"], top["task_id"])
        self.assertEqual(state["recommendations"][0]["intended_question"], top["intended_question"])
        self.assertEqual(len(state["project_targets"]), 3)
        self.director.seed_milestones()
        self.assertEqual(len(self.director.snapshot()["tasks"]), 107)
        claim = self.director.claim(top["task_id"], "worker")
        with self.assertRaisesRegex(ValueError, "scope"):
            self.submit(top, claim, "blocked", scope={"dimension": "d=2 only"})
        with self.assertRaisesRegex(ValueError, "unverified"):
            self.submit(top, claim, "verified", artifacts={"claimed_upper_bound": True})

    def test_milestone_canonical_input_change_rejects_old_lease_and_versions_question(self):
        from conductor import Coordinator
        coordinator = Coordinator(Path(self.temp.name) / "engine")
        director = ResearchDirector(Path(self.temp.name) / "director", coordinator)
        director.seed_milestones()
        task = next(t for t in director.snapshot()["tasks"] if t.get("milestone_id") == "d5-second-factorial-upper")
        claim = director.claim(task["task_id"], "worker")
        record = next(r for r in coordinator.snapshot()["records"] if r["id"] == "D5d-M")
        proposal = coordinator.propose("statement", "D5d-M", record["statement"] + " Reviewed input changed.")
        coordinator.decide(proposal["proposal_id"], True)
        with self.assertRaisesRegex(ValueError, "stale"):
            director.record_attempt(task["task_id"], "worker", claim["strategy"], "blocked", "missing_evidence",
                claim_id=claim["claim_id"], input_fingerprint=claim["input_fingerprint"])
        director.seed_milestones()
        versions = [t for t in director.snapshot()["tasks"] if t.get("milestone_id") == "d5-second-factorial-upper"]
        self.assertEqual(len(versions), 2)
        self.assertEqual(sum(t["stale"] for t in versions), 1)
        self.assertEqual({t["intended_question"] for t in versions}, {task["intended_question"]})

    def test_bad_inputs_and_tampered_journal_fail_closed(self):
        task = self.example()
        with self.assertRaises(ValueError): self.director.claim(task["task_id"], "x", "shell:rm")
        with self.assertRaises(ValueError): self.director.claim(task["task_id"], "x", lease_seconds=float("nan"))
        with self.assertRaises(ValueError): self.director.claim(task["task_id"], "x", strategy=[])
        with self.assertRaises(ValueError): self.director.record_attempt(task["task_id"], "x", "rational_sampling", [], "none")
        with self.assertRaises(ValueError): self.director.resolve_conflict([], [], "owner:x", "Reason", ["ref"])
        with self.assertRaises(ValueError): self.director.add_assertion(task["task_id"], "x", "P" * 20001, True, task["scope"])
        path = self.director.journal.path
        with path.open("ab") as stream: stream.write(b'{"partial":')
        with self.assertRaises(ValueError): ResearchDirector(self.temp.name)


if __name__ == "__main__":
    unittest.main()
