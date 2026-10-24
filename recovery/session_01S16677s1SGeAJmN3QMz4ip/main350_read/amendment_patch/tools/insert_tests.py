import sys
from pathlib import Path
pop = Path(sys.argv[1])

# ---- (a) verifier root-completeness replay: test_global_h0_selector.py ----
sel = pop / 'test_global_h0_selector.py'
text = sel.read_text()
anchor_a = "    def test_changed_original_polynomial_sign_witness_is_rejected(self):\n"
test_a = '''    def test_omitted_bounding_roots_are_rejected_by_the_completeness_replay_alone(self):
        # Break caught: the verifier trusts the supplied regular-level root
        # list. The younger component's two bounding quartic roots are omitted
        # and every dependent record (components, sign witnesses, incidence,
        # events, finite bars, projection) is rewritten consistently, so the
        # Sturm root-count replay is the only check that can still object.
        u, v, certificate, _ = self.hostile()
        bad = copy.deepcopy(certificate)
        index, level = next((i, l) for i, l in enumerate(bad["levels"]) if len(l["components"]) == 2)
        older, younger = level["components"]
        rootset = level["root_certificate"]
        rootset["roots"] = [r for r in rootset["roots"] if r["ordinal"] not in younger["boundary_root_ordinals"]]
        for ordinal, root in enumerate(rootset["roots"]):
            root["ordinal"] = ordinal
        level["components"] = [older]
        level["sign_intervals"] = [
            {"boundary_root_ordinals": [i - 1 if i else None, i if i < len(rootset["roots"]) else None],
             "witness": (lo + hi) / 2, "value": evaluate(rootset["polynomial"], (lo + hi) / 2)}
            for i, lo, hi in selector._component_intervals(rootset)]
        bad["incidence"] = [e for e in bad["incidence"] if younger["id"] not in (e["upper_component"], e["lower_component"])]
        for batch in bad["events"]:
            batch["births"] = [b for b in batch["births"] if b["component"] != younger["id"]]
            for merge in batch["merges"]:
                batch["continuations"].append({"upper_component": older["id"],
                    "lower_component": merge["lower_component"], "survivor_birth": merge["survivor_birth"]})
            batch["merges"] = []
        bad["actual_barcode"]["finite"] = []
        bad["population_projection"].update(count=0, actual_finite_count=0)
        result = selector.verify_h0(u, v, bad, budget=self.budget())
        self.assertEqual(result["status"], "FAIL_IMPLEMENTATION")
        name = "verify/level%d/complete0" % index
        self.assertIs(result["checks"][name], False)
        self.assertEqual([k for k, value in result["checks"].items() if value is not True], [name, "rejected/" + name])

'''
assert text.count(anchor_a) == 1, text.count(anchor_a)
assert 'test_omitted_bounding_roots' not in text
sel.write_text(text.replace(anchor_a, test_a + anchor_a))
print('inserted test (a) into', sel.name)

# ---- (b) driver core_identity_overlap gate: test_global_h0_runner.py ----
run = pop / 'test_global_h0_runner.py'
text = run.read_text()
assert "if fault == 'height_disagree':" in text, 'fault branch missing'
anchor_b = "    def test_main_missing_premise_leaves_scientific_operations_not_run(self):\n"
test_b = '''    def test_main_root_box_height_disagreement_fails_the_core_identity_gate_before_expectations(self):
        # Break caught: the root-box field-versus-algebra overlap check is
        # trivially true, so a critical height the field contradicts still
        # certifies; every other replay flag, field check and gate passes.
        result, emitted, events, _ = self.run_synthetic(fault='height_disagree')
        certificate = result['certificate']
        row = certificate['fixtures'][0]
        self.assertEqual(certificate['status'], 'FAIL_IMPLEMENTATION')
        self.assertEqual(certificate['failure']['phase'], 'fixture/TEST_ONLY')
        self.assertEqual(certificate['failure']['detail'], 'Required soundness gate failed')
        self.assertEqual(set(nested_statuses(certificate)), {'FAIL_IMPLEMENTATION'})
        self.assertNotIn(('expected', 'TEST_ONLY'), events)
        self.assertEqual(row['comparison']['computed_status_before_withdrawal'], 'NOT_COMPARED')
        box = next(e for e in row['field_evaluations'] if e['route'] == 'root_box')
        self.assertEqual(box['returned_interval'], (Q(4, 9), Q(4, 9)))
        self.assertEqual(box['algebraic_field_interval'], (Q(13, 9), Q(13, 9)))
        self.assertEqual({name: check['computed_value'] for name, check in row['field_checks']['checks'].items()},
            {'core_identity_overlap': False, 'anchor_seed_values': True, 'periodic_translation': True})
        self.assertEqual(row['field_checks']['computed_status_before_withdrawal'], 'FAIL')
        self.assertEqual({name: gate['computed_status_before_withdrawal'] for name, gate in row['soundness'].items()},
            {name: 'FAIL' if name == 'core_identity_overlap' else 'PASS' for name in SOUNDNESS_GATES})
        self.assertEqual(certificate['work']['field_calls'], 33)
        self.assertIsNotNone(emitted)

'''
assert text.count(anchor_b) == 1, text.count(anchor_b)
assert 'test_main_root_box_height_disagreement' not in text
run.write_text(text.replace(anchor_b, test_b + anchor_b))
print('inserted test (b) into', run.name)
