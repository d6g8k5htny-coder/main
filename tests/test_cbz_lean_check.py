"""Adversarial protocol tests; actual Lean controls execute in cbz_lean_check."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'tools/cbz_lean_check.py'


class CensusProtocolTests(unittest.TestCase):
    def gate(self):
        self.assertTrue(MODULE.is_file(), 'portable Lean gate has not been implemented')
        spec = importlib.util.spec_from_file_location('cbz_lean_check', MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def row(self, **changes):
        row = dict(name='Demo.honest', module='Demo', kind='theorem', axioms=[],
                   statement='True', prop_premises=[])
        row.update(changes)
        return row

    def log(self, rows=None, count=None, modules=None):
        rows = [self.row()] if rows is None else rows
        return '\n'.join(['CBZ_RECORD ' + json.dumps(row) for row in rows] +
                         ['CBZ_END ' + json.dumps(dict(count=len(rows) if count is None else count,
                                                       modules=['Demo'] if modules is None else modules))])

    def test_audit_generator_refuses_missing_or_duplicate_boundary(self):
        gate = self.gate()
        self.assertTrue(hasattr(gate, 'audit_program'), 'self-audit program generator is not implemented')
        for source in ['', gate.MODULES_LINE + '\n' + gate.MODULES_LINE]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                gate.audit_program(source, 'WitnessModule')

    def test_audit_generator_refuses_module_injection(self):
        gate = self.gate()
        self.assertTrue(hasattr(gate, 'audit_program'), 'self-audit program generator is not implemented')
        with self.assertRaises(ValueError):
            gate.audit_program('open Lean Meta Elab Command\n' + gate.MODULES_LINE, 'Bad\naxiom injected : False')

    def test_all_same_named_true_witnesses_rejected(self):
        gate = self.gate()
        rows = [self.row(name=name, statement='True') for name in gate.WITNESSES]
        with self.assertRaises(ValueError):
            gate.validate_census(self.log(rows), set(gate.WITNESSES), {'Demo'})

    def test_each_weakened_witness_type_rejected(self):
        gate = self.gate()
        for name in gate.WITNESSES:
            with self.subTest(name=name), self.assertRaises(ValueError):
                gate.validate_census(self.log([self.row(name=name, statement='True')]), {name}, {'Demo'})

    def test_exact_witness_types_accepted(self):
        gate = self.gate()
        self.assertTrue(hasattr(gate, 'WITNESS_TYPES'), 'exact witness type contract is missing')
        rows = [self.row(name=name, statement=gate.WITNESS_TYPES[name]) for name in gate.WITNESSES]
        result = gate.validate_census(self.log(rows), set(gate.WITNESSES), {'Demo'})
        self.assertEqual(len(result['proofs']), 4)

    def test_partial_membership_witness_rejected(self):
        gate = self.gate()
        name = 'CBZNonvacuity.pairing_all_seven'
        with self.assertRaises(ValueError):
            gate.validate_census(self.log([self.row(name=name, statement='0 ∈ [0, 1, 2, 3, 4, 5, 6]')]), {name}, {'Demo'})

    def test_witness_contract_import_injection_rejected(self):
        gate = self.gate()
        self.assertTrue(hasattr(gate, 'witness_contract_program'), 'Lean witness type contract generator is missing')
        with self.assertRaises(ValueError):
            gate.witness_contract_program('Wrong\naxiom hidden : False')

    def test_honest_theorem_accepted(self):
        result = self.gate().validate_census(self.log(), {'Demo.honest'}, {'Demo'})
        self.assertEqual([r['name'] for r in result['proofs']], ['Demo.honest'])

    def test_definition_not_promoted_to_proof(self):
        rows = [self.row(), self.row(name='Demo.statement', kind='definition', statement='Prop')]
        result = self.gate().validate_census(self.log(rows), {'Demo.honest'}, {'Demo'})
        self.assertEqual(len(result['proofs']), 1)
        with self.assertRaises(ValueError):
            self.gate().validate_census(self.log(rows), {'Demo.honest', 'Demo.statement'}, {'Demo'})

    def test_unused_definition_with_custom_axiom_rejected(self):
        rows = [self.row(), self.row(name='Demo.data', kind='definition', axioms=['Hidden.ax'])]
        with self.assertRaises(ValueError):
            self.gate().validate_census(self.log(rows), {'Demo.honest'}, {'Demo'})

    def test_project_axiom_declaration_rejected(self):
        rows = [self.row(), self.row(name='Demo.hidden', kind='axiom')]
        with self.assertRaises(ValueError):
            self.gate().validate_census(self.log(rows), {'Demo.honest'}, {'Demo'})

    def test_false_premise_not_promoted(self):
        row = self.row(prop_premises=[dict(name='h', type='False')])
        with self.assertRaises(ValueError):
            self.gate().validate_census(self.log([row]), {'Demo.honest'}, {'Demo'})

    def test_changed_known_finite_premise_rejected(self):
        row = self.row(name='UniversalLaw.Side24.exponent_bounds',
                       prop_premises=[dict(name='h', type='False')])
        with self.assertRaises(ValueError):
            self.gate().validate_census(self.log([row]), {row['name']}, {'Demo'})

    def test_known_finite_premise_retained(self):
        row = self.row(name='UniversalLaw.Side24.exponent_bounds',
                       prop_premises=[dict(name='h', type='d ∈ [2, 3]')])
        result = self.gate().validate_census(self.log([row]), {row['name']}, {'Demo'})
        self.assertEqual(len(result['proofs'][0]['prop_premises']), 1)

    def test_duplicate_declaration_rejected(self):
        with self.assertRaises(ValueError):
            self.gate().validate_census(self.log([self.row(), self.row()]), {'Demo.honest'}, {'Demo'})

    def test_forged_count_and_missing_module_rejected(self):
        for log in [self.log(count=2), self.log(modules=[]), self.log(modules=['Other'])]:
            with self.subTest(log=log), self.assertRaises(ValueError):
                self.gate().validate_census(log, {'Demo.honest'}, {'Demo'})

    def test_missing_target_rejected(self):
        with self.assertRaises(ValueError):
            self.gate().validate_census(self.log(), {'Demo.missing'}, {'Demo'})

    def test_duplicate_json_key_rejected(self):
        log = self.log().replace('"kind": "theorem"', '"kind": "theorem", "kind": "definition"')
        with self.assertRaises(ValueError):
            self.gate().validate_census(log, {'Demo.honest'}, {'Demo'})

    def test_missing_end_or_unknown_line_rejected(self):
        for log in [self.log().split('\n')[0], self.log() + '\nCBZ_FORGED anything']:
            with self.subTest(log=log), self.assertRaises(ValueError):
                self.gate().validate_census(log, {'Demo.honest'}, {'Demo'})


if __name__ == '__main__':
    unittest.main()
