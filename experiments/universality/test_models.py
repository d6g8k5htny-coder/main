"""Behavioral controls for distinct finite Fourier law candidates."""
import copy
import importlib
import importlib.util
import math
import json
from pathlib import Path
import tempfile
import unittest


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.universality.models'),
                             'The declarative law catalog and generator are missing')
        self.models = importlib.import_module('experiments.universality.models')
        self.catalog = self.models.load_catalog()

    def test_fifty_laws_are_spectrum_distribution_pairs_not_seed_labels(self):
        items = self.catalog['models']
        self.assertEqual(len(items), 50)
        self.assertEqual(len({(x['spectrum_id'], x['coefficient_law_id']) for x in items}), 50)
        self.assertEqual(sum(x['class'] == 'gaussian_candidate' for x in items), 10)
        self.assertEqual(sum(x['class'] == 'non_gaussian_stress_candidate' for x in items), 40)
        for item in items:
            self.assertNotIn('seed', item)
            self.assertNotIn('scientific_status', item)

    def test_duplicate_distribution_spectrum_pair_is_rejected_despite_new_id(self):
        other = copy.deepcopy(self.catalog)
        other['models'][1]['spectrum_id'] = other['models'][0]['spectrum_id']
        other['models'][1]['coefficient_law_id'] = other['models'][0]['coefficient_law_id']
        other['models'][1]['class'] = other['models'][0]['class']
        with self.assertRaises(ValueError):
            self.models.validate_catalog(other)

    def test_seed_cannot_turn_one_law_into_another_model(self):
        other = copy.deepcopy(self.catalog)
        other['models'][0]['seed'] = 123
        with self.assertRaises(ValueError):
            self.models.validate_catalog(other)

    def test_boolean_schema_cannot_impersonate_schema_one(self):
        other = copy.deepcopy(self.catalog)
        other['schema_version'] = True
        with self.assertRaises(ValueError):
            self.models.validate_catalog(other)

    def test_catalog_metadata_cannot_relabel_the_exploratory_candidates(self):
        replacements = [
            (('purpose',), 'blinded_scientific_confirmation'),
            (('domain', 'space'), 'Euclidean plane'),
            (('domain', 'side'), 48),
            (('normalization',), 'No variance normalization'),
            (('scope',), ['All fifty candidates are scientifically confirmed']),
        ]
        replacements.extend((('default_parameters', key), value+1)
                            for key, value in self.catalog['default_parameters'].items())
        for path, wrong in replacements:
            other = copy.deepcopy(self.catalog)
            parent = other if len(path) == 1 else other[path[0]]
            parent[path[-1]] = wrong
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.models.validate_catalog(other)

    def test_catalog_metadata_requires_the_complete_supported_schema(self):
        for key in self.catalog:
            other = copy.deepcopy(self.catalog)
            del other[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.models.validate_catalog(other)
        for section in (None, 'domain', 'default_parameters'):
            other = copy.deepcopy(self.catalog)
            parent = other if section is None else other[section]
            parent['scientific_status'] = 'CONFIRMED'
            with self.subTest(extra=section), self.assertRaises(ValueError):
                self.models.validate_catalog(other)
        for section in ('domain', 'default_parameters'):
            for key in self.catalog[section]:
                other = copy.deepcopy(self.catalog)
                del other[section][key]
                with self.subTest(section=section, missing=key), self.assertRaises(ValueError):
                    self.models.validate_catalog(other)
        for key in ('purpose', 'domain', 'default_parameters', 'normalization', 'scope'):
            other = copy.deepcopy(self.catalog)
            other[key] = None
            with self.subTest(wrong_type=key), self.assertRaises(ValueError):
                self.models.validate_catalog(other)

    def test_catalog_metadata_rejects_equal_numeric_type_impostors(self):
        paths = [('schema_version',), ('scientific_status_authority',),
                 ('domain', 'dimension'), ('domain', 'side')]
        paths.extend(('default_parameters', key) for key in self.catalog['default_parameters'])
        for path in paths:
            other = copy.deepcopy(self.catalog)
            parent = other if len(path) == 1 else other[path[0]]
            parent[path[-1]] = float(parent[path[-1]])
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.models.validate_catalog(other)

    def test_alternate_catalog_and_injected_definition_reject_catalog_claims(self):
        for mutation in ('added_status', 'changed_scope', 'changed_space'):
            other = copy.deepcopy(self.catalog)
            if mutation == 'added_status':
                other['scientific_status'] = 'CONFIRMED'
            elif mutation == 'changed_scope':
                other['scope'] = ['These fields certify continuum universality']
            else:
                other['domain']['space'] = 'Euclidean plane'
            with tempfile.TemporaryDirectory() as scratch:
                path = Path(scratch)/'misdescribed-catalog.json'
                path.write_text(json.dumps(other), encoding='utf-8')
                with self.subTest(mutation=mutation, entry='file'), self.assertRaises(ValueError):
                    self.models.load_catalog(path)
            with self.subTest(mutation=mutation, entry='definition'), self.assertRaises(ValueError):
                self.models.definition(other['models'][0], catalog=other)

    def test_law_moments_and_input_contract_cannot_misdescribe_the_sampler(self):
        for law_id in self.catalog['coefficient_laws']:
            for key, wrong in [('component_fourth_moment', '0'),
                               ('input_contract', 'Certified IID Gaussian sampler')]:
                other = copy.deepcopy(self.catalog)
                other['coefficient_laws'][law_id][key] = wrong
                with self.subTest(law_id=law_id, key=key), self.assertRaises(ValueError):
                    self.models.validate_catalog(other)
        student = copy.deepcopy(self.catalog)
        student['coefficient_laws']['student5']['component_fourth_moment'] = '3'
        with self.assertRaises(ValueError):
            self.models.validate_catalog(student)

    def test_law_metadata_requires_the_complete_supported_schema(self):
        for mutation in ('missing', 'extra', 'not_mapping'):
            other = copy.deepcopy(self.catalog)
            law = other['coefficient_laws']['gaussian']
            if mutation == 'missing':
                del law['component_fourth_moment']
            elif mutation == 'extra':
                law['scientific_status'] = 'confirmed'
            else:
                other['coefficient_laws']['gaussian'] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.models.validate_catalog(other)

    def test_alternate_catalog_and_injected_definition_cannot_bypass_law_validation(self):
        other = copy.deepcopy(self.catalog)
        other['coefficient_laws']['student5']['component_fourth_moment'] = '3'
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch)/'misdescribed.json'
            path.write_text(json.dumps(other), encoding='utf-8')
            with self.assertRaises(ValueError):
                self.models.load_catalog(path)
        with self.assertRaises(ValueError):
            self.models.definition(other['models'][0], catalog=other)

    def test_invalid_declared_dimension_does_not_describe_a_planar_sampler(self):
        for dimension in (True, 3, 2.0):
            other = copy.deepcopy(self.catalog)
            other['domain']['dimension'] = dimension
            with self.assertRaises(ValueError):
                self.models.validate_catalog(other)

    def test_spectrum_metadata_cannot_misdescribe_the_implemented_weights(self):
        for spectrum_id in self.catalog['spectra']:
            for key, wrong in [('mechanism', 'Certified continuum Gaussian law'),
                               ('formula', '1')]:
                other = copy.deepcopy(self.catalog)
                other['spectra'][spectrum_id][key] = wrong
                with self.subTest(spectrum_id=spectrum_id, key=key), self.assertRaises(ValueError):
                    self.models.validate_catalog(other)

    def test_spectrum_metadata_requires_the_complete_supported_schema(self):
        for mutation in ('missing_mechanism', 'missing_formula', 'extra', 'not_mapping'):
            other = copy.deepcopy(self.catalog)
            spectrum = other['spectra']['gaussian']
            if mutation.startswith('missing_'):
                del spectrum[mutation.removeprefix('missing_')]
            elif mutation == 'extra':
                spectrum['scientific_status'] = 'confirmed'
            else:
                other['spectra']['gaussian'] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.models.validate_catalog(other)

    def test_alternate_catalog_and_injected_definition_reject_false_spectrum_metadata(self):
        for spectrum_id in self.catalog['spectra']:
            other = copy.deepcopy(self.catalog)
            other['spectra'][spectrum_id]['mechanism'] = 'Certified continuum Gaussian law'
            model = next(x for x in other['models'] if x['spectrum_id'] == spectrum_id)
            with tempfile.TemporaryDirectory() as scratch:
                path = Path(scratch)/'misdescribed-spectrum.json'
                path.write_text(json.dumps(other), encoding='utf-8')
                with self.subTest(spectrum_id=spectrum_id, entry='file'), self.assertRaises(ValueError):
                    self.models.load_catalog(path)
            with self.subTest(spectrum_id=spectrum_id, entry='definition'), self.assertRaises(ValueError):
                self.models.definition(model, catalog=other)

    def test_normalized_spectra_are_positive_symmetric_and_covariance_distinct(self):
        for cutoff in (2, 3):
            profiles = []
            for spectrum_id in self.catalog['spectra']:
                weights = self.models.normalized_spectrum(spectrum_id, cutoff)
                self.assertEqual(len(weights), (2*cutoff+1)**2)
                self.assertAlmostEqual(math.fsum(weights.values()), 1.0, places=14)
                self.assertTrue(all(w > 0 and math.isfinite(w) for w in weights.values()))
                for (x, y), weight in weights.items():
                    self.assertEqual(weight, weights[-x, -y])
                profiles.append(tuple(weights[k] for k in sorted(weights)))
                self.assertAlmostEqual(self.models.covariance(weights, (0.0, 0.0)), 1.0, places=14)
            for i, left in enumerate(profiles):
                for right in profiles[i+1:]:
                    self.assertGreater(max(abs(a-b) for a, b in zip(left, right)), 1e-10)

    def test_laws_have_different_marginals_without_cartesian_discrete_sampling(self):
        laws = self.catalog['coefficient_laws']
        self.assertEqual({x['component_fourth_moment'] for x in laws.values()},
                         {'3', '3/2', '2', '6', '9'})
        self.assertTrue(all(x['ideal_component_variance'] == '1' for x in laws.values()))
        self.assertTrue(all(x['ideal_vector_symmetry'] == 'circular' for x in laws.values()))

    def test_cutoff_one_is_refused_because_two_spectra_coincide(self):
        # On {-1, 0, 1}, x*x+y*y equals abs(x)+abs(y), so these
        # mechanisms cannot describe distinct covariance laws at cutoff one.
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                self.assertEqual(x*x+y*y, abs(x)+abs(y))
        for spectrum_id in self.catalog['spectra']:
            with self.subTest(spectrum_id=spectrum_id), self.assertRaises(ValueError):
                self.models.normalized_spectrum(spectrum_id, 1)
        model = self.catalog['models'][0]
        with self.assertRaises(ValueError):
            self.models.definition(model, cutoff=1)
        with self.assertRaises(ValueError):
            self.models.sample_grid(model, 123, n=8, cutoff=1)

    def test_real_grid_replays_for_each_coefficient_law(self):
        for law_id in self.catalog['coefficient_laws']:
            model = next(x for x in self.catalog['models'] if x['coefficient_law_id'] == law_id)
            first = self.models.sample_grid(model, 321, n=8, cutoff=2)
            second = self.models.sample_grid(model, 321, n=8, cutoff=2)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 64)
            self.assertTrue(all(type(x) is float and math.isfinite(x) for x in first))
            self.assertNotEqual(first, self.models.sample_grid(model, 322, n=8, cutoff=2))

    def test_aliasing_and_invalid_grid_parameters_fail_before_sampling(self):
        model = self.catalog['models'][0]
        for n, cutoff in [(6, 3), (5, 3), (True, 2), (8, 0), (8, True)]:
            with self.subTest(n=n, cutoff=cutoff), self.assertRaises(ValueError):
                self.models.sample_grid(model, 1, n=n, cutoff=cutoff)

    def test_definition_hash_is_seed_independent_but_cutoff_and_law_sensitive(self):
        first, second = self.catalog['models'][:2]
        a = self.models.definition(first, cutoff=2)
        self.assertNotIn('seed', a)
        self.assertNotEqual(self.models.digest(a), self.models.digest(self.models.definition(first, cutoff=3)))
        self.assertNotEqual(self.models.digest(a), self.models.digest(self.models.definition(second, cutoff=2)))
        self.assertEqual(len(self.models.digest(a)), 64)


if __name__ == '__main__':
    unittest.main()
