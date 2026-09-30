"""Exact arithmetic and scope controls for rounded finite Fourier polynomials."""
import copy
import importlib.util
import unittest
import hashlib
import json
import subprocess
import sys
import tempfile
from fractions import Fraction as Q
from pathlib import Path

if importlib.util.find_spec('finite_certificate'):
    import finite_certificate as fc
else:
    fc = None


def sample():
    return {'schema_version': 1, 'model': 'exact_dyadic_rounded_coefficients',
            'side': 24, 'cutoff': 1, 'grids': [4, 8],
            'records': [{'seed': 0, 'dc': ['0', '0'],
                         'modes': [[0, 1, '0', '0'], [1, -1, '0', '0'],
                                   [1, 0, '1/2', '0'], [1, 1, '0', '0']]}]}


class CertificateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(fc, 'Exact finite-polynomial certificate is not implemented')

    def test_sqrt_enclosures_include_perfect_squares_and_nonsquares(self):
        for p in range(41):
            for q in range(1, 18):
                x = Q(p, q)
                lower, upper = fc.sqrt_bounds(x, bits=32)
                self.assertLessEqual(lower * lower, x)
                self.assertGreaterEqual(upper * upper, x)
                self.assertLessEqual(upper - lower, Q(1, 2**32))
        self.assertEqual(fc.sqrt_bounds(Q(9, 16)), (Q(3, 4), Q(3, 4)))
        with self.assertRaises(ValueError):
            fc.sqrt_bounds(Q(-1))

    def test_pi_bounds_enclose_independent_decimal_bracket(self):
        lower, upper = fc.pi_bounds()
        # This wide familiar bracket is a sanity control, not the pi proof.
        self.assertGreater(lower, Q('3.14159265358979323846264338327950288'))
        self.assertLess(upper, Q('3.14159265358979323846264338327950289'))
        self.assertLessEqual(upper - lower, Q(2, 2**128))

    def test_pi_encloses_independent_two_arctangent_identity(self):
        # atan(1/2)+atan(1/3)=pi/4, with positive sum below pi/2.
        low = high = Q(0)
        for denominator, terms in [(2, 160), (3, 120)]:
            total = sum((Q((-1)**j, (2*j+1)*denominator**(2*j+1)) for j in range(terms)), Q(0))
            low += total
            high += total + Q(1, (2*terms+1)*denominator**(2*terms+1))
        lower, upper = fc.pi_bounds()
        self.assertLessEqual(lower, 4*low)
        self.assertGreaterEqual(upper, 4*high)

    def test_single_cosine_majorants_and_no_nodal_promotion(self):
        result = fc.certify(sample())
        row = result['records'][0]
        pi_upper = Q(result['pi_interval'][1])
        self.assertEqual(Q(row['bounds']['H']), pi_upper**2 / 144)
        self.assertEqual(Q(row['bounds']['Mxx']), pi_upper**2 / 144)
        self.assertEqual(Q(row['bounds']['Myy']), 0)
        self.assertEqual(Q(row['bounds']['Mxy']), 0)
        self.assertEqual(Q(row['grids'][0]['spatial_bound']), pi_upper**2 / 32)
        self.assertIsNone(row['grids'][0]['nodal_error'])
        self.assertIsNone(row['grids'][0]['diagram_error'])
        self.assertFalse(result['continuum_field_certified'])
        self.assertFalse(result['historical_fft_certified'])

    def test_output_decimal_is_outward_and_never_negative_zero(self):
        self.assertEqual(fc.decimal_up(Q(1, 3), 6), '0.333334')
        self.assertEqual(fc.decimal_up(Q(1, 8), 6), '0.125000')
        self.assertEqual(fc.decimal_up(Q(0), 3), '0.000')

    def test_mixed_mode_captures_both_components_and_cross_derivative(self):
        data = sample()
        data['records'][0]['modes'][2][2] = '0'
        data['records'][0]['modes'][3][2:] = ['3/8', '1/2']
        result = fc.certify(data); row = result['records'][0]
        frequency2 = (Q(result['pi_interval'][1])/12)**2
        self.assertEqual(Q(row['bounds']['amplitude_sum']), Q(5, 4))
        for key in ('Mxx', 'Myy', 'Mxy'):
            self.assertEqual(Q(row['bounds'][key]), Q(5, 4)*frequency2)
        self.assertEqual(Q(row['bounds']['H']), Q(5, 2)*frequency2)
        self.assertEqual(Q(row['grids'][0]['spatial_bound']), Q(5, 32)*Q(result['pi_interval'][1])**2)

    def test_strict_coefficient_schema_rejects_mutants(self):
        mutations = []
        for bad in ['NaN', 'Infinity', '1/3', '2/4', '+1', '01', '0.5', True, 0.5]:
            data = sample(); data['records'][0]['modes'][2][2] = bad; mutations.append(data)
        for bad in [True, 1.0, '1']:
            data = sample(); data['records'][0]['modes'][2][0] = bad; mutations.append(data)
        data = sample(); data['records'][0]['modes'][0][0] = -1; mutations.append(data)
        data = sample(); data['records'][0]['modes'][0][1] = 0; mutations.append(data)
        data = sample(); data['records'][0]['modes'].pop(); mutations.append(data)
        data = sample(); data['records'][0]['modes'][0] = data['records'][0]['modes'][1]; mutations.append(data)
        data = sample(); data['records'][0]['dc'][1] = '1'; mutations.append(data)
        data = sample(); data['records'].append(copy.deepcopy(data['records'][0])); mutations.append(data)
        data = sample(); data['side'] = True; mutations.append(data)
        data = sample(); data['grids'] = [2]; mutations.append(data)
        data = sample(); data['cutoff'] = 2; mutations.append(data)
        for data in mutations:
            with self.subTest(data=data), self.assertRaises(ValueError):
                fc.certify(data)

    def test_certificate_binding_and_false_scope_mutants_fail(self):
        data = sample(); result = fc.certify(data)
        self.assertTrue(fc.verify(data, result))
        bad = copy.deepcopy(result); bad['records'][0]['bounds']['H'] = '0'
        with self.assertRaises(ValueError): fc.verify(data, bad)
        bad = copy.deepcopy(result); bad['historical_fft_certified'] = True
        with self.assertRaises(ValueError): fc.verify(data, bad)
        bad = copy.deepcopy(result); bad['records'][0]['grids'][0]['nodal_error'] = '0'
        with self.assertRaises(ValueError): fc.verify(data, bad)
        bad = copy.deepcopy(data); bad['records'][0]['modes'][2][2] = '1'
        with self.assertRaises(ValueError): fc.verify(bad, result)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'Extraction needs the pinned numerical environment; core replay does not')
    def test_extractor_captures_the_actual_generator_input(self):
        self.assertIsNotNone(importlib.util.find_spec('run_certificate'), 'Coefficient extractor is not implemented')
        import run_certificate as runner
        import experiment as ex
        import math
        config = {'side': 24, 'dimension': 2, 'cutoff': 24, 'bank_max_cutoff': 24,
                  'seeds': [34000], 'grids': [128, 256, 512, 1024]}
        original = ex.np.fft.ifft2
        data = runner.extract_coefficients(config)
        self.assertIs(ex.np.fft.ifft2, original)
        self.assertEqual(len(data['records'][0]['modes']), 1200)
        bank = ex.mode_bank(34000, 24)
        self.assertEqual(Q(data['records'][0]['dc'][0]), Q.from_float(bank[0, 0][0] / math.sqrt(ex.denominator())))
        modes = {(x, y): (Q(a), Q(b)) for x, y, a, b in data['records'][0]['modes']}
        for x, y in [(0, 1), (1, -1), (24, 24)]:
            a, b = bank[x, y]
            weight = math.exp(-2*math.pi**2*(x*x+y*y)/24**2) / ex.denominator()
            z = math.sqrt(weight/2)*complex(a, -b)
            self.assertEqual(modes[x, y], (Q.from_float(z.real), Q.from_float(z.imag)))
        self.assertTrue(fc.verify(data, fc.certify(data)))

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(self):
        self.assertTrue(hasattr(fc, 'load_json'), 'Strict JSON loading is not implemented')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'input.json'
            for text in ['{"a":1,"a":1}', '{"nested":{"x":0,"x":1}}', '{"x":NaN}', '{"x":Infinity}']:
                path.write_text(text)
                with self.assertRaises(ValueError): fc.load_json(path)

    def test_real_bundle_and_cli_reject_byte_source_scope_and_report_mutants(self):
        self.assertTrue(hasattr(fc, 'render_report'), 'Deterministic report verification is not implemented')
        base = Path(__file__).resolve().parent
        config = json.loads((base/'refinement_config.json').read_text())
        k = config['cutoff']
        modes = [[x, y, '0', '0'] for x in range(k+1) for y in range(-k, k+1) if x > 0 or y > 0]
        data = {'schema_version': 1, 'model': fc.MODEL, 'side': 24, 'cutoff': k, 'grids': config['grids'],
                'records': [{'seed': seed, 'dc': ['0', '0'], 'modes': modes} for seed in config['seeds']]}
        cert = fc.certify(data)
        raw = {'COEFFICIENTS.json': fc.canonical_bytes(data), 'CERTIFICATE.json': fc.canonical_bytes(cert),
               'RESULTS.md': fc.render_report(cert).encode()}
        receipt = {'schema_version': 1, 'outputs': {}, 'sources': {}}
        for name in ('COEFFICIENTS.json', 'CERTIFICATE.json'):
            receipt['outputs'][name] = {'bytes': len(raw[name]), 'sha256': hashlib.sha256(raw[name]).hexdigest()}
        for role, name in {'core':'finite_certificate.py', 'extractor':'run_certificate.py', 'generator':'experiment.py',
                           'configuration':'refinement_config.json', 'dependencies':'requirements.txt'}.items():
            body = (base/name).read_bytes()
            receipt['sources'][role] = {'filename':name, 'bytes':len(body), 'sha256':hashlib.sha256(body).hexdigest()}
        raw['RUN.json'] = fc.canonical_bytes(receipt)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            def reset():
                for name, body in raw.items(): (directory/name).write_bytes(body)
            reset()
            self.assertTrue(fc.verify_directory(directory))
            command = [sys.executable, '-B', '-S', str(base/'finite_certificate.py'), '--verify', str(directory)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            for name in ('COEFFICIENTS.json', 'CERTIFICATE.json', 'RUN.json'):
                reset(); (directory/name).write_bytes(b'{"schema_version":1,'+raw[name][1:])
                with self.assertRaises(ValueError): fc.verify_directory(directory)
            reset(); altered = copy.deepcopy(receipt); altered['sources']['generator']['sha256'] = '0'*64
            (directory/'RUN.json').write_bytes(fc.canonical_bytes(altered))
            with self.assertRaises(ValueError): fc.verify_directory(directory)
            reset(); altered = copy.deepcopy(receipt); altered['outputs']['COEFFICIENTS.json']['sha256'] = '0'*64
            (directory/'RUN.json').write_bytes(fc.canonical_bytes(altered))
            with self.assertRaises(ValueError): fc.verify_directory(directory)
            reset(); changed = copy.deepcopy(cert); changed['records'][0]['grids'][0]['nodal_error'] = '0'
            content = fc.canonical_bytes(changed); (directory/'CERTIFICATE.json').write_bytes(content)
            altered = copy.deepcopy(receipt)
            altered['outputs']['CERTIFICATE.json'] = {'bytes':len(content), 'sha256':hashlib.sha256(content).hexdigest()}
            (directory/'RUN.json').write_bytes(fc.canonical_bytes(altered))
            with self.assertRaises(ValueError): fc.verify_directory(directory)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            reset(); (directory/'RESULTS.md').write_text('All FFT samples certified.\n')
            with self.assertRaises(ValueError): fc.verify_directory(directory)


if __name__ == '__main__':
    unittest.main()
