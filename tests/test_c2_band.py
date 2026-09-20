"""Internal exact arithmetic checks; no independent acceptance credit."""
import json
from fractions import Fraction as F
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pytest

from research.parallel.c2 import c2_band as C
from research.interval import Interval as I, exp
from research.rn.side24 import image_tail, kernel_derivative


class BindingTests(unittest.TestCase):
    def test_exact_source_definition_and_pin_ast(self):
        self.assertEqual(len(C.source_binding()), 2)

    def test_source_digest_corruption_rejected(self):
        with patch.dict(C.SOURCE_SHA, {'sources/H5_PROMOTE.md': '0'*64}):
            with self.assertRaisesRegex(ValueError, 'source identity mismatch'):
                C.source_binding()

    def test_six_pin_displacements_and_covariance_signs(self):
        g = C.geometry(*C.BAND)
        self.assertEqual(len(g['coordinates']), 6)
        self.assertEqual(len(g['covariance_pairs']), 36)
        for pair in g['covariance_pairs']:
            ci = g['coordinates'][pair['i']]
            cj = g['coordinates'][pair['j']]
            difference = F(ci['x_r_coefficient'])-F(cj['x_r_coefficient'])
            self.assertIn(difference, (F(-1), F(0), F(1)))
            self.assertEqual(F(pair['x_r_coefficient']), difference)
            self.assertEqual(pair['second_argument_sign'], (-1)**sum(cj['derivative']))
        self.assertEqual(g['covariance_pairs'][3]['x_box'], {'lo': '-1/20', 'hi': '-7071/200000'})

    def test_missing_second_argument_sign_gives_negative_variance(self):
        k2 = kernel_derivative(2, F(0), prec=220, bits=768)
        self.assertGreater((-k2).lo, 0)  # Cov(f_x,f_x)=-k''(0)
        self.assertLess(k2.hi, 0)       # INVALID omission would be a negative variance.

    def test_precise_scope_fences(self):
        r = C.build_report()
        self.assertEqual(r['jet']['jets_bound'], 1)
        self.assertEqual(r['jet']['jets_required'], 24)
        self.assertFalse(r['obligation_closed'])
        self.assertEqual(r['independence_credit'], 0)
        self.assertEqual(r['scientific_promotions'], 0)
        self.assertTrue(r['modulus_comparison'].startswith('INSUFFICIENT_DATA'))


class ArithmeticTests(unittest.TestCase):
    def test_series_continuous_extension(self):
        self.assertEqual(C.iv(C.central_series(F(0))), {'lo': '1/2', 'hi': '1/2'})

    def test_whole_band_width_and_point_controls(self):
        enc, _ = C.enclose()
        self.assertLess(enc.width(), F(78093, 500000000))
        # Supplementary checks only. The integral/tail argument proves the continuum.
        for j in range(17):
            r = C.BAND[0]+(C.BAND[1]-C.BAND[0])*F(j, 16)
            direct = C.direct_point(r)
            self.assertLessEqual(enc.lo, direct.lo)
            self.assertGreaterEqual(enc.hi, direct.hi)

    def test_subband_enclosure_tightens(self):
        full, _ = C.enclose()
        mid = sum(C.BAND)/2
        left, _ = C.enclose(C.BAND[0], mid)
        right, _ = C.enclose(mid, C.BAND[1])
        self.assertLess(left.width(), full.width())
        self.assertLess(right.width(), full.width())
        self.assertLessEqual(full.lo, min(left.lo, right.lo))
        self.assertGreaterEqual(full.hi, max(left.hi, right.hi))

    def test_numerator_image_omission_fails_containment(self):
        r = F(1, 20)
        truth = C.direct_point(r)
        wrong, _ = C.enclose(r, r, omit_images=True)
        self.assertGreater(wrong.lo, truth.hi)

    def test_image_sign_flip_fails_containment(self):
        r = F(1, 20)
        truth = C.direct_point(r)
        wrong, _ = C.enclose(r, r, flip_image_sign=True)
        self.assertGreater(wrong.lo, truth.hi)

    def test_missing_denominator_tail_fails_strict_positive_witness(self):
        full, truncated, tail = C.denominator()
        self.assertGreater(tail.lo, 0)
        self.assertGreater(full.lo, truncated.hi)
        # Even one omitted pair is enough: Z_true >= Z0 + 2 exp(-1152).
        lower_witness = truncated.lo+2*exp(I.exact(-1152), 400).lo
        self.assertGreater(lower_witness, truncated.hi)

    def test_hundredfold_weakened_numerator_tail_fails(self):
        radius = F(1, 20)
        tail = image_tail(2, radius, prec=220)
        x = 48-radius
        actual_one_term = (x*x-1)*exp(I.exact(-x*x/2), 220)
        self.assertGreater(actual_one_term.lo, tail/100)
        self.assertGreater(tail, actual_one_term.hi)

    def test_hundredfold_weakened_denominator_tail_fails(self):
        tail = image_tail(0, F(0), prec=220)
        actual_first_pair = 2*exp(I.exact(-1152), 220)
        self.assertGreater(actual_first_pair.lo, tail/100)
        self.assertGreaterEqual(tail, actual_first_pair.hi)

    def test_wrong_r_power_excludes_correct_value(self):
        r = F(1, 20)
        truth = C.direct_point(r)
        wrong = truth*r  # Raw covariance difference divided by r, not r².
        self.assertLess(wrong.hi, truth.lo)

    def test_zero_image_limit_matches_plane(self):
        central = C.central_series(F(1,20))
        correct, _ = C.enclose(F(1,20), F(1,20))
        self.assertLess(correct.hi, central.lo)


class ReplayTests(unittest.TestCase):
    def test_report_replays_exactly(self):
        self.assertEqual(C.build_report(), json.loads((C.HERE/'candidate.json').read_text()))

    def test_normal_and_optimized_cli_match_and_mutation_rejected(self):
        script = str(C.HERE/'c2_band.py')
        good = str(C.HERE/'candidate.json')
        cmd = [sys.executable, '-B', script, '--verify', good]
        normal = subprocess.run(cmd, text=True, capture_output=True, check=False)
        optimized = subprocess.run([sys.executable, '-B', '-O', script, '--verify', good],
                                   text=True, capture_output=True, check=False)
        self.assertEqual(normal.returncode, 0, normal.stderr)
        self.assertEqual(optimized.returncode, 0, optimized.stderr)
        self.assertEqual(normal.stdout, optimized.stdout)
        with tempfile.TemporaryDirectory() as td:
            altered = json.loads(Path(good).read_text())
            altered['enclosure']['lo'] = '1/2'
            path = Path(td)/'mutated.json'
            path.write_text(json.dumps(altered))
            for mode in ([], ['-O']):
                bad = subprocess.run([sys.executable, '-B', *mode, script, '--verify', str(path)],
                                     text=True, capture_output=True, check=False)
                self.assertNotEqual(bad.returncode, 0)
                self.assertIn('differs from exact replay', bad.stderr)

    def test_default_candidate_replays_outside_checkout_and_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'retained.json'
            output.write_bytes(b'original retained bytes')
            script = str(C.HERE/'c2_band.py')
            result = subprocess.run([sys.executable, '-B', script], cwd=directory,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            refused = subprocess.run([sys.executable, '-B', script, '--output', str(output)],
                                     cwd=directory, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(refused.returncode, 0)
            self.assertEqual(output.read_bytes(), b'original retained bytes')


# Each formerly nested subTest is a separately collected pytest case. The
# complete verification runner requires declared JUnit tests to equal actual
# testcase elements; no outcome is hidden inside an aggregate parent case.
@pytest.mark.parametrize('bad', [True, 0.05, '1/20'], ids=['bool', 'float', 'string'])
def test_type_guards(bad):
    with pytest.raises(TypeError):
        C.enclose(bad, F(1, 20))


@pytest.mark.parametrize('lo,hi', [(0, F(1, 20)), (F(1, 20), F(1, 40)),
                                  (F(1, 40), F(1, 10))],
                         ids=['zero-lower', 'reversed', 'above-domain'])
def test_band_guards(lo, hi):
    with pytest.raises(ValueError):
        C.enclose(lo, hi)


@pytest.mark.parametrize('r', [F(1, 10000), F(1, 40), C.BAND[0], F(1, 20)],
                         ids=['small-radius', 'one-fortieth', 'band-lower', 'band-upper'])
def test_series_against_direct_certified_exponential(r):
    direct = (1-exp(I.exact(-r*r/2), 400))/(r*r)
    series = C.central_series(r, terms=32)
    assert series.intersect(direct) is not None
    # The high-precision direct enclosure is strictly inside the larger
    # 32-term rational alternating-series enclosure.
    assert series.lo <= direct.lo
    assert series.hi >= direct.hi


@pytest.mark.parametrize('kind', ['bool-alias', 'duplicate-key', 'float-number'])
@pytest.mark.parametrize('optimized', [False, True], ids=['normal', 'optimized'])
def test_strict_cli_refuses_aliases_duplicates_and_noninteger_numbers(tmp_path, kind, optimized):
    raw = (C.HERE/'candidate.json').read_text()
    if kind == 'bool-alias':
        alias = json.loads(raw)
        alias['scientific_promotions'] = False  # Python dict equality aliases 0.
        content = json.dumps(alias)
    elif kind == 'duplicate-key':
        content = raw.replace('"schema_version": 1',
                              '"schema_version": 1, "schema_version": 1', 1)
    else:
        content = raw.replace('"schema_version": 1', '"schema_version": 1.0', 1)
    path = tmp_path/'invalid.json'
    path.write_text(content)
    mode = ['-O'] if optimized else []
    result = subprocess.run([sys.executable, '-B', *mode,
        str(C.HERE/'c2_band.py'), '--verify', str(path)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'PASS:' not in result.stdout


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
