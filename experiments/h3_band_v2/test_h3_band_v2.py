"""Controls for the H3 band certificate successor v2.

Every check goes through the program's command line, so a test exercises the same code path
as a replay. Structure and record checks need only the standard library; the replays and the
mutation controls need mpmath and are skipped without it.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROG = HERE / 'h3_band_v2.py'
RESULTS = HERE / 'results'

try:
    import mpmath  # noqa: F401
    HAVE_MPMATH = True
except ImportError:
    HAVE_MPMATH = False


def run(*args):
    # under python -O the program runs with -O too, so both modes exercise every control
    opt = ['-O'] if sys.flags.optimize else []
    return subprocess.run([sys.executable, '-I', '-B', *opt, str(PROG), *map(str, args)],
                          capture_output=True, text=True, timeout=1800)


def run_with_mpmath(stub, *args):
    """Run the program without site-packages (-S) and with sys.modules['mpmath'] preset: None hides mpmath,
    a string fakes that version."""
    opt = ['-O'] if sys.flags.optimize else []
    code = ('import runpy, sys, types\n'
            'stub = %r\n'
            'if stub is None:\n'
            '    sys.modules["mpmath"] = None\n'
            'else:\n'
            '    fake = types.ModuleType("mpmath")\n'
            '    fake.__version__, fake.mp, fake.iv, fake.mpf = stub, None, None, None\n'
            '    sys.modules["mpmath"] = fake\n'
            'sys.argv = [%r] + %r\n'
            'sys.exit(runpy.run_path(%r, run_name="__main__"))\n') % (stub, str(PROG), [str(a) for a in args], str(PROG))
    return subprocess.run([sys.executable, '-I', '-B', '-S', *opt, '-c', code], capture_output=True, text=True, timeout=600)


def run_with_mpmath_version(version, *args):
    """Run the program with the installed mpmath but its __version__ replaced, as if another release ran."""
    opt = ['-O'] if sys.flags.optimize else []
    code = ('import runpy, sys, mpmath\n'
            'mpmath.__version__ = %r\n'
            'sys.argv = [%r] + %r\n'
            'sys.exit(runpy.run_path(%r, run_name="__main__"))\n') % (version, str(PROG), [str(a) for a in args], str(PROG))
    return subprocess.run([sys.executable, '-I', '-B', *opt, '-c', code], capture_output=True, text=True, timeout=1800)


def load_program():
    spec = importlib.util.spec_from_file_location('h3_band_v2_under_test', PROG)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(result):
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith('H3_BAND_V2')]
    return lines[-1] if lines else ''


class StructureLemma(unittest.TestCase):
    def test_lemma_holds_for_the_functionals_of_record(self):
        r = run('structure')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('STRUCTURE PASS', summary(r))

    def test_each_branch_of_the_lemma_catches_its_violation(self):
        for mutation, message in (('structure_pin_pole', 'CB2s (a) pin V5 must have no z^-2 term'),
                                  ('structure_soft_pole2', 'CB2s (b) soft Y2 must have no z^-2 term'),
                                  ('structure_soft_span', 'CB2s (c) soft Y0 pole part must lie in the span'),
                                  ('structure_window', 'CB2s (d) E zV = 0 required')):
            with self.subTest(mutation=mutation):
                r = run('structure', '--mutation', mutation)
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn('STRUCTURE FAIL', summary(r))
                self.assertIn(message, summary(r))

    def test_a_mutation_the_lemma_cannot_see_is_not_a_pass(self):
        # wrong_window moves a soft functional the lemma accepts; planar_moments leaves the tables alone
        for mutation in ('wrong_window', 'planar_moments'):
            with self.subTest(mutation=mutation):
                r = run('structure', '--mutation', mutation)
                self.assertEqual(r.returncode, 3, r.stdout)
                self.assertIn('NOT A CERTIFICATE', summary(r))
                self.assertNotIn('STRUCTURE PASS', r.stdout)


class OutwardDecimals(unittest.TestCase):
    def test_rounding_directions(self):
        m = load_program()
        x = Fraction('3.657186832400983468')
        self.assertEqual(m.decimal_outward(x, 'up'), '3.65718683240099')
        self.assertEqual(m.decimal_outward(x, 'down'), '3.65718683240098')
        self.assertEqual(m.decimal_outward(Fraction('-2.5'), 'up', 3), '-2.50')
        self.assertEqual(m.decimal_outward(Fraction(1, 3), 'down'), '0.333333333333333')
        self.assertEqual(m.decimal_outward(Fraction(1, 3), 'up'), '0.333333333333334')
        self.assertEqual(m.decimal_outward(Fraction(123456), 'up', 3), '124000')

    def test_rounded_value_brackets_the_input(self):
        m = load_program()
        for x in (Fraction(2, 7), Fraction('3.73661723530032999'), Fraction(-5, 3), Fraction(10 ** 20 + 1, 3)):
            self.assertGreaterEqual(Fraction(m.decimal_outward(x, 'up')), x)
            self.assertLessEqual(Fraction(m.decimal_outward(x, 'down')), x)


class ReviewedRecords(unittest.TestCase):
    """verify-record compares a transcript with a reviewed record without recomputing."""

    def check(self, transcript, record):
        return run('verify-record', '--transcript', transcript, '--results', record)

    def test_committed_records_match_committed_transcripts(self):
        for mode in ('ceil', 'floor'):
            r = self.check(RESULTS / ('%s.transcript.txt' % mode), RESULTS / ('%s.json' % mode))
            self.assertEqual(r.returncode, 0, r.stdout)
            self.assertIn('VERIFY PASS', summary(r))

    def test_verify_record_does_not_depend_on_the_verifiers_mpmath(self):
        # verify-record recomputes nothing, so neither a missing mpmath nor another version may matter
        for stub in (None, '0.0'):
            for mode in ('ceil', 'floor'):
                with self.subTest(mpmath=stub, mode=mode):
                    r = run_with_mpmath(stub, 'verify-record', '--transcript', RESULTS / ('%s.transcript.txt' % mode),
                                        '--results', RESULTS / ('%s.json' % mode))
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    self.assertIn('VERIFY PASS', summary(r))

    def test_compare_record_gates_mpmath_only_when_runtime(self):
        # the replay's own call site is covered by Replays.test_replay_rejects_a_record_made_with_another_mpmath
        m = load_program()
        expected = json.loads((RESULTS / 'ceil.json').read_text())
        fresh = json.loads(json.dumps(expected))
        fresh['environment']['mpmath'] = '0.0'
        self.assertIn('environment.mpmath', m.compare_record(fresh, expected, runtime=True))
        self.assertNotIn('environment.mpmath', m.compare_record(fresh, expected, runtime=False))
        # a reviewed record naming no producer, or with a malformed environment, is a mismatch in both paths,
        # never a traceback
        for change in ('no version', 'no mpmath key', 'no environment', 'environment not an object'):
            malformed = json.loads(json.dumps(expected))
            if change == 'no version':
                malformed['environment']['mpmath'] = None
            elif change == 'no mpmath key':
                del malformed['environment']['mpmath']
            elif change == 'no environment':
                del malformed['environment']
            else:
                malformed['environment'] = '1.4.1'
            with self.subTest(change=change):
                self.assertEqual(m.compare_record(expected, malformed, runtime=True),
                                 ['environment.mpmath pin', 'environment.mpmath'])
                self.assertEqual(m.compare_record(expected, malformed, runtime=False), ['environment.mpmath pin'])

    def test_committed_records_name_the_pinned_mpmath(self):
        # MPMATH_PIN, requirements.txt and the committed records must agree; every check then requires the pin
        pin = next(ln.split()[0].split('==')[1] for ln in (HERE / 'requirements.txt').read_text().splitlines()
                   if ln.startswith('mpmath=='))
        self.assertEqual(load_program().MPMATH_PIN, pin)
        for mode in ('ceil', 'floor'):
            rec = json.loads((RESULTS / ('%s.json' % mode)).read_text())
            self.assertEqual(rec['environment']['mpmath'], pin, mode)

    def test_a_record_naming_another_mpmath_is_rejected(self):
        # the reviewed record must name the pinned producer, whatever mpmath the verifier has
        for change in ('another version', 'no version', 'no mpmath key', 'no environment', 'environment not an object'):
            rec = json.loads((RESULTS / 'ceil.json').read_text())
            if change == 'another version':
                rec['environment']['mpmath'] = '0.0'
            elif change == 'no version':
                rec['environment']['mpmath'] = None
            elif change == 'no mpmath key':
                del rec['environment']['mpmath']
            elif change == 'no environment':
                del rec['environment']
            else:
                rec['environment'] = '1.4.1'
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'r.json'
                p.write_text(json.dumps(rec))
                for r in (self.check(RESULTS / 'ceil.transcript.txt', p),
                          run_with_mpmath(None, 'verify-record', '--transcript', RESULTS / 'ceil.transcript.txt', '--results', p)):
                    self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                    self.assertEqual(summary(r), 'H3_BAND_V2 VERIFY_FAIL: record mismatch in environment.mpmath pin')

    def test_a_relabelled_record_is_rejected(self):
        # the mode is read from the transcript's header, not taken from the record
        rec = json.loads((RESULTS / 'ceil.json').read_text())
        rec['mode'] = 'floor'
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'r.json'
            p.write_text(json.dumps(rec))
            r = self.check(RESULTS / 'ceil.transcript.txt', p)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertEqual(summary(r), 'H3_BAND_V2 VERIFY_FAIL: record mismatch in mode')

    def test_a_transcript_whose_header_names_no_envelope_is_rejected(self):
        # even with a record made consistent with the transcript; only the first line may name the envelope
        original = (RESULTS / 'ceil.transcript.txt').read_text()
        for layout, text, mode in (('no header', original.split('\n', 1)[1], None),
                                   ('no header', original.split('\n', 1)[1], 'ceil'),
                                   ('header on line 2', 'copied transcript\n' + original, 'ceil'),
                                   ('blank first line', '\n' + original, 'ceil'),
                                   ('indented header', ' ' + original, 'ceil')):
            rec = json.loads((RESULTS / 'ceil.json').read_text())
            rec['mode'] = mode
            rec['transcript_sha256'] = hashlib.sha256(text.encode()).hexdigest()
            with self.subTest(layout=layout, mode=mode), tempfile.TemporaryDirectory() as d:
                t = Path(d) / 't.txt'
                p = Path(d) / 'r.json'
                t.write_text(text)
                p.write_text(json.dumps(rec))
                r = self.check(t, p)
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertEqual(summary(r), 'H3_BAND_V2 VERIFY_FAIL: the transcript header names no envelope')

    NOT_OBJECTS = ('[]', '[1]', 'null', '"x"', '1', 'true')  # falsy and truthy, so the check must be on the type

    def test_a_record_that_is_not_an_object_is_rejected(self):
        for payload in self.NOT_OBJECTS:
            with self.subTest(record=payload), tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'r.json'
                p.write_text(payload)
                r = self.check(RESULTS / 'ceil.transcript.txt', p)
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertEqual(summary(r), 'H3_BAND_V2 VERIFY_FAIL: the reviewed record is not a JSON object')

    def test_replay_checks_the_record_type_before_running(self):
        # the fake mpmath cannot compute, so only a check made before the run can produce this summary;
        # Replays.test_replay_rejects_a_record_that_is_not_an_object_before_running repeats it with mpmath
        pin = load_program().MPMATH_PIN
        for payload in self.NOT_OBJECTS:
            with self.subTest(record=payload), tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'r.json'
                p.write_text(payload)
                r = run_with_mpmath(pin, 'ceil', '--verify', p)
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertEqual(summary(r), 'H3_BAND_V2 VERIFY_FAIL: the reviewed record is not a JSON object')

    def test_a_changed_check_line_is_rejected(self):
        # only the transcript digest binds the lines that carry no certified decimal
        text = (RESULTS / 'ceil.transcript.txt').read_text()
        line = next(ln for ln in text.splitlines() if ln.startswith('CB0 PASS:'))
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / 't.txt'
            t.write_text(text.replace(line, 'CB0 SKIPPED: H2 library digest not checked'))
            for r in (self.check(t, RESULTS / 'ceil.json'),
                      run_with_mpmath(None, 'verify-record', '--transcript', t, '--results', RESULTS / 'ceil.json')):
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertEqual(summary(r), 'H3_BAND_V2 VERIFY_FAIL: record mismatch in transcript_sha256')

    def test_records_carry_review_status_and_no_scientific_effect(self):
        for mode in ('ceil', 'floor'):
            rec = json.loads((RESULTS / ('%s.json' % mode)).read_text())
            self.assertEqual(rec['scientific_effect'], 'NONE')
            self.assertEqual(rec['review_status'], 'REVIEW_REQUIRED')
            self.assertEqual(len(rec['cells']), 8)

    def test_floor_lies_below_ceiling_on_every_cell(self):
        ceil = json.loads((RESULTS / 'ceil.json').read_text())['cells']
        floor = json.loads((RESULTS / 'floor.json').read_text())['cells']
        self.assertEqual([c['cell'] for c in ceil], [c['cell'] for c in floor])
        for c, f in zip(ceil, floor):
            self.assertEqual((c['relation'], f['relation']), ('<=', '>='))
            self.assertLess(Fraction(f['bound']), Fraction(c['bound']), c['cell'])

    def test_a_lowered_ceiling_in_the_transcript_is_rejected(self):
        text = (RESULTS / 'ceil.transcript.txt').read_text()
        line = next(ln for ln in text.splitlines() if ln.startswith('CERT cell=(0,1/16384]'))
        doctored = line[:-1] + str((int(line[-1]) + 9) % 10)
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / 't.txt'
            t.write_text(text.replace(line, doctored))
            r = self.check(t, RESULTS / 'ceil.json')
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn('VERIFY_FAIL', summary(r))

    def test_a_doctored_record_is_rejected(self):
        rec = json.loads((RESULTS / 'floor.json').read_text())
        rec['cells'][-1]['bound'] = '2.30659559567154'
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'r.json'
            p.write_text(json.dumps(rec))
            r = self.check(RESULTS / 'floor.transcript.txt', p)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn('cells', summary(r))

    def doctored_pair(self, mode, edit):
        # a transcript edited by `edit` and a record made consistent with it from the same program
        text = edit((RESULTS / ('%s.transcript.txt' % mode)).read_text())
        rec = load_program().make_record(mode, text)
        # the producer environment of the reviewed record, so only the intended defect differs
        rec['environment'] = json.loads((RESULTS / ('%s.json' % mode)).read_text())['environment']
        return text, rec

    def test_a_mutated_transcript_with_a_matching_record_is_rejected(self):
        text, rec = self.doctored_pair('ceil', lambda t: t.replace('MUTATION=none', 'MUTATION=drop_fail_piece'))
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / 't.txt'
            p = Path(d) / 'r.json'
            t.write_text(text)
            p.write_text(json.dumps(rec))
            r = self.check(t, p)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertEqual(summary(r), 'H3_BAND_V2 VERIFY_FAIL: record mismatch in mutation is not none')

    def test_a_transcript_without_all_checks_pass_is_rejected(self):
        text, rec = self.doctored_pair('floor', lambda t: t.replace('ALL_CHECKS_PASS\n', ''))
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / 't.txt'
            p = Path(d) / 'r.json'
            t.write_text(text)
            p.write_text(json.dumps(rec))
            r = self.check(t, p)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertEqual(summary(r), 'H3_BAND_V2 VERIFY_FAIL: record mismatch in ALL_CHECKS_PASS missing')

    def test_record_and_verify_together_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'same.json'
            r = run('ceil', '--record', p, '--verify', p)
            self.assertFalse(p.exists())
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn('not allowed with argument', r.stderr)

    def test_a_record_for_other_program_bytes_is_rejected(self):
        rec = json.loads((RESULTS / 'ceil.json').read_text())
        rec['program_sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'r.json'
            p.write_text(json.dumps(rec))
            r = self.check(RESULTS / 'ceil.transcript.txt', p)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn('program_sha256', summary(r))


@unittest.skipUnless(HAVE_MPMATH, 'mpmath is required for the certificate replays')
class Replays(unittest.TestCase):
    def test_ceiling_replay_matches_the_reviewed_record(self):
        r = run('ceil', '--verify', RESULTS / 'ceil.json')
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        self.assertIn('matches the reviewed record', summary(r))

    def test_floor_replay_matches_the_reviewed_record(self):
        r = run('floor', '--verify', RESULTS / 'floor.json')
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        self.assertIn('matches the reviewed record', summary(r))

    def test_replay_rejects_a_record_made_with_another_mpmath(self):
        # the record differs from the reviewed one only in the producer's mpmath version
        rec = json.loads((RESULTS / 'ceil.json').read_text())
        rec['environment']['mpmath'] = '0.0'
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'r.json'
            p.write_text(json.dumps(rec))
            r = run('ceil', '--verify', p)
        self.assertEqual(r.returncode, 1, r.stdout[-2000:])
        self.assertEqual(summary(r), 'H3_BAND_V2 VERIFY_FAIL: record mismatch in environment.mpmath pin, environment.mpmath')

    def test_replay_rejects_another_running_mpmath_for_the_pinned_record(self):
        # the record names the pin, so only the runtime gate can fire
        r = run_with_mpmath_version('0.0', 'ceil', '--verify', RESULTS / 'ceil.json')
        self.assertEqual(r.returncode, 1, r.stdout[-2000:] + r.stderr[-2000:])
        self.assertEqual(summary(r), 'H3_BAND_V2 VERIFY_FAIL: record mismatch in environment.mpmath')

    def test_replay_rejects_a_record_that_is_not_an_object_before_running(self):
        for payload in ReviewedRecords.NOT_OBJECTS:
            with self.subTest(record=payload), tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'r.json'
                p.write_text(payload)
                r = run('ceil', '--verify', p)
                self.assertEqual(r.returncode, 1, r.stdout[-2000:])
                self.assertEqual(summary(r), 'H3_BAND_V2 VERIFY_FAIL: the reviewed record is not a JSON object')
                self.assertNotIn('ALL_CHECKS_PASS', r.stdout)

    def test_replay_rejects_a_self_consistent_doctored_record(self):
        # transcript and record doctored together, so verify-record alone would accept them;
        # only recomputation exposes the change
        text = (RESULTS / 'ceil.transcript.txt').read_text().replace(
            'E[G_r] <= 3.73661723530033', 'E[G_r] <= 3.23510806397398')
        m = load_program()
        rec = json.loads((RESULTS / 'ceil.json').read_text())
        fresh = m.make_record('ceil', text)
        rec.update({k: fresh[k] for k in ('transcript_sha256', 'cells', 'uniform')})
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / 't.txt'
            p = Path(d) / 'r.json'
            t.write_text(text)
            p.write_text(json.dumps(rec))
            self.assertEqual(run('verify-record', '--transcript', t, '--results', p).returncode, 0)
            r = run('ceil', '--verify', p)
        self.assertEqual(r.returncode, 1, r.stdout[-2000:])
        self.assertIn('VERIFY_FAIL', summary(r))


@unittest.skipUnless(HAVE_MPMATH, 'mpmath is required for the mutation controls')
class Mutations(unittest.TestCase):
    # mutation -> (envelope, the check that must fail)
    CAUGHT_IN_RUN = {
        'structure_pin_pole': ('ceil', 'CB2s (a) pin V5 must have no z^-2 term'),
        'structure_soft_pole2': ('floor', 'CB2s (b) soft Y2 must have no z^-2 term'),
        'structure_soft_span': ('ceil', 'CB2s (c) soft Y0 pole part must lie in the span'),
        'structure_window': ('floor', 'CB2s (d) E zV = 0 required'),
        'planar_moments': ('ceil', 'CB1d consumed a_{2m} meets the spectral enclosure'),
        'wrong_window': ('ceil', 'CB3 mu_aM(0) = -1'),
        'signed_maxima': ('ceil', 'CB4 majorants dominate every |entry| (B1)'),
        'wide_first_cell': ('ceil', 'CB4 R-majorant converges on the first cell (0, 1/400]'),
        'one_tail': ('floor', 'CB5 event bound P(|X| > t) self-test'),
        'tail_helper_one_side': ('ceil', 'CB5 two-sided tail helper dominates twice the Gordon lower bound'),
        'no_preconditioner': ('ceil', 'CB6 Neumann contraction'),
        'cap_raised': ('ceil', 'CC6 consumption cap'),
    }
    CAUGHT_BY_REPLAY = ('drop_fail_piece', 'old_remainder')

    def test_every_mutation_has_a_control(self):
        self.assertEqual(set(load_program().MUTATIONS), set(self.CAUGHT_IN_RUN) | set(self.CAUGHT_BY_REPLAY))

    def test_mutations_are_caught_inside_the_run(self):
        for mutation, (mode, message) in self.CAUGHT_IN_RUN.items():
            with self.subTest(mutation=mutation):
                r = run(mode, '--mutation', mutation)
                self.assertEqual(r.returncode, 1, r.stdout[-2000:])
                self.assertIn('H3_BAND_V2 FAIL', summary(r))
                self.assertIn(message, summary(r))

    def test_mutations_are_caught_by_the_replay(self):
        # the mutation header alone already changes the transcript digest, so require the
        # certified decimals themselves to differ from the reviewed record
        reviewed = json.loads((RESULTS / 'ceil.json').read_text())
        for mutation in self.CAUGHT_BY_REPLAY:
            with self.subTest(mutation=mutation):
                r = run('ceil', '--mutation', mutation, '--verify', RESULTS / 'ceil.json')
                self.assertEqual(r.returncode, 1, r.stdout[-2000:])
                self.assertIn('VERIFY_FAIL', summary(r))
                self.assertIn('cells', summary(r))
                cells, _ = load_program().parse_certified(r.stdout)
                self.assertEqual(len(cells), len(reviewed['cells']))
                self.assertTrue(any(c['bound'] != e['bound'] for c, e in zip(cells, reviewed['cells'])), mutation)

    def test_a_mutated_run_is_never_reported_as_a_pass(self):
        r = run('ceil', '--mutation', 'drop_fail_piece')
        self.assertEqual(r.returncode, 3, r.stdout[-2000:])
        self.assertIn('NOT A CERTIFICATE', summary(r))
        self.assertFalse(any(ln.startswith('H3_BAND_V2 PASS') for ln in r.stdout.splitlines()))

    def test_record_is_refused_under_a_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'r.json'
            r = run('ceil', '--mutation', 'drop_fail_piece', '--record', p)
            self.assertFalse(p.exists())
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn('--record is refused under --mutation', summary(r))


if __name__ == '__main__':
    unittest.main()
