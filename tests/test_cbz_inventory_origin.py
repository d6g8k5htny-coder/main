"""Adverse controls for inventory mutation-failure provenance."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cbz_inventory_origin as origin


FIXTURE = '''
import os
import unittest

def subject():
    raise AssertionError('subject assertion')

class Fixture(unittest.TestCase):
    def helper(self):
        raise AssertionError('helper assertion')
    def test_pass(self):
        self.assertEqual(2, 2)
    def test_oracle(self):
        self.assertEqual(1, 2)
    def test_direct(self):
        raise AssertionError('direct oracle')
    def test_raises_oracle(self):
        with self.assertRaises(ValueError):
            pass
    def test_subtest_oracle(self):
        with self.subTest(n=1):
            self.fail('oracle subtest')
    def test_subject(self):
        subject()
    def test_helper(self):
        self.helper()
    def test_nested(self):
        def callback():
            raise AssertionError('nested assertion')
        callback()
    def test_subtest_subject(self):
        with self.subTest(n=1):
            subject()
    def test_error(self):
        raise RuntimeError('runtime error')
    def test_skip(self):
        self.skipTest('unexpected')
    async def test_coroutine(self):
        self.fail('deferred')
    def test_generator(self):
        yield None
    def test_spoof_stdout(self):
        os.write(1, b'not a valid child report')
        self.fail('oracle')

class Setup(unittest.TestCase):
    def setUp(self):
        raise AssertionError('setup')
    def test_it(self):
        pass

class Teardown(unittest.TestCase):
    def tearDown(self):
        raise AssertionError('teardown')
    def test_it(self):
        pass

class Cleanup(unittest.TestCase):
    def test_it(self):
        def cleanup():
            raise AssertionError('cleanup')
        self.addCleanup(cleanup)

class DoubleFailure(unittest.TestCase):
    def tearDown(self):
        raise AssertionError('teardown')
    def test_it(self):
        self.fail('oracle')

class ExpectedFailure(unittest.TestCase):
    @unittest.expectedFailure
    def test_it(self):
        self.fail('expected')

class UnexpectedSuccess(unittest.TestCase):
    @unittest.expectedFailure
    def test_it(self):
        pass

class Mixed(unittest.TestCase):
    def test_a_pass(self):
        pass
    def test_b_fail(self):
        self.assertEqual(1, 2)
    @unittest.skip('declared recursion guard')
    def test_c_controller(self):
        pass

class MixedSubject(unittest.TestCase):
    def test_a_oracle(self):
        self.fail('selected oracle')
    def test_b_subject(self):
        subject()
'''


class InventoryOriginControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'origin_fixture.py').write_text(textwrap.dedent(FIXTURE))

    def selected(self, class_name, name):
        identity = 'origin_fixture.' + class_name + '.' + name
        return identity, origin.run(self.root, identity, [identity], identity,
                                    env=os.environ.copy(), timeout=20)

    def test_passing_baseline_and_selected_oracles(self):
        _, passing = self.selected('Fixture', 'test_pass')
        self.assertEqual(passing['classification'], 'baseline', passing)
        for name in ('test_oracle', 'test_direct', 'test_raises_oracle', 'test_subtest_oracle'):
            with self.subTest(name=name):
                _, result = self.selected('Fixture', name)
                self.assertEqual(result['classification'], 'kill', result)

    def test_subject_helper_nested_and_non_oracle_failures_are_refused(self):
        for cls, name in (('Fixture', 'test_subject'), ('Fixture', 'test_helper'),
                          ('Fixture', 'test_nested'), ('Fixture', 'test_subtest_subject'),
                          ('Fixture', 'test_error'), ('Fixture', 'test_skip'),
                          ('Fixture', 'test_coroutine'), ('Fixture', 'test_generator'),
                          ('Fixture', 'test_spoof_stdout'),
                          ('Setup', 'test_it'), ('Teardown', 'test_it'),
                          ('Cleanup', 'test_it'), ('DoubleFailure', 'test_it'),
                          ('ExpectedFailure', 'test_it'), ('UnexpectedSuccess', 'test_it')):
            with self.subTest(cls=cls, name=name):
                _, result = self.selected(cls, name)
                self.assertEqual(result['classification'], 'invalid', result)

    def test_full_suite_requires_only_declared_skips_and_authorized_origins(self):
        ids = ['origin_fixture.Mixed.test_a_pass', 'origin_fixture.Mixed.test_b_fail',
               'origin_fixture.Mixed.test_c_controller']
        skips = {ids[2]: 'declared recursion guard'}
        report = origin.run(self.root, 'origin_fixture.Mixed', ids, ids[1],
                            allowed_skips=skips, env=os.environ.copy(), timeout=20)
        self.assertEqual(report['classification'], 'kill', report)
        self.assertEqual(origin.classify(report, ids, ids[1], {}), 'invalid')
        self.assertEqual(origin.classify(report, ids, ids[0], skips), 'invalid')
        retargeted = copy.deepcopy(report)
        retargeted['selected_id'] = ids[0]
        self.assertEqual(origin.classify(retargeted, ids, ids[0], skips), 'invalid')
        other = ['origin_fixture.MixedSubject.test_a_oracle',
                 'origin_fixture.MixedSubject.test_b_subject']
        result = origin.run(self.root, 'origin_fixture.MixedSubject', other, other[0],
                            env=os.environ.copy(), timeout=20)
        self.assertEqual(result['classification'], 'invalid', result)

    def test_missing_extra_and_forged_protocol_fields_fail_closed(self):
        identity, report = self.selected('Fixture', 'test_oracle')
        self.assertEqual(report['classification'], 'kill', report)
        for key, value in (('events', []), ('error_count', True), ('tests_run', 2),
                           ('tests_run', True),
                           ('child_exit', 0), ('selected_id', 'wrong'),
                           ('protocol_error', True),
                           ('started', []), ('stopped', []), ('schema', 'other')):
            with self.subTest(key=key):
                forged = copy.deepcopy(report)
                forged[key] = value
                self.assertEqual(origin.classify(forged, [identity], identity, {}), 'invalid')
        forged = copy.deepcopy(report)
        forged['events'][0]['authorized'] = 'true'
        self.assertEqual(origin.classify(forged, [identity], identity, {}), 'invalid')
        forged = copy.deepcopy(report)
        forged['events'][0]['frames'] = []
        self.assertEqual(origin.classify(forged, [identity], identity, {}), 'invalid')
        forged = copy.deepcopy(report)
        forged['succeeded'] = [[]]
        self.assertEqual(origin.classify(forged, [identity], identity, {}), 'invalid')


if __name__ == '__main__':
    unittest.main()
