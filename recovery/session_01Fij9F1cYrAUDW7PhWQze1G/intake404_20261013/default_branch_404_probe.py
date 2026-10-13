"""Probe run on 2026-10-13 against main#391 head 66776948 (posted as data at main#391 comment 6154187346).
Run from a checkout of that head: python3 -B -S default_branch_404_probe.py <checkout>/tests"""
import sys, urllib.error, unittest
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else 'tests')
import test_public_intake as t
class Probe(t.IntakeTests):
    def runTest(self):
        # The default branch itself answers 404 (commit has unrelated history with main), and no other branch vouches.
        route = f'/repos/{t.MATH}/compare/main...{t.SOURCE}?per_page=1'
        self.api.routes[route] = urllib.error.HTTPError('u', 404, 'Not Found', None, None)
        self.reject('not reachable from any branch')
        # Same, but a second branch positively vouches: accepted only through that branch's evidence.
        self.api.routes[f'/repos/{t.MATH}/branches?per_page=100&page=1'] = [{'name': 'main'}, {'name': 'research'}]
        self.api.routes[f'/repos/{t.MATH}/compare/research...{t.SOURCE}?per_page=1'] = t.compare(t.SOURCE)
        self.assertEqual(self.run_check()['verified_sources'], 1)
r = unittest.TextTestRunner(verbosity=0).run(Probe())
print('default-branch probe:', 'OK' if r.wasSuccessful() else 'FAIL')
