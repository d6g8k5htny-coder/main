"""Skeptic #2 independent probe (lens identity-schema): built directly on the repository's own
tests/test_public_intake.py fixture (IntakeTests), not on the finder's harness.
EXPECT=green: assert the fabricated-source cases are ACCEPTED (defect present).
EXPECT=closed: assert they are REFUSED with a reachability message (defect absent)."""
import os, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_public_intake as t

EXPECT_GREEN = os.environ.get('EXPECT', 'green') == 'green'
PARENT = 'f' * 40


class SkepticProbe(t.IntakeTests):
    def serve_object_store_only(self, repo, commit, path, raw):
        """Only what GitHub's git-database serves for an object in the store: metadata, commit, tree, blob.
        Also serve the reachability routes a fixed checker would ask, answering 'not on any branch'."""
        self.api.routes[f'/repos/{repo}'] = {'private': False, 'full_name': repo, 'default_branch': 'main'}
        self.api.routes[f'/repos/{repo}/git/commits/{commit}'] = {'sha': commit}
        self.api.routes[f'/repos/{repo}/git/trees/{commit}?recursive=1'] = {'truncated': False, 'tree': [
            {'path': path, 'mode': '100644', 'type': 'blob', 'size': len(raw), 'sha': t.blob(raw)}]}
        self.api.routes[f'/repos/{repo}/git/blobs/{t.blob(raw)}'] = {
            'encoding': 'base64', 'size': len(raw), 'content': __import__('base64').b64encode(raw).decode(), 'sha': t.blob(raw)}
        self.api.routes[f'/repos/{repo}/branches?per_page=100&page=1'] = [{'name': 'main', 'commit': {'sha': t.BASE}}]
        self.api.routes[f'/repos/{repo}/compare/main...{commit}?per_page=1'] = {
            'status': 'diverged', 'ahead_by': 1, 'behind_by': 4, 'merge_base_commit': {'sha': t.BASE}}

    def expect(self):
        if EXPECT_GREEN:
            result = self.run_check()
            self.assertEqual(result['lane'], 'incoming')
            self.assertEqual(result['verified_sources'], 1)
            self.assertFalse(any('/compare/' in c or '/branches' in c for c in self.api.calls), 'no reachability question was asked')
        else:
            self.reject('not reachable')

    def test_ancestor_of_fork_pr_head_cited_as_main_source(self):
        forged = b'{"forged": "outsider bytes at a path main never held"}\n'
        self.manifest['sources'] = [{'repository': t.REPO, 'path': 'coefficients/side24_v1/ENCLOSURE.json', 'commit': PARENT, 'sha256': t.sha(forged)}]
        self.save()
        self.serve_object_store_only(t.REPO, PARENT, 'coefficients/side24_v1/ENCLOSURE.json', forged)
        self.expect()

    def test_pr_head_self_citation(self):
        own = self.raw['output.json']
        self.manifest['sources'] = [{'repository': t.REPO, 'path': self.package + 'output.json', 'commit': t.HEAD, 'sha256': t.sha(own)}]
        self.save()
        # The repository fixture's install() overwrites the HEAD tree route when a source commit equals HEAD;
        # restore the package tree GitHub actually serves for the PR head, then add metadata/commit/reachability.
        self.api.routes[f'/repos/{t.REPO}/git/trees/{t.HEAD}?recursive=1'] = {'truncated': False, 'tree': [
            {'path': self.package + name, 'type': 'blob', 'mode': '100644', 'sha': t.blob(raw), 'size': len(raw)} for name, raw in self.files.items()]}
        self.api.routes[f'/repos/{t.REPO}'] = {'private': False, 'full_name': t.REPO, 'default_branch': 'main'}
        self.api.routes[f'/repos/{t.REPO}/git/commits/{t.HEAD}'] = {'sha': t.HEAD}
        self.api.routes[f'/repos/{t.REPO}/branches?per_page=100&page=1'] = [{'name': 'main', 'commit': {'sha': t.BASE}}]
        self.api.routes[f'/repos/{t.REPO}/compare/main...{t.HEAD}?per_page=1'] = {'status': 'ahead', 'ahead_by': 1, 'behind_by': 0, 'merge_base_commit': {'sha': t.BASE}}
        self.expect()

    def test_unmerged_math_pr_head_cited_as_math_source(self):
        forged = b'fabricated coefficient table\n'
        repo = 'd6g8k5htny-coder/Math-'
        self.manifest['sources'] = [{'repository': repo, 'path': 'coefficients/forged.json', 'commit': 'e' * 40, 'sha256': t.sha(forged)}]
        self.save()
        self.serve_object_store_only(repo, 'e' * 40, 'coefficients/forged.json', forged)
        self.expect()


del t.IntakeTests  # run only the probe cases, not the whole inherited suite twice

if __name__ == '__main__':
    unittest.main()
