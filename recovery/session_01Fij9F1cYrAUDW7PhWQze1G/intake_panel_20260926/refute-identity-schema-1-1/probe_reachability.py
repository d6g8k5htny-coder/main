"""FINDING identity-schema-1 probe: a pinned source commit is never checked for reachability from
any branch of the pillar repository. GitHub serves commit/tree/blob objects for any SHA in the
repository's object store, including refs/pull/N/head of an outsider's PR and fork-network objects.
The checker itself relies on exactly that to read the fork PR's head tree from the base repository.
On the CURRENT checker every test here PASSES = the wrong green is demonstrated."""
import json, os, unittest
from harness import Base, intake, REPO, HEAD, BASE, sha

PARENT = 'f' * 40      # an ancestor of the PR head, authored by the submitter in their fork
MATH_PR_HEAD = 'e' * 40  # head of a throwaway, never-merged PR the submitter opened against Math-
EXPECT_GREEN = os.environ.get('EXPECT', 'green') == 'green'


class ReachabilityProbe(Base):
    def expect(self, message):
        if EXPECT_GREEN:
            result = self.run_check()
            self.assertEqual(result['lane'], 'incoming')
            self.assertEqual(result['verified_sources'], 1)
            self.assertEqual(self.reachability_routes_requested(), [], 'checker asked no reachability question')
        else:
            self.reject(message)

    def test_fabricated_source_in_pr_head_ancestor(self):
        forged = b'{"enclosure": [0, 0], "note": "bytes chosen by the submitter, never in main"}\n'
        self.manifest['sources'] = [{'repository': REPO, 'path': 'coefficients/side24_v1/ENCLOSURE.json',
                                     'commit': PARENT, 'sha256': sha(forged)}]
        self.save()
        # refs/pull/7/head brings PARENT into d6g8k5htny-coder/main's object store; no branch contains it.
        self.install_source(REPO, PARENT, 'coefficients/side24_v1/ENCLOSURE.json', forged, reachable_from=())
        self.expect('not reachable')

    def test_self_citation_of_own_package_at_pr_head(self):
        own = self.raw['output.json']
        self.manifest['sources'] = [{'repository': REPO, 'path': 'incoming/example/output.json', 'commit': HEAD, 'sha256': sha(own)}]
        self.save()
        # HEAD's tree and blob routes already exist (the checker reads them for the package itself).
        self.api.routes[f'/repos/{REPO}'] = {'private': False, 'full_name': REPO, 'default_branch': 'main'}
        self.api.routes[f'/repos/{REPO}/git/commits/{HEAD}'] = {'sha': HEAD}
        self.api.routes[f'/repos/{REPO}/branches?per_page=100&page=1'] = [{'name': 'main', 'commit': {'sha': BASE}}]
        self.api.routes[f'/repos/{REPO}/compare/main...{HEAD}?per_page=1'] = {'status': 'ahead', 'ahead_by': 1, 'behind_by': 0, 'merge_base_commit': {'sha': BASE}}
        self.expect('not reachable')

    def test_unmerged_math_pull_request_head_cited_as_math_source(self):
        forged = b'fabricated coefficient table\n'
        repo = 'd6g8k5htny-coder/Math-'
        self.manifest['sources'] = [{'repository': repo, 'path': 'coefficients/forged.json', 'commit': MATH_PR_HEAD, 'sha256': sha(forged)}]
        self.save()
        self.install_source(repo, MATH_PR_HEAD, 'coefficients/forged.json', forged, reachable_from=())
        self.expect('not reachable')

    def test_positive_control_reachable_from_research_branch_is_accepted(self):
        # The documented main inventory snapshot 7caac254 is on a research branch, not on main.
        repo = REPO
        raw = b'public numbers on a research branch\n'
        self.manifest['sources'] = [{'repository': repo, 'path': 'research/numbers.json', 'commit': 'd' * 40, 'sha256': sha(raw)}]
        self.save()
        self.install_source(repo, 'd' * 40, 'research/numbers.json', raw, reachable_from=('chatgpt/drive-github-hardening-20260919',),
                            branches=('main', 'chatgpt/drive-github-hardening-20260919'))
        result = self.run_check()
        self.assertEqual(result['verified_sources'], 1)


if __name__ == '__main__':
    unittest.main()
