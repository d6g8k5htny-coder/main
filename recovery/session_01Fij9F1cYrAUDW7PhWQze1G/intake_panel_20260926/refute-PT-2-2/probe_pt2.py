"""PT-2 re-run. Each probe PASSES when it demonstrates the stated behaviour on the copy ROOT points at."""
import unittest
from harness import intake, REPO, IntakeTests


def serve_pages(self, n, prefix='registers/json/item', status='modified'):
    rows = [{'filename': f'{prefix}{i}.json', 'status': status, 'sha': 'd' * 40} for i in range(n)]
    for page in range(1, (n + 99) // 100 + 1):
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page={page}'] = rows[(page-1)*100:page*100]
    self.pr['changed_files'] = n
    return rows


class F2_AsFiled(IntakeTests):
    """Finder's probe verbatim: same-repo, no label, 501 modified rows over 6 pages."""
    def test_501_file_maintainer_pr_rejected(self):
        self.pr['head']['repo']['full_name'] = REPO
        serve_pages(self, 501)
        self.reject('unsupported changed file count')


class Companion_500_is_accepted(IntakeTests):
    """Boundary: exactly MAX_FILES rows are accepted on both copies (the cap is inclusive)."""
    def test_500_file_maintainer_pr_accepted(self):
        self.pr['head']['repo']['full_name'] = REPO
        serve_pages(self, 500)
        self.assertEqual(self.run_check()['lane'], 'maintainer-engineering')


class Companion_FindersOwnExample_3880(IntakeTests):
    """The finder cites a 3879/3880-file commit. Under EITHER cap such a PR is rejected:
    the API ceiling is 3000 and the patch only moves the constant to 3000."""
    def test_3880_file_maintainer_pr_rejected_on_both_copies(self):
        self.pr['head']['repo']['full_name'] = REPO
        serve_pages(self, 3880)
        self.reject('unsupported changed file count')


class Companion_OutsiderAmplification(IntakeTests):
    """Cost of an outsider-triggered run: a fork PR whose changed_files equals the cap.
    Reports the number of API GETs spent before rejection; asserts the shape only."""
    def test_fork_pr_at_cap_api_calls(self):
        cap = intake.MAX_FILES
        serve_pages(self, cap, prefix='incoming/x/f', status='added')
        self.reject('too many package files')
        pages = sum('/files?' in c for c in self.api.calls)
        print(f'\n[MAX_FILES={cap}] fork PR with {cap} files: {len(self.api.calls)} API GETs, {pages} file-listing pages before rejection')
        self.assertEqual(pages, cap // 100)

    def test_fork_pr_above_cap_api_calls(self):
        cap = intake.MAX_FILES
        serve_pages(self, cap + 1, prefix='incoming/x/f', status='added')
        self.reject('unsupported changed file count')
        print(f'\n[MAX_FILES={cap}] fork PR with {cap+1} files: {len(self.api.calls)} API GET(s) before rejection')
        self.assertEqual(len(self.api.calls), 1)


if __name__ == '__main__':
    unittest.main()
