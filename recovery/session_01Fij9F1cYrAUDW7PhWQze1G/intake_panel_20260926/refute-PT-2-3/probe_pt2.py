"""Skeptic re-run of finding PT-2 against a fresh copy of the checker (ROOT selects fresh/ or patched/). No network."""
import importlib.util, os, sys, unittest
from pathlib import Path
ROOT = Path(os.environ.get('ROOT', Path(__file__).resolve().parent / 'fresh'))
_spec = importlib.util.spec_from_file_location('test_public_intake', ROOT / 'tests/test_public_intake.py')
T = importlib.util.module_from_spec(_spec); sys.modules['test_public_intake'] = T; _spec.loader.exec_module(T)
REPO = T.REPO

class PT2(T.IntakeTests):
    def _same_repo_rows(self, n, prefix='registers/json/item'):
        self.pr['head']['repo']['full_name'] = REPO
        self.pr['changed_files'] = n
        rows = [{'filename': f'{prefix}{i}.json', 'status': 'modified', 'sha': 'd' * 40} for i in range(n)]
        for page in range(1, (n + 99)//100 + 1):
            self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page={page}'] = rows[(page-1)*100:page*100]
        return rows

    def test_501_file_maintainer_pr_rejected(self):
        # Finder's probe, re-created: passes on fresh (rejection), fails on patched (lane granted).
        self._same_repo_rows(501)
        self.reject('unsupported changed file count')
        self.assertFalse(any('/files?' in c for c in self.api.calls), 'cap fires before any file page is fetched')

    def test_500_file_maintainer_pr_is_accepted_on_fresh(self):
        self._same_repo_rows(500)
        self.assertEqual(self.run_check()['lane'], 'maintainer-engineering')
        self.assertEqual(sum('/files?' in c for c in self.api.calls), 5)

    def test_finders_own_3879_file_example_rejected_even_when_patched(self):
        # The 3879-file history example the finding cites is above the API ceiling; the patch does not admit it.
        self._same_repo_rows(3879)
        self.reject('unsupported changed file count')

    def test_fork_pr_page_cost_bounded_by_MAX_FILES_not_MAX_PACKAGE_FILES(self):
        # Outsider (fork head) with N files: every page is fetched before the 50-file package cap applies.
        n = T.intake.MAX_FILES
        self.pr['changed_files'] = n
        rows = [{'filename': f'incoming/example/f{i}.txt', 'status': 'added', 'sha': 'd' * 40} for i in range(n)]
        for page in range(1, (n + 99)//100 + 1):
            self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page={page}'] = rows[(page-1)*100:page*100]
        self.reject('too many package files')
        pages = sum('/files?' in c for c in self.api.calls)
        print(f'\n[MAX_FILES={n}] outsider forced {pages} file-listing pages before the 50-file package cap fired')
        self.assertEqual(pages, (n + 99)//100)

if __name__ == '__main__':
    unittest.main()
