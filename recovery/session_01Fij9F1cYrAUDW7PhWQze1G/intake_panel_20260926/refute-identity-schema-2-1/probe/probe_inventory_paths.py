"""FINDING identity-schema-2 probe (MINOR, wrong rejection): safe_path() is applied to SOURCE paths,
so 1,894 of the 2,138 rows of the public inventory CONTRIBUTING tells contributors to cite
(docs/public-math/sources-*.json) cannot be pinned. This is a real inventory row with its real
bytes (taken from the local object store at 7caac254). The source path is only used as a dict key
into the commit tree; it never enters a URL."""
import base64, json, os, unittest
from pathlib import Path
from harness import Base, intake, REPO, sha, blob

ROW = json.load(open(Path(__file__).resolve().parent / 'inventory_row.json'))
EXPECT_GREEN = os.environ.get('EXPECT', 'green') == 'green'


class InventoryPathProbe(Base):
    def test_documented_inventory_row_is_pinnable(self):
        raw = base64.b64decode(ROW['content_b64'])
        assert sha(raw) == ROW['sha256'] and blob(raw) == ROW['blob'] and len(raw) == ROW['bytes']
        self.manifest['sources'] = [{'repository': ROW['repository'], 'path': ROW['path'], 'commit': ROW['commit'], 'sha256': ROW['sha256']}]
        self.save()
        self.install_source(ROW['repository'], ROW['commit'], ROW['path'], raw,
                            reachable_from=('chatgpt/drive-github-hardening-20260919',),
                            branches=('main', 'chatgpt/drive-github-hardening-20260919'))
        if EXPECT_GREEN:
            # current checker: refused before the source is even looked up
            self.reject('unsafe path')
            self.assertFalse(any('/git/commits/' in c for c in self.api.calls))
        else:
            self.assertEqual(self.run_check()['verified_sources'], 1)


if __name__ == '__main__':
    unittest.main()
