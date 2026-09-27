"""Tests for tools/dropbox_reconcile.py. Scientific effect: NONE."""
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import dropbox_reconcile as dr  # noqa: E402

# A 185-byte Dropbox file whose content_hash was reported by the Dropbox API.
README = (b"\n# CHPPF v1.0.1 (Release Candidate)\n\nQuick start:\n"
          b"- Open `notebooks/00_reproduce_Table13.ipynb`\n"
          b"- Open `notebooks/01_compute_phi_omega_te.ipynb`\n\n"
          b"Generated: 2025-09-23T01:37:29.689679Z\n")
README_DBX = '458d93bf5275d150aba12dd4742c69f287374622a68bf5e83c28906e91cfb296'


class ContentHash(unittest.TestCase):
    def test_known_dropbox_value(self):
        self.assertEqual(len(README), 185)
        self.assertEqual(dr.dbx_hash_bytes(README), README_DBX)

    def test_one_byte_change_is_detected(self):
        self.assertNotEqual(dr.dbx_hash_bytes(README + b'\n'), README_DBX)

    def test_single_block_is_double_sha256(self):
        data = os.urandom(1000)
        self.assertEqual(dr.dbx_hash_bytes(data),
                         hashlib.sha256(hashlib.sha256(data).digest()).hexdigest())

    def test_multi_block_file_matches_bytes(self):
        data = os.urandom(dr.BLOCK * 2 + 5)
        with tempfile.NamedTemporaryFile(delete=False) as fh:
            fh.write(data)
        try:
            self.assertEqual(dr.dbx_hash_file(Path(fh.name)), dr.dbx_hash_bytes(data))
            self.assertNotEqual(dr.dbx_hash_bytes(data),
                                hashlib.sha256(hashlib.sha256(data).digest()).hexdigest())
        finally:
            os.unlink(fh.name)


class Classify(unittest.TestCase):
    def run_classify(self, inventory, git, drive):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'inv.json').write_text(json.dumps(inventory))
            (p / 'git.json').write_text(json.dumps(git))
            (p / 'drive.json').write_text(json.dumps(drive))
            buf = io.StringIO()
            old, sys.stdout = sys.stdout, buf
            try:
                dr.main(['classify', '--inventory', str(p / 'inv.json'), '--git-index', str(p / 'git.json'),
                         '--drive-index', str(p / 'drive.json'), '--out', str(p / 'out.json')])
            finally:
                sys.stdout = old
            return {r['path']: r['status'] for r in json.loads((p / 'out.json').read_text())}

    def entry(self, path, size, fid, h=None):
        e = {'object_type': 'file', 'name': path.rsplit('/', 1)[-1], 'path': 'ns:1//' + path,
             'file_id': fid, 'file': {'size': size}}
        if h:
            e['content_hash'] = h
        return e

    def test_tiers(self):
        sha = hashlib.sha256(README).hexdigest()
        drive_row = {'name': 'renamed.md', 'size': 185, 'sha256': sha, 'where': 'drive:x',
                     'dbx': hashlib.sha256(bytes.fromhex(sha)).hexdigest()}
        git = {'r:1': {'repo': 'r', 'blob': '1', 'size': 10, 'sha256': 'a' * 64, 'dbx': 'b' * 64,
                       'paths': ['docs/a.txt']}}
        inv = [self.entry('Notes.app/Contents/x.nib', 5, 'id:1'),
               self.entry('a.txt', 10, 'id:2'),
               self.entry('x/README.md', 185, 'id:3', README_DBX),
               self.entry('y/README.md', 185, 'id:4', README_DBX),
               self.entry('z/new.md', 7, 'id:5'),
               self.entry('z/new_copy/new.md', 7, 'id:6')]
        got = self.run_classify(inv, git, [drive_row])
        self.assertEqual(got['Notes.app/Contents/x.nib'], 'SKIP_SOFTWARE_BUNDLE')
        self.assertEqual(got['a.txt'], 'GIT_NAME_SIZE')
        self.assertEqual(got['x/README.md'], 'DRIVE_EXACT')  # exact despite a different Drive name
        self.assertEqual(got['y/README.md'], 'DRIVE_EXACT')
        self.assertEqual(got['z/new.md'], 'MISSING')
        self.assertEqual(got['z/new_copy/new.md'], 'DUP_IN_DROPBOX')

    def test_wrong_size_is_not_a_match(self):
        git = {'r:1': {'repo': 'r', 'blob': '1', 'size': 11, 'sha256': 'a' * 64, 'dbx': 'b' * 64,
                       'paths': ['a.txt']}}
        got = self.run_classify([self.entry('a.txt', 10, 'id:2')], git, [])
        self.assertEqual(got['a.txt'], 'MISSING')


class Harvest(unittest.TestCase):
    def test_links_and_hashes_from_transcript(self):
        payload = {'entries': [{'id': 'id:A', 'path_display': '/x.md', 'download_url': 'https://h/1',
                                'content_hash': README_DBX, 'size': 185}]}
        lines = [json.dumps({'timestamp': '2026-01-01T00:00:00Z',
                             'message': {'content': [{'type': 'tool_result',
                                                      'content': [{'type': 'text', 'text': json.dumps(payload)}]}]}}),
                 json.dumps({'timestamp': '2026-01-02T00:00:00Z', 'message': {'content': 'unrelated'}})]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 't.jsonl').write_text('\n'.join(lines))
            old, sys.stdout = sys.stdout, io.StringIO()
            try:
                dr.main(['harvest-links', '--transcript', str(p / 't.jsonl'), '--since', '2026-01-01T12:00:00Z',
                         '--urls-out', str(p / 'u.json'), '--hashes-out', str(p / 'h.json')])
            finally:
                sys.stdout = old
            self.assertEqual(json.loads((p / 'h.json').read_text()), {'id:A': README_DBX})
            self.assertEqual(json.loads((p / 'u.json').read_text()), [])  # expired: issued before --since


class Extract(unittest.TestCase):
    def test_docx_and_zip_members(self):
        doc = io.BytesIO()
        with zipfile.ZipFile(doc, 'w') as z:
            z.writestr('word/document.xml', '<w:document><w:p><w:t>Theorem &amp; proof</w:t></w:p></w:document>')
        ex, text, _ = dr.extract_bytes('a.docx', doc.getvalue())
        self.assertEqual(ex, 'docx')
        self.assertIn('Theorem & proof', text)
        outer = io.BytesIO()
        with zipfile.ZipFile(outer, 'w') as z:
            z.writestr('inner/a.md', '# A')
            z.writestr('inner/b.docx', doc.getvalue())
        ex, text, members = dr.extract_bytes('pack.zip', outer.getvalue())
        self.assertEqual(ex, 'zip')
        self.assertEqual({m['member'] for m in members}, {'inner/a.md', 'inner/b.docx'})
        self.assertEqual(next(m for m in members if m['member'] == 'inner/a.md')['dbx'],
                         dr.dbx_hash_bytes(b'# A'))

    def test_url_and_json(self):
        self.assertEqual(dr.extract_bytes('x.url', b'[InternetShortcut]\r\nURL=https://example.org/a\r\n')[1],
                         'https://example.org/a')
        self.assertEqual(dr.extract_bytes('x.json', b'{ "a" : 1 }')[1], '{"a":1}')

    def test_pii_patterns_flag_credentials(self):
        self.assertTrue(any(p.search('api_key = abc') for p in dr.PII))
        self.assertFalse(any(p.search('Theorem R at scope') for p in dr.PII))


if __name__ == '__main__':
    unittest.main()
