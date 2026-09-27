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

    def test_personal_names(self):
        for name in ('x/my_resume.docx', 'Résumé 2025.pdf', 'tax_return.pdf', 'W2_2025.pdf',
                     'bank statement.pdf', 'drivers license.jpg'):
            self.assertTrue(dr.personal_name(name), name)
        for name in ('LICENSE', 'syntax.md', 'arm1c v2 instr.py', 'cvx_check.py', 'C104_E_LEDGER.md',
                     'RECEIPT.json'):
            self.assertFalse(dr.personal_name(name), name)


def quiet(argv):
    old, sys.stdout = sys.stdout, io.StringIO()
    try:
        dr.main(argv)
        return json.loads(sys.stdout.getvalue())
    finally:
        sys.stdout = old


class StageAndPack(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        p = self.p = Path(self.d.name)
        store, text = p / 'store', p / 'text'
        store.mkdir()
        text.mkdir()
        loose = b'# note\nsame bytes as the zip member\n'
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, 'w') as z:
            z.writestr('pack/note.md', loose)
        blobs = {'id:z': ('pack.zip', zbuf.getvalue()), 'id:n': ('pack/note.md', loose),
                 'id:r': ('resume_2025.txt', b'name, phone\n'), 'id:t': ('theorem.md', b'# Theorem\n'),
                 'id:big': ('big.txt', os.urandom(5000))}
        rows, catalog = [], []
        for fid, (path, data) in blobs.items():
            sha = hashlib.sha256(data).hexdigest()
            (store / sha).write_bytes(data)
            ex, tx, members = dr.extract_bytes(path, data)
            if tx:
                (text / (sha + '.txt')).write_text(tx)
            rows.append({'id': fid, 'path': path, 'size': len(data), 'status': 'MISSING'})
            catalog.append({'id': fid, 'path': path, 'sha256': sha, 'dbx': dr.dbx_hash_bytes(data),
                            'bytes': len(data), 'extractor': ex, 'chars': len(tx), 'members': members})
        rows.append({'id': 'id:g', 'path': 'in_git.md', 'size': 3, 'status': 'GIT_EXACT'})
        (p / 'cls.json').write_text(json.dumps(rows))
        (p / 'cat.json').write_text(json.dumps(catalog))
        self.stage = quiet(['stage', '--classified', str(p / 'cls.json'), '--catalog', str(p / 'cat.json'),
                            '--store', str(store), '--text-dir', str(text), '--github-out', str(p / 'gh'),
                            '--drive-out', str(p / 'drive')])
        self.uploads = {u['dropbox_path']: u for u in json.loads((p / 'drive' / 'UPLOADS.json').read_text())}

    def tearDown(self):
        self.d.cleanup()

    def test_stage_flags_zip_members_and_personal_names(self):
        self.assertEqual(set(self.uploads), {'pack.zip', 'pack/note.md', 'resume_2025.txt', 'theorem.md',
                                             'big.txt'})  # GIT_EXACT is never uploaded
        self.assertEqual(self.uploads['pack/note.md'].get('covered_by_zip'), 'pack.zip')
        self.assertNotIn('covered_by_zip', self.uploads['theorem.md'])
        self.assertTrue(self.uploads['resume_2025.txt'].get('personal_name'))
        staged = {a['from_dropbox'] for a in json.loads((self.p / 'gh' / 'PACKET_ARTIFACTS.json').read_text())}
        self.assertIn('theorem.md', staged)
        self.assertNotIn('resume_2025.txt', staged)

    def test_pack_holds_and_round_trips(self):
        res = quiet(['pack', '--uploads', str(self.p / 'drive' / 'UPLOADS.json'), '--out-dir', str(self.p / 'packs'),
                     '--part-bytes', '1000', '--max-file-bytes', '4096'])
        man = json.loads((self.p / 'packs' / 'dropbox_import_MANIFEST.json').read_text())
        held = {h['dropbox_path']: h['reason'] for h in man['held']}
        self.assertEqual(held, {'pack/note.md': 'covered_by_zip', 'resume_2025.txt': 'personal_name',
                                'big.txt': 'over_max_file_bytes'})
        self.assertEqual(res['members'], 2)
        for m in man['members']:
            with zipfile.ZipFile(self.p / 'packs' / m['part']) as z:
                self.assertEqual(hashlib.sha256(z.read(m['dropbox_path'])).hexdigest(), m['sha256'])

    def test_pack_splits_parts_at_the_cap(self):
        ups = [{'dropbox_path': f'f{i}.bin', 'sha256': '', 'bytes': 3000, 'source': str(self.p / f'f{i}.bin')}
               for i in range(3)]
        for u in ups:
            Path(u['source']).write_bytes(os.urandom(3000))  # incompressible
        (self.p / 'u.json').write_text(json.dumps(ups))
        res = quiet(['pack', '--uploads', str(self.p / 'u.json'), '--out-dir', str(self.p / 'p2'),
                     '--part-bytes', '4000'])
        self.assertEqual(res['parts'], 3)

    def test_report(self):
        res = quiet(['report', '--classified', str(self.p / 'cls.json'), '--out', str(self.p / 'r.md')])
        body = (self.p / 'r.md').read_text()
        self.assertEqual(res['missing'], 5)
        self.assertEqual(res['personal_name'], 1)
        self.assertIn('| GIT_EXACT | 1 |', body)
        self.assertIn('`resume_2025.txt`', body)


if __name__ == '__main__':
    unittest.main()
