"""Real delivery archives and hostile controls; no network or payload execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile
import zlib


TOOL = Path(__file__).resolve().parents[1] / 'tools/research_delivery.py'
MANIFEST = 'DELIVERY_MANIFEST.json'


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.is_file(), 'delivery builder/verifier is not implemented')
        spec = importlib.util.spec_from_file_location('research_delivery', TOOL)
        self.delivery = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.delivery)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = {
            'protocol_version': 'OP-PROT-019-v1.1',
            'checkpoint_utc': '2026-09-30T03:00:00Z',
            'task_id': 'delivery-test', 'claim_id': 'test-claim',
            'stable_target_key': 'delivery-test-v1',
            'source_identities': [{'path': 'report.txt', 'sha256': digest(b'report\n')}],
            'output_head': 'a' * 40, 'verified_boundary': 'local packaging only',
            'next_action': 'nonauthor review; no scientific promotion',
        }
        self.raws = {'report.txt': b'report\n', 'nested/proof.bin': b'\x00\xff\n'}
        self.raws['HANDOFF.json'] = encoded({
            'payload': self.payload, 'payload_sha256': digest(encoded(self.payload))})
        for name, raw in self.raws.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        self.spec = {'version': 1, 'handoff': 'HANDOFF.json', 'files': [
            {'path': name, 'bytes': len(raw), 'sha256': digest(raw)}
            for name, raw in self.raws.items()]}
        self.spec_path = self.root / 'spec.json'
        self.archive = self.root / 'delivery.zip'

    def build(self, spec=None, output=None):
        self.spec_path.write_bytes(encoded(self.spec if spec is None else spec))
        return self.delivery.build(self.root, self.spec_path, output or self.archive)

    def rewrite(self, mutate):
        with zipfile.ZipFile(self.archive) as archive:
            entries = [(info, archive.read(info)) for info in archive.infolist()]
        entries = mutate(entries)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(self.archive, 'w') as archive:
                for info, raw in entries:
                    archive.writestr(info, raw)

    def test_round_trip_is_deterministic_and_never_autocrawls(self):
        (self.root / 'unlisted-secret.txt').write_text('must stay local')
        first = self.build()
        second_path = self.root / 'second.zip'
        spec = copy.deepcopy(self.spec)
        spec['files'].reverse()
        self.build(spec, second_path)
        self.assertEqual(self.archive.read_bytes(), second_path.read_bytes())
        checked = self.delivery.verify(self.archive)
        self.assertEqual(first['archive_sha256'], digest(self.archive.read_bytes()))
        self.assertEqual(first['archive_sha256'], checked['archive_sha256'])
        self.assertTrue(checked['ok'])
        with zipfile.ZipFile(self.archive) as archive:
            self.assertEqual(archive.namelist(), sorted([MANIFEST, *self.raws]))
            for info in archive.infolist():
                self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
            manifest = json.loads(archive.read(MANIFEST))
            self.assertEqual(manifest['files'], [
                {'filename': name, 'size': len(self.raws[name]),
                 'sha256': digest(self.raws[name])} for name in sorted(self.raws)])

    def test_changed_input_fails_before_output_is_created(self):
        (self.root / 'report.txt').write_bytes(b'report!')
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.archive.exists())

    def test_refuses_to_overwrite_an_existing_output(self):
        self.archive.write_bytes(b'keep me')
        with self.assertRaises((ValueError, FileExistsError)):
            self.build()
        self.assertEqual(self.archive.read_bytes(), b'keep me')

    def test_unsafe_or_duplicate_input_paths_are_rejected(self):
        for path in ('../report.txt', '/report.txt', 'nested/../report.txt',
                     'nested//proof.bin', './report.txt', 'C:/report.txt',
                     'nested\\proof.bin', 'bad\x00name', MANIFEST):
            with self.subTest(path=path):
                spec = copy.deepcopy(self.spec)
                spec['files'][0]['path'] = path
                with self.assertRaises(ValueError):
                    self.build(spec)
        spec = copy.deepcopy(self.spec)
        spec['files'].append(spec['files'][0])
        with self.assertRaises(ValueError):
            self.build(spec)

    def test_input_symlink_and_symlink_directory_are_rejected(self):
        original = self.root / 'report.txt'
        original.unlink()
        original.symlink_to(self.root / 'nested/proof.bin')
        with self.assertRaises((ValueError, OSError)):
            self.build()
        original.unlink()
        original.write_bytes(self.raws['report.txt'])
        (self.root / 'nested').rename(self.root / 'actual')
        (self.root / 'nested').symlink_to(self.root / 'actual', target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            self.build()

    def test_strict_control_json_rejects_duplicate_and_nonfinite_values(self):
        for bad in (b'{"version":1,"version":1}', b'{"x":NaN}',
                    b'{"x":Infinity}', b'{"x":1e400}', b'{"x":1.5}',
                    b'{"x":"\\ud800"}'):
            with self.subTest(raw=bad):
                self.spec_path.write_bytes(bad)
                with self.assertRaises(ValueError):
                    self.delivery.build(self.root, self.spec_path, self.archive)

    def test_required_handoff_fields_and_digest_are_checked(self):
        for field in (*self.payload, 'payload_sha256'):
            with self.subTest(field=field):
                payload = dict(self.payload)
                if field != 'payload_sha256':
                    del payload[field]
                envelope = {'payload': payload, 'payload_sha256': digest(encoded(payload))}
                if field == 'payload_sha256':
                    envelope['payload_sha256'] = '0' * 64
                raw = encoded(envelope)
                (self.root / 'HANDOFF.json').write_bytes(raw)
                spec = copy.deepcopy(self.spec)
                spec['files'][-1].update(bytes=len(raw), sha256=digest(raw))
                with self.assertRaises(ValueError):
                    self.build(spec)

    def test_structured_verified_boundary_remains_valid_r17(self):
        payload = dict(self.payload)
        payload['output_head'] = {'repository': 'example/research', 'commit': 'a' * 40}
        payload['checkpoint_utc'] = '2026-09-30T03:00:00.123456+00:00'
        payload['verified_boundary'] = {'local': ['byte custody'], 'scientific_effect': 'NONE'}
        raw = encoded({'payload': payload, 'payload_sha256': digest(encoded(payload))})
        (self.root / 'HANDOFF.json').write_bytes(raw)
        spec = copy.deepcopy(self.spec)
        spec['files'][-1].update(bytes=len(raw), sha256=digest(raw))
        self.build(spec)
        self.assertTrue(self.delivery.verify(self.archive)['ok'])

    def test_missing_or_unlisted_handoff_is_rejected(self):
        for change in ('missing', 'unlisted'):
            spec = copy.deepcopy(self.spec)
            if change == 'missing':
                del spec['handoff']
            else:
                spec['handoff'] = 'not-listed.json'
            with self.assertRaises(ValueError):
                self.build(spec)

    def test_oversized_inputs_and_metadata_are_rejected(self):
        spec = copy.deepcopy(self.spec)
        spec['files'][0]['bytes'] = self.delivery.MAX_FILE_BYTES + 1
        with self.assertRaises(ValueError):
            self.build(spec)
        with (self.root / 'report.txt').open('wb') as stream:
            stream.truncate(self.delivery.MAX_FILE_BYTES + 1)
        with self.assertRaises(ValueError):
            self.build()
        self.spec_path.write_bytes(b' ' * (self.delivery.MAX_JSON_BYTES + 1))
        with self.assertRaises(ValueError):
            self.delivery.build(self.root, self.spec_path, self.archive)

    def test_missing_extra_duplicate_and_unsafe_zip_members_are_rejected(self):
        mutations = [
            lambda es: [(i, b) for i, b in es if i.filename != 'report.txt'],
            lambda es: es + [('extra.txt', b'extra')],
            lambda es: es + [('report.txt', b'report\n')],
            lambda es: es + [('../escape.txt', b'extra')],
            lambda es: [(i, b) for i, b in es if i.filename != MANIFEST],
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.build()
                self.rewrite(mutate)
                with self.assertRaises(ValueError):
                    self.delivery.verify(self.archive)
                self.archive.unlink()

    def test_modified_bytes_and_manifest_self_hash_are_rejected(self):
        self.build()
        self.rewrite(lambda es: [(i, b'report!' if i.filename == 'report.txt' else b)
                                 for i, b in es])
        with self.assertRaises(ValueError):
            self.delivery.verify(self.archive)
        self.archive.unlink()
        self.build()
        def self_reference(es):
            changed = []
            for info, raw in es:
                if info.filename == MANIFEST:
                    manifest = json.loads(raw)
                    manifest['files'].append({'filename': MANIFEST, 'size': 1, 'sha256': '0' * 64})
                    raw = encoded(manifest)
                changed.append((info, raw))
            return changed
        self.rewrite(self_reference)
        with self.assertRaises(ValueError):
            self.delivery.verify(self.archive)

    def test_zip_symlink_and_duplicate_manifest_json_are_rejected(self):
        self.build()
        def symlink(es):
            for info, _ in es:
                if info.filename == 'report.txt':
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
            return es
        self.rewrite(symlink)
        with self.assertRaises(ValueError):
            self.delivery.verify(self.archive)
        self.archive.unlink()
        self.build()
        self.rewrite(lambda es: [(i, b'{"version":1,"version":1}'
                                  if i.filename == MANIFEST else b) for i, b in es])
        with self.assertRaises(ValueError):
            self.delivery.verify(self.archive)

    def test_crc_corruption_and_oversized_archive_are_rejected(self):
        self.build()
        raw = self.archive.read_bytes().replace(b'report\n', b'report!')
        self.archive.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.delivery.verify(self.archive)
        with self.archive.open('wb') as stream:
            stream.truncate(self.delivery.MAX_ARCHIVE_BYTES + 1)
        with self.assertRaises(ValueError):
            self.delivery.verify(self.archive)

    def test_bad_deflate_stream_is_a_clean_verification_failure(self):
        self.build()
        def compressed(es):
            for info, _ in es:
                info.compress_type = zipfile.ZIP_DEFLATED
            return es
        self.rewrite(compressed)
        with zipfile.ZipFile(self.archive) as package:
            info = package.getinfo('report.txt')
        raw = bytearray(self.archive.read_bytes())
        name_size, extra_size = struct.unpack_from('<HH', raw, info.header_offset + 26)
        start = info.header_offset + 30 + name_size + extra_size
        raw[start] = 0xff  # Reserved deflate block type, before a CRC can be checked.
        self.archive.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.delivery.verify(self.archive)

    def test_deflate_hidden_suffix_cannot_verify_a_truncated_handoff(self):
        self.build()
        def hidden_suffix(es):
            changed = []
            for info, raw in es:
                info.compress_type = zipfile.ZIP_DEFLATED
                if info.filename == 'HANDOFF.json':
                    raw += b'hidden unmanifested expansion\n' * 1000
                changed.append((info, raw))
            return changed
        self.rewrite(hidden_suffix)
        raw = bytearray(self.archive.read_bytes())
        with zipfile.ZipFile(self.archive) as package:
            info = package.getinfo('HANDOFF.json')
        prefix = self.raws['HANDOFF.json']
        crc = zlib.crc32(prefix)
        struct.pack_into('<L', raw, info.header_offset + 14, crc)
        struct.pack_into('<L', raw, info.header_offset + 22, len(prefix))
        central = raw.rfind(b'PK\x05\x06')
        position = struct.unpack_from('<L', raw, central + 16)[0]
        while raw[position:position + 4] == b'PK\x01\x02':
            name_size, extra_size, comment_size = struct.unpack_from('<HHH', raw, position + 28)
            if raw[position + 46:position + 46 + name_size] == b'HANDOFF.json':
                struct.pack_into('<L', raw, position + 16, crc)
                struct.pack_into('<L', raw, position + 24, len(prefix))
            position += 46 + name_size + extra_size + comment_size
        self.archive.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.delivery.verify(self.archive)

    def test_central_directory_is_bounded_before_zip_object_allocation(self):
        # EOCD lies about the record count; each central entry is otherwise
        # parsable by ZipFile, which used to allocate all 50,000 ZipInfo objects.
        entry = struct.pack('<4s6H3L5H2L', b'PK\x01\x02', 788, 20, 0, 0, 0, 33,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 0) + b'x'
        directory = entry * 50000
        end = struct.pack('<4s4H2LH', b'PK\x05\x06', 0, 0, 2, 2, len(directory), 0, 0)
        self.archive.write_bytes(directory + end)
        created = []
        original = zipfile.ZipInfo
        def counted_info(*args, **kwargs):
            created.append(1)
            return original(*args, **kwargs)
        with mock.patch.object(self.delivery.zipfile, 'ZipInfo', side_effect=counted_info):
            with self.assertRaises(ValueError):
                self.delivery.verify(self.archive)
        self.assertEqual(len(created), 0, 'hostile directory reached ZipInfo allocation')

    def test_stored_member_local_sizes_cannot_disagree_with_directory(self):
        self.build()
        raw = bytearray(self.archive.read_bytes())
        with zipfile.ZipFile(self.archive) as package:
            info = package.getinfo('report.txt')
        prefix = b'report'
        struct.pack_into('<LLL', raw, info.header_offset + 14,
                         zlib.crc32(prefix), len(prefix), len(prefix))
        # Central/local mismatch must be rejected before trusting either view.
        self.archive.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.delivery.verify(self.archive)

    def test_malformed_canonical_zip_boundaries_and_headers_are_rejected(self):
        self.build()
        original = self.archive.read_bytes()
        end = len(original) - 22
        start = struct.unpack_from('<L', original, end + 16)[0]
        mutations = []
        wrong_count = bytearray(original)
        struct.pack_into('<HH', wrong_count, end + 8, 2, 2)
        mutations.append(wrong_count)
        compressed_size = bytearray(original)
        struct.pack_into('<L', compressed_size, start + 20, 0)
        mutations.append(compressed_size)
        overlapping = bytearray(original)
        struct.pack_into('<L', overlapping, start + 42, 1)
        mutations.append(overlapping)
        local_flags = bytearray(original)
        struct.pack_into('<H', local_flags, 6, 8)  # Undeclared data descriptor.
        mutations.append(local_flags)
        gap = bytearray(original[:start] + b'x' + original[start:])
        struct.pack_into('<L', gap, end + 1 + 16, start + 1)
        mutations.append(gap)
        mutations.extend([original + b'ignored tail', b'prefix' + original])
        for index, raw in enumerate(mutations):
            with self.subTest(mutation=index):
                self.archive.write_bytes(raw)
                with self.assertRaises(ValueError):
                    self.delivery.verify(self.archive)

    def test_canonical_utf8_member_round_trip(self):
        path = 'nested/élan-😀.md'
        raw = b'explicit unicode member\n'
        (self.root / path).write_bytes(raw)
        self.spec['files'].append({'path': path, 'bytes': len(raw), 'sha256': digest(raw)})
        built = self.build()
        self.assertEqual(self.delivery.verify(self.archive)['archive_sha256'],
                         built['archive_sha256'])

    def test_count_total_size_and_boolean_size_limits_are_enforced(self):
        for rows in (
            [{'path': 'file' + str(i), 'bytes': 0, 'sha256': digest(b'')}
             for i in range(self.delivery.MAX_FILES + 1)],
            [{'path': 'file' + str(i), 'bytes': self.delivery.MAX_FILE_BYTES, 'sha256': '0' * 64}
             for i in range(5)],
            [{'path': 'HANDOFF.json', 'bytes': True, 'sha256': '0' * 64}],
        ):
            spec = copy.deepcopy(self.spec)
            spec['files'] = rows
            with self.assertRaises(ValueError):
                self.build(spec)

    def test_rehashed_archive_still_checks_handoff_envelope(self):
        self.build()
        def corrupt_envelope(es):
            envelope = json.loads(self.raws['HANDOFF.json'])
            envelope['payload']['next_action'] = 'changed without a new envelope digest'
            damaged = encoded(envelope)
            changed = []
            for info, raw in es:
                if info.filename == 'HANDOFF.json':
                    raw = damaged
                elif info.filename == MANIFEST:
                    manifest = json.loads(raw)
                    for row in manifest['files']:
                        if row['filename'] == 'HANDOFF.json':
                            row.update(size=len(damaged), sha256=digest(damaged))
                    raw = encoded(manifest)
                changed.append((info, raw))
            return changed
        self.rewrite(corrupt_envelope)
        with self.assertRaises(ValueError):
            self.delivery.verify(self.archive)

    def prefix_archive(self, additions):
        """Construct exact STORED headers and consistent identities, not bad hashes."""
        payloads = dict(self.raws)
        payloads.update(additions)
        document = {'version': 1, 'handoff': 'HANDOFF.json', 'files': [
            {'filename': name, 'size': len(raw), 'sha256': digest(raw)}
            for name, raw in sorted(payloads.items())]}
        members = dict(payloads)
        members[MANIFEST] = encoded(document)
        with zipfile.ZipFile(self.archive, 'w', compression=zipfile.ZIP_STORED) as archive:
            for name, raw in sorted(members.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, raw)
        return self.archive.read_bytes()

    def test_generated_manifest_is_not_a_payload_directory(self):
        name = MANIFEST + '/source.txt'
        path = self.root / name
        path.parent.mkdir()
        path.write_bytes(b'source\n')
        spec = copy.deepcopy(self.spec)
        spec['files'].append({'path': name, 'bytes': 7, 'sha256': digest(b'source\n')})
        with self.assertRaisesRegex(ValueError, 'file/directory member conflict'):
            self.build(spec)
        self.assertFalse(self.archive.exists())
        self.assertEqual(path.read_bytes(), b'source\n')

    def test_prefix_invariant_checks_all_rows_in_both_identity_schemas(self):
        cases = [
            ['a', 'a/source.txt'],
            ['a/b', 'a/b/c'],
            ['a', 'a/b/c'],
            ['a', 'a-bridge', 'a/source.txt'],
            ['HANDOFF.json/note.txt'],
            [MANIFEST + '/nested/source.txt'],
            ['\u00e9tude', '\u00e9tude/\u03bb.txt'],
        ]
        for names in cases:
            for reverse in (False, True):
                for manifest in (False, True):
                    with self.subTest(names=names, reverse=reverse, manifest=manifest):
                        rows = [('HANDOFF.json', self.raws['HANDOFF.json'])]
                        rows.extend((name, b'x') for name in names)
                        if reverse:
                            rows.reverse()
                        pk, sk = ('filename', 'size') if manifest else ('path', 'bytes')
                        document = {'version': 1, 'handoff': 'HANDOFF.json', 'files': [
                            {pk: name, sk: len(raw), 'sha256': digest(raw)}
                            for name, raw in rows]}
                        with self.assertRaisesRegex(ValueError, 'file/directory member conflict'):
                            self.delivery.identity_rows(document, manifest=manifest)

    def test_valid_hashes_do_not_excuse_conflicting_archive_paths(self):
        cases = [
            {'nested': b'parent file'},
            {'HANDOFF.json/note.txt': b'x'},
            {MANIFEST + '/source.txt': b'x'},
            {'a': b'x', 'a-bridge': b'y', 'a/source.txt': b'z'},
        ]
        for additions in cases:
            with self.subTest(additions=additions):
                original = self.prefix_archive(additions)
                # Separate the namespace defect from framing, CRC, identity or handoff failure.
                members = self.delivery.stored_members(original)
                manifest = json.loads(members[MANIFEST])
                for row in manifest['files']:
                    raw = members[row['filename']]
                    self.assertEqual((len(raw), digest(raw)), (row['size'], row['sha256']))
                self.delivery.handoff_payload(members['HANDOFF.json'])
                with self.assertRaisesRegex(ValueError, 'file/directory member conflict'):
                    self.delivery.verify(self.archive)
                self.assertEqual(self.archive.read_bytes(), original)

    def test_prefix_conflict_cli_is_a_clean_nonzero_result(self):
        original = self.prefix_archive({'nested': b'parent file'})
        flags = ['-B', '-O', '-S'] if sys.flags.optimize else ['-B', '-S']
        result = subprocess.run([sys.executable, *flags, str(TOOL), 'verify',
                                 str(self.archive)], capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertEqual(result.stderr, b'')
        response = json.loads(result.stdout)
        self.assertEqual(response, {'ok': False,
                                   'error': 'file/directory member conflict: nested/proof.bin'})
        self.assertEqual(self.archive.read_bytes(), original)

    def test_component_neighbors_and_shared_directories_still_round_trip(self):
        additions = {
            'a': b'plain file', 'a-bridge': b'neighbor', 'ab/source.txt': b'child',
            'sources/a.txt': b'one', 'sources/deeper/b.txt': b'two',
            'sources/\u03bb.txt': b'unicode', 'HANDOFF.json.notes': b'not a child',
            MANIFEST + '.notes/source.txt': b'not the manifest directory',
        }
        spec = copy.deepcopy(self.spec)
        for name, raw in additions.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            spec['files'].append({'path': name, 'bytes': len(raw), 'sha256': digest(raw)})
        built = self.build(spec)
        self.assertEqual(self.delivery.verify(self.archive), built)
        reverse_path = self.root / 'reordered.zip'
        spec['files'].reverse()
        self.build(spec, reverse_path)
        self.assertEqual(self.archive.read_bytes(), reverse_path.read_bytes())
        destination = self.root / 'extracted'
        with zipfile.ZipFile(self.archive) as archive:
            archive.extractall(destination)
        for name, raw in {**self.raws, **additions}.items():
            self.assertEqual((destination / name).read_bytes(), raw)

    def test_cli_returns_compact_results_and_nonzero_on_bad_archive(self):
        self.spec_path.write_bytes(encoded(self.spec))
        built = subprocess.run([sys.executable, '-B', str(TOOL), 'build', '--root', str(self.root),
                                '--spec', str(self.spec_path), '--output', str(self.archive)],
                               capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertTrue(json.loads(built.stdout)['ok'])
        checked = subprocess.run([sys.executable, '-B', str(TOOL), 'verify', str(self.archive)],
                                 capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.archive.write_bytes(b'not a ZIP')
        failed = subprocess.run([sys.executable, '-B', str(TOOL), 'verify', str(self.archive)],
                                capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse(json.loads(failed.stdout)['ok'])


if __name__ == '__main__':
    unittest.main()
