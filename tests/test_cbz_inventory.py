"""Real-Git contract and source-mutation tests for the narrow CBZ inventory."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOL = Path(os.environ.get('CBZ_INVENTORY_UNDER_TEST',
                          Path(__file__).resolve().parents[1] / 'tools/cbz_inventory.py'))


def git(repo, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    return subprocess.check_output(['git', '-C', str(repo), *args], env=env,
                                   stderr=subprocess.PIPE)


class InventoryContract(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.is_file(), 'CBZ inventory implementation is not supplied')
        spec = importlib.util.spec_from_file_location('cbz_inventory_under_test', TOOL)
        self.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tool)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'repos'
        self.root.mkdir()
        for name in ('alpha', 'beta'):
            repo = self.root / name
            repo.mkdir()
            git(repo, 'init', '-q', '--object-format=sha1')
            git(repo, 'config', 'user.name', 'Fixture')
            git(repo, 'config', 'user.email', 'fixture@example.invalid')
            git(repo, 'config', 'commit.gpgsign', 'false')
            (repo / 'a.txt').write_bytes(b'alpha\n')
            self.commit(repo)
        self.names = ['alpha', 'beta']

    def commit(self, repo):
        git(repo, 'add', '--all')
        git(repo, 'commit', '-qm', 'fixture')

    def capture(self, obligations=None):
        return self.tool.generate(self.root, self.names, obligations or [])

    def binding(self, data, path='a.txt', repo='alpha'):
        source = next(r for r in data['repositories'] if r['name'] == repo)
        row = next(f for f in source['files'] if f['path'] == path)
        return {'repository': repo, 'commit': source['head'], 'path': path,
                'git_object': row['git_object'], 'sha256': row['sha256']}

    def quarantine(self, data):
        return {'id': 'claim-1', 'status': 'QUARANTINED',
                'source_bindings': [self.binding(data)],
                'exact_boundary': 'No kernel-checked proof of the continuum statement.',
                'blocking_reason': 'Only an informal derivation is present.',
                'isolation': 'Excluded from every CLOSED dependency.',
                'owner': 'none', 'maintenance': 'zero'}

    def cli(self, *args, env=None):
        optimize = ['-O'] if sys.flags.optimize else []
        return subprocess.run([sys.executable, '-B', *optimize, '-S', str(TOOL), *args],
                              text=True, capture_output=True, env=env)

    def test_exact_tracked_bytes_and_deterministic_roundtrip(self):
        data = self.capture()
        self.assertEqual(data, self.capture())
        row = data['repositories'][0]['files'][0]
        self.assertEqual(row['sha256'], 'b6a98d9ce9a2d9149288fa3df42d377c3e42737afdcdaf714e33c0a100b51060')
        self.assertEqual(row['bytes'], 6)
        self.assertFalse(data['cbz_green'])
        self.assertFalse(data['semantic_claim_completeness'])
        self.tool.verify(data, self.root, self.names)

    def test_added_index_file_and_dirty_state_are_captured(self):
        repo = self.root / 'alpha'
        (repo / 'added').write_bytes(b'staged')
        git(repo, 'add', 'added')
        data = self.capture()
        self.assertTrue(data['repositories'][0]['dirty'])
        self.assertEqual([r['path'] for r in data['repositories'][0]['files']], ['a.txt', 'added'])

    def test_staged_deletion_cannot_hide_head_tracked_source(self):
        repo = self.root / 'alpha'
        git(repo, 'rm', '-q', 'a.txt')
        with self.assertRaises(ValueError): self.capture()

    def test_head_and_index_objects_remain_distinct_after_staging_change(self):
        repo = self.root / 'alpha'
        head_oid = git(repo, 'rev-parse', 'HEAD:a.txt').decode().strip()
        (repo / 'a.txt').write_bytes(b'staged replacement\n')
        git(repo, 'add', 'a.txt')
        row = self.capture()['repositories'][0]['files'][0]
        self.assertEqual(row['git_object'], head_oid)
        self.assertEqual(row['head_git_object'], head_oid)
        self.assertNotEqual(row['index_git_object'], head_oid)

    def test_index_mode_change_cannot_hide_behind_same_dirty_status(self):
        repo = self.root / 'alpha'
        (repo / 'a.txt').write_bytes(b'staged change\n'); git(repo, 'add', 'a.txt')
        (repo / 'a.txt').write_bytes(b'unstaged change\n')
        before = self.capture()
        oid = before['repositories'][0]['files'][0]['index_git_object']
        git(repo, 'update-index', '--cacheinfo', '100755,' + oid + ',a.txt')
        after = self.capture()
        self.assertEqual(before['repositories'][0]['status_porcelain_base64'],
                         after['repositories'][0]['status_porcelain_base64'])
        with self.assertRaises(ValueError): self.tool.verify(before, self.root, self.names)

    def test_configured_git_filter_is_refused_without_running_it(self):
        repo = self.root / 'alpha'
        (repo / '.gitattributes').write_text('a.txt filter=probe\n'); self.commit(repo)
        marker = Path(self.temp.name) / 'filter-was-executed'
        script = Path(self.temp.name) / 'filter.py'
        script.write_text('import pathlib,sys\npathlib.Path(' + repr(str(marker)) +
                          ').write_text("executed")\nsys.stdout.buffer.write(sys.stdin.buffer.read())\n')
        git(repo, 'config', 'filter.probe.clean', shlex.quote(sys.executable) + ' ' + shlex.quote(str(script)))
        (repo / 'a.txt').write_bytes(b'alpha\n')
        try:
            with self.assertRaises(ValueError): self.capture()
        finally:
            self.assertFalse(marker.exists(), 'inventory must never execute repository clean filters')

    def test_boolean_byte_count_is_not_accepted_as_integer_one(self):
        repo = self.root / 'alpha'
        (repo / 'one').write_bytes(b'x'); self.commit(repo)
        data = self.capture()
        row = next(r for r in data['repositories'][0]['files'] if r['path'] == 'one')
        row['bytes'] = True
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_dirty_working_proof_cannot_bind_itself_to_head_commit(self):
        (self.root / 'alpha/a.txt').write_bytes(b'dirty proof\n')
        data = self.capture(); data['obligations'] = [self.quarantine(data)]
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_ancestor_source_binding_survives_document_only_commit(self):
        before = self.capture(); entry = self.quarantine(before)
        (self.root / 'alpha/README.md').write_bytes(b'document-only update\n')
        self.commit(self.root / 'alpha')
        after = self.capture(); after['obligations'] = [entry]
        try:
            self.tool.verify(after, self.root, self.names)
        except ValueError as error:
            self.fail('Unchanged ancestral source must remain bindable: ' + str(error))

    def test_ancestor_source_binding_rejects_changed_current_source(self):
        before = self.capture(); entry = self.quarantine(before)
        (self.root / 'alpha/a.txt').write_bytes(b'different proof\n')
        self.commit(self.root / 'alpha')
        after = self.capture(); after['obligations'] = [entry]
        with self.assertRaises(ValueError): self.tool.verify(after, self.root, self.names)

    def test_explicit_historical_quarantine_resolves_removed_source(self):
        repo = self.root / 'alpha'
        (repo / 'archived.md').write_bytes(b'historical claim\n'); self.commit(repo)
        before = self.capture(); entry = self.quarantine(before)
        entry['source_bindings'] = [{**self.binding(before, 'archived.md'), 'scope': 'historical'}]
        git(repo, 'rm', '-q', 'archived.md'); git(repo, 'commit', '-qm', 'remove old packet')
        after = self.capture(); after['obligations'] = [entry]
        try:
            self.tool.verify(after, self.root, self.names)
        except ValueError as error:
            self.fail('Explicit historical source must resolve exact archived bytes: ' + str(error))

    def test_missing_immutable_source_object_fails_without_network(self):
        data = self.capture(); entry = self.quarantine(data)
        entry['source_bindings'][0].update(commit='f' * 40, scope='historical')
        data['obligations'] = [entry]
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_complete_filename_bytes_include_tabs_newlines_and_non_utf8(self):
        repo = self.root / 'alpha'
        for name in ('tab\tname', 'line\nname', os.fsdecode(b'non-utf8-\xff')):
            (repo / name).write_bytes(b'x')
        self.commit(repo)
        data = self.capture()
        names = {r['path'] for r in data['repositories'][0]['files']}
        self.assertEqual(names, {'a.txt', 'tab\tname', 'line\nname', os.fsdecode(b'non-utf8-\xff')})
        self.tool.verify(json.loads(json.dumps(data)), self.root, self.names)

    def test_symlink_is_link_bytes_and_does_not_read_target(self):
        repo = self.root / 'alpha'
        (Path(self.temp.name) / 'secret').write_bytes(b'must never be read')
        (repo / 'link').symlink_to('../../secret')
        self.commit(repo)
        row = next(x for x in self.capture()['repositories'][0]['files'] if x['path'] == 'link')
        self.assertEqual(row['kind'], 'symlink')
        self.assertEqual(row['bytes'], len(b'../../secret'))
        self.assertEqual(row['sha256'], hashlib.sha256(b'../../secret').hexdigest())

    def test_unmaterialized_gitlink_is_explicit(self):
        repo = self.root / 'alpha'
        git(repo, 'update-index', '--add', '--cacheinfo', '160000,' + 'a' * 40 + ',vendor')
        git(repo, 'commit', '-qm', 'gitlink')
        row = next(x for x in self.capture()['repositories'][0]['files'] if x['path'] == 'vendor')
        self.assertEqual(row['kind'], 'gitlink')
        self.assertEqual(row['git_object'], 'a' * 40)
        self.assertEqual(row['materialization'], 'absent')
        self.assertIsNone(row['sha256'])

    def test_missing_repository_is_rejected(self):
        shutil.rmtree(self.root / 'beta')
        with self.assertRaises(ValueError): self.capture()

    def test_unexpected_or_duplicate_manifest_repository_is_rejected(self):
        for mutation in ('extra', 'duplicate'):
            data = self.capture()
            added = copy.deepcopy(data['repositories'][0])
            if mutation == 'extra': added['name'] = 'unexpected'
            data['repositories'].append(added)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.tool.verify(data, self.root, self.names)

    def test_missing_manifest_file_is_rejected(self):
        data = self.capture()
        data['repositories'][0]['files'] = []
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_declared_file_hash_substitution_is_rejected(self):
        data = self.capture()
        data['repositories'][0]['files'][0]['sha256'] = '0' * 64
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_working_file_change_or_deletion_is_rejected(self):
        data = self.capture()
        path = self.root / 'alpha/a.txt'
        path.write_bytes(b'changed\n')
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)
        path.unlink()
        with self.assertRaises(ValueError): self.capture()

    def test_fifo_and_symlink_parent_substitutions_are_rejected(self):
        path = self.root / 'alpha/a.txt'
        path.unlink(); os.mkfifo(path)
        with self.assertRaises(ValueError): self.capture()
        path.unlink(); path.write_bytes(b'alpha\n')
        (self.root / 'alpha/folder').mkdir()
        (self.root / 'alpha/folder/nested').write_bytes(b'nested')
        self.commit(self.root / 'alpha')
        shutil.rmtree(self.root / 'alpha/folder')
        (self.root / 'alpha/folder').symlink_to(self.root / 'beta', target_is_directory=True)
        with self.assertRaises(ValueError): self.capture()

    def test_file_content_race_is_rejected(self):
        original = self.tool._read_regular
        changed = False
        def race(*args, **kwargs):
            nonlocal changed
            result = original(*args, **kwargs)
            if not changed:
                changed = True
                (self.root / 'alpha/a.txt').write_bytes(b'bravo\n')
            return result
        with mock.patch.object(self.tool, '_read_regular', race), self.assertRaises(ValueError):
            self.capture()

    def test_head_race_is_rejected(self):
        original = self.tool._read_regular
        changed = False
        def race(*args, **kwargs):
            nonlocal changed
            result = original(*args, **kwargs)
            if not changed:
                changed = True
                git(self.root / 'alpha', 'commit', '--allow-empty', '-qm', 'racing head')
            return result
        with mock.patch.object(self.tool, '_read_regular', race), self.assertRaises(ValueError):
            self.capture()

    def test_complete_quarantine_is_verified_without_semantic_claim(self):
        data = self.capture()
        data['obligations'] = [self.quarantine(data)]
        self.tool.verify(data, self.root, self.names)
        self.assertFalse(data['semantic_claim_completeness'])

    def test_invalid_status_or_incomplete_quarantine_is_rejected(self):
        for key, value in [('status', 'ACCEPT'), ('exact_boundary', ''),
                           ('blocking_reason', ''), ('isolation', ''),
                           ('owner', 'agent'), ('maintenance', 0)]:
            data = self.capture(); entry = self.quarantine(data); entry[key] = value
            data['obligations'] = [entry]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.tool.verify(data, self.root, self.names)

    def test_stale_source_binding_and_duplicate_obligation_are_rejected(self):
        data = self.capture(); entry = self.quarantine(data)
        entry['source_bindings'][0]['sha256'] = 'f' * 64
        data['obligations'] = [entry]
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)
        data['obligations'] = [self.quarantine(data), self.quarantine(data)]
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_closed_word_without_actual_kernel_binding_is_rejected(self):
        data = self.capture()
        data['obligations'] = [{'id': 'claim-1', 'status': 'CLOSED',
                                'source_bindings': [self.binding(data)],
                                'kernel_evidence': {'status': 'kernel-checked'}}]
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def closed_fixture(self, manifest_target='Example.t', checked_commit_override=None):
        repo = self.root / 'alpha'
        manifest = (json.dumps({'targets': [{'name': manifest_target}]}) + '\n').encode()
        (repo / 'formal.json').write_bytes(manifest)
        if git(repo, 'status', '--porcelain'):
            self.commit(repo)
        checked = checked_commit_override or git(repo, 'rev-parse', 'HEAD').decode().strip()
        receipt = {'formalization_status': 'kernel-checked', 'checked_commit': checked,
                   'manifest_sha256': hashlib.sha256(manifest).hexdigest(),
                   'axioms': {'Example.t': []}}
        (repo / 'receipt.json').write_text(json.dumps(receipt))
        self.commit(repo)
        data = self.capture(); source = self.binding(data)
        data['obligations'] = [{'id': 'claim-1', 'status': 'CLOSED', 'source_bindings': [source],
            'kernel_evidence': {'receipt': self.binding(data, 'receipt.json'),
                'manifest': self.binding(data, 'formal.json'), 'checked_commit': checked,
                'targets': ['Example.t'], 'source_bindings': [source]}}]
        return data

    def test_closed_binding_checks_bytes_without_authenticating_a_kernel_run(self):
        data = self.closed_fixture()
        self.tool.verify(data, self.root, self.names)
        self.assertFalse(data['cbz_green'])
        self.assertFalse(data['semantic_claim_completeness'])

    def test_closed_missing_target_manifest_or_receipt_binding_is_rejected(self):
        for key in ('targets', 'manifest', 'receipt', 'source_bindings'):
            data = self.closed_fixture()
            del data['obligations'][0]['kernel_evidence'][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.tool.verify(data, self.root, self.names)

    def test_closed_target_not_in_receipt_is_rejected(self):
        data = self.closed_fixture()
        data['obligations'][0]['kernel_evidence']['targets'] = ['Other.t']
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_closed_nonexistent_checked_commit_is_rejected(self):
        data = self.closed_fixture(checked_commit_override='f' * 40)
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_closed_target_not_in_actual_manifest_is_rejected(self):
        data = self.closed_fixture(manifest_target='Other.t')
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_closed_receipt_cannot_be_attached_to_changed_proof(self):
        data = self.closed_fixture(); entry = data['obligations'][0]
        (self.root / 'alpha/a.txt').write_bytes(b'new proof after kernel run\n')
        self.commit(self.root / 'alpha')
        after = self.capture(); source = self.binding(after)
        entry['source_bindings'] = [source]
        entry['kernel_evidence']['source_bindings'] = [source]
        after['obligations'] = [entry]
        with self.assertRaises(ValueError): self.tool.verify(after, self.root, self.names)

    def test_closed_proof_cannot_use_dormant_historical_scope(self):
        data = self.closed_fixture()
        data['obligations'][0]['source_bindings'][0]['scope'] = 'historical'
        data['obligations'][0]['kernel_evidence']['source_bindings'][0]['scope'] = 'historical'
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def rf312_fixture(self, overlay=None, receipt_digest=None, replace_after_check=False,
                      marker=None, repository='Math-'):
        repo = self.root / repository
        if repository != 'alpha':
            (self.root / 'alpha').rename(repo)
            self.names = [repository, 'beta']
        formal = repo / 'formal'; formal.mkdir()
        source_raw = b'theorem example : True := by trivial\n'
        (formal / 'A.lean').write_bytes(source_raw)
        target = 'ResearchFormalCoreR1.example'
        base = {'source_identity': marker or {'schema': 'rf312-overlay-v1',
                    'overlay': 'manifest.overlay.json'}, 'source_modules': ['A.lean'],
                'targets': [target], 'files': {'A.lean': hashlib.sha256(source_raw).hexdigest()}}
        base_raw = (json.dumps(base) + '\n').encode()
        overlay_data = overlay if overlay is not None else {
            'schema_version': 1, 'files': {'A.lean': hashlib.sha256(source_raw).hexdigest()}}
        overlay_raw = (json.dumps(overlay_data) + '\n').encode()
        (formal / 'manifest.json').write_bytes(base_raw)
        (formal / 'manifest.overlay.json').write_bytes(overlay_raw)
        self.commit(repo)
        checked = git(repo, 'rev-parse', 'HEAD').decode().strip()
        if replace_after_check:
            overlay_raw += b'\n'
            (formal / 'manifest.overlay.json').write_bytes(overlay_raw)
        digest = hashlib.sha256(base_raw + b'\0RF312-OVERLAY\0' + overlay_raw).hexdigest()
        if receipt_digest == 'raw': digest = hashlib.sha256(base_raw).hexdigest()
        elif receipt_digest is not None: digest = receipt_digest
        receipt = {'formalization_status': 'kernel-checked', 'checked_commit': checked,
                   'manifest_sha256': digest, 'axioms': {target: []}}
        (repo / 'receipt.json').write_text(json.dumps(receipt)); self.commit(repo)
        data = self.capture()
        source = self.binding(data, 'formal/A.lean', repository)
        data['obligations'] = [{'id': 'rf312-claim', 'status': 'CLOSED', 'source_bindings': [source],
            'kernel_evidence': {'receipt': self.binding(data, 'receipt.json', repository),
                'manifest': self.binding(data, 'formal/manifest.json', repository),
                'manifest_overlay': self.binding(data, 'formal/manifest.overlay.json', repository),
                'checked_commit': checked, 'targets': [target], 'source_bindings': [source]}}]
        return data

    def test_rf312_exact_composite_receipt_is_bound(self):
        data = self.rf312_fixture()
        try:
            self.tool.verify(data, self.root, self.names)
        except ValueError as error:
            self.fail('Exact RF312 composite receipt must be supported: ' + str(error))
        self.assertFalse(data['cbz_green'])
        self.assertFalse(data['semantic_claim_completeness'])

    def test_rf312_wrong_or_raw_only_receipt_digest_is_rejected(self):
        for digest in ('0' * 64, 'raw'):
            data = self.rf312_fixture(receipt_digest=digest)
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                self.tool.verify(data, self.root, self.names)
            shutil.rmtree(self.root / 'Math-')
            # Recreate the alpha source expected by this fixture's next iteration.
            shutil.copytree(self.root / 'beta', self.root / 'alpha')
            self.names = ['alpha', 'beta']

    def test_rf312_marker_cannot_fall_back_to_raw_digest_without_overlay(self):
        data = self.rf312_fixture(receipt_digest='raw')
        del data['obligations'][0]['kernel_evidence']['manifest_overlay']
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_rf312_replaced_overlay_binding_is_rejected(self):
        data = self.rf312_fixture()
        data['obligations'][0]['kernel_evidence']['manifest_overlay']['sha256'] = '0' * 64
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_rf312_overlay_changed_after_checked_cut_is_rejected(self):
        data = self.rf312_fixture(replace_after_check=True)
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_rf312_overlay_rejects_extra_keys_and_invalid_file_schema(self):
        cases = [{'schema_version': 1, 'files': {}, 'targets': ['Injected.t']},
                 {'schema_version': True, 'files': {}},
                 {'schema_version': 1, 'files': []},
                 {'schema_version': 1, 'files': {'A.lean': 'not-a-sha256'}}]
        for overlay in cases:
            data = self.rf312_fixture(overlay=overlay)
            with self.subTest(overlay=overlay), self.assertRaises(ValueError):
                self.tool.verify(data, self.root, self.names)
            shutil.rmtree(self.root / 'Math-'); shutil.copytree(self.root / 'beta', self.root / 'alpha')
            self.names = ['alpha', 'beta']

    def test_rf312_unknown_scheme_and_other_repository_are_rejected(self):
        data = self.rf312_fixture(marker={'schema': 'rf312-overlay-v2', 'overlay': 'manifest.overlay.json'})
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)
        shutil.rmtree(self.root / 'Math-'); shutil.copytree(self.root / 'beta', self.root / 'alpha')
        self.names = ['alpha', 'beta']
        data = self.rf312_fixture(repository='alpha')
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_raw_main_format_cannot_accept_manifest_overlay_field(self):
        data = self.closed_fixture()
        evidence = data['obligations'][0]['kernel_evidence']
        evidence['manifest_overlay'] = dict(evidence['manifest'])
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def closed_with_axioms(self, reported):
        before = self.closed_fixture(); entry = before['obligations'][0]
        repo = self.root / 'alpha'; path = repo / 'receipt.json'
        receipt = json.loads(path.read_text()); receipt['axioms']['Example.t'] = reported
        path.write_text(json.dumps(receipt)); self.commit(repo)
        after = self.capture()
        entry['kernel_evidence']['receipt'] = self.binding(after, 'receipt.json')
        after['obligations'] = [entry]
        return after

    def test_closed_report_accepts_only_the_approved_standard_axiom_lists(self):
        for axioms in (['propext'], ['Classical.choice', 'Quot.sound'],
                       ['propext', 'Classical.choice', 'Quot.sound']):
            data = self.closed_with_axioms(axioms)
            with self.subTest(axioms=axioms): self.tool.verify(data, self.root, self.names)

    def test_closed_report_rejects_unknown_or_invalid_axiom_strings(self):
        for axioms in (['sorryAx'], ['customAxiom'], ['propext '], ['']):
            data = self.closed_with_axioms(axioms)
            with self.subTest(axioms=axioms), self.assertRaises(ValueError):
                self.tool.verify(data, self.root, self.names)

    def test_closed_report_rejects_malformed_axiom_lists(self):
        for axioms in ('propext', {}, None, 1, [1]):
            data = self.closed_with_axioms(axioms)
            with self.subTest(axioms=axioms), self.assertRaises(ValueError):
                self.tool.verify(data, self.root, self.names)

    def test_closed_report_rejects_duplicate_axiom_names(self):
        data = self.closed_with_axioms(['propext', 'propext'])
        with self.assertRaises(ValueError): self.tool.verify(data, self.root, self.names)

    def test_cli_roundtrip_rejects_duplicate_json_keys_and_floats(self):
        output = Path(self.temp.name) / 'inventory.json'
        result = self.cli('generate', '--root', str(self.root), '--output', str(output),
                          '--repositories', *self.names)
        self.assertEqual(result.returncode, 0, result.stderr)
        original = output.read_text()
        result = self.cli('verify', '--root', str(self.root), '--manifest', str(output),
                          '--repositories', *self.names)
        self.assertEqual(result.returncode, 0, result.stderr)
        for text in (original.replace('"schema":', '"schema": "duplicate", "schema":', 1),
                     original.replace('"bytes": 6', '"bytes": 6.0', 1)):
            output.write_text(text)
            result = self.cli('verify', '--root', str(self.root), '--manifest', str(output),
                              '--repositories', *self.names)
            self.assertNotEqual(result.returncode, 0)

    def test_control_file_inside_source_is_rejected(self):
        result = self.cli('generate', '--root', str(self.root), '--output',
                          str(self.root / 'alpha/inventory.json'), '--repositories', *self.names)
        self.assertNotEqual(result.returncode, 0)

    def test_ambient_git_override_does_not_substitute_repository(self):
        with mock.patch.dict(os.environ, {'GIT_DIR': str(self.root / 'beta/.git'),
                                         'GIT_WORK_TREE': str(self.root / 'beta')}):
            data = self.capture()
        self.assertEqual(data['repositories'][0]['head'],
                         git(self.root / 'alpha', 'rev-parse', 'HEAD').decode().strip())

    @unittest.skipIf(os.environ.get('CBZ_INVENTORY_MUTANT') == '1', 'outer mutation controller only')
    def test_source_mutations_are_killed(self):
        source = TOOL.read_text()
        mutations = {
            'remove_file_set_check': ('require(set(expected_files) == set(actual_files),',
                                      'require(True,'),
            'remove_file_identity_check': ('require(record == actual_files[path],',
                                           'require(True,')}
        for name, (before, after) in mutations.items():
            with self.subTest(mutation=name):
                self.assertEqual(source.count(before), 1, 'mutation anchor must be unique')
                mutant = Path(self.temp.name) / (name + '.py')
                mutant.write_text(source.replace(before, after))
                env = dict(os.environ, CBZ_INVENTORY_UNDER_TEST=str(mutant),
                           CBZ_INVENTORY_MUTANT='1', PYTHONDONTWRITEBYTECODE='1')
                optimize = ['-O'] if sys.flags.optimize else []
                result = subprocess.run([sys.executable, '-B', *optimize, '-S', str(Path(__file__).resolve()), '-v'],
                                        env=env, text=True, capture_output=True, timeout=180)
                self.assertNotEqual(result.returncode, 0, 'mutated scanner escaped the tests')
                expected = ('test_missing_manifest_file_is_rejected' if name == 'remove_file_set_check'
                            else 'test_declared_file_hash_substitution_is_rejected')
                self.assertIn('FAIL: ' + expected, result.stderr)


if __name__ == '__main__':
    unittest.main()
