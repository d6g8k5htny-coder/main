"""Behavioral contract for the deliberately finite CBZ static auditor.

The same tests run against source-mutated copies during `cbz_audit.py selftest`.
No fixture asserts total mathematical soundness from a lexical scan.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(os.environ.get('CBZ_AUDIT_UNDER_TEST', Path(__file__).resolve().parents[1] / 'tools/cbz_audit.py'))
spec = importlib.util.spec_from_file_location('cbz_audit_under_test', TOOL)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class AuditContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)

    def put(self, name, data, tracked=True):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = data.encode() if isinstance(data, str) else data
        path.write_bytes(raw)
        if tracked:
            subprocess.run(['git', '-C', str(self.root), 'add', '--', name], check=True)
        return raw

    def scan(self, **kwargs):
        return audit.scan(self.root, **kwargs)

    def codes(self, result):
        return {item['code'] for item in result['findings']}

    def gaps(self, result):
        return {item['code'] for item in result['coverage_gaps']}

    def registry(self, name, raw, role='certification'):
        return {'version': 1, 'files': [{'path': name, 'sha256': sha(raw), 'role': role}]}

    def test_each_real_placeholder_and_axiom_is_visible(self):
        for token in ('sorry', 'admit', 'axiom', 'native_decide'):
            with self.subTest(token=token):
                self.put('Proof.lean', 'theorem x : True := by\n  ' + token + '\n')
                result = self.scan()
                self.assertIn('lean.' + token, self.codes(result))
                self.assertEqual(result['result'], 'FAIL')
                self.assertEqual(result['findings'][0]['line'], 2)

    def test_nested_comments_strings_and_quoted_identifiers_are_not_keywords(self):
        self.put('Clean.lean', '/- sorry /- axiom -/ admit -/\n-- native_decide\ndef «sorry» := "admit \\" axiom"\ndef sorryish := 1\ndef x\' := \'a\'\n')
        result = self.scan()
        self.assertEqual(result['findings'], [])
        self.assertEqual(result['coverage_gaps'], [])
        self.assertEqual(result['static_result'], 'PASS')

    def test_comment_then_real_token_and_unicode_name(self):
        self.put('A.lean', '/- nested /- sorry -/ -/\ndef αsorry := 0\ntheorem x := by sorry')
        result = self.scan()
        self.assertEqual([f['code'] for f in result['findings']], ['lean.sorry'])
        self.assertEqual(result['findings'][0]['line'], 3)

    def test_trust_escape_hatches_and_quoted_trust_names(self):
        self.put('A.lean', 'unsafe def x := sorryAx\n@[implemented_by other] opaque y : Nat\n@[extern "f"] opaque z : Nat\n#check «sorryAx»')
        self.assertTrue({'lean.unsafe', 'lean.sorryAx', 'lean.implemented_by', 'lean.extern'} <= self.codes(self.scan()))

    def test_unterminated_lexemes_and_interpolation_are_gaps(self):
        for text in ('/- nested /- x -/', 'def x := "unclosed', 'def «unclosed', 'def x := s!"{sorry}"'):
            with self.subTest(text=text):
                self.put('A.lean', text)
                self.assertIn('lean.lexical_gap', self.gaps(self.scan()))

    def test_raw_lean_strings_hide_only_string_contents(self):
        self.put('A.lean', 'def x := r#"sorry " axiom"#\ntheorem y := by admit')
        self.assertEqual(self.codes(self.scan()), {'lean.admit'})

    def test_complete_tracked_inventory_including_hidden_and_spaces(self):
        self.put('normal.txt', 'ordinary')
        self.put('.hidden/a b.lean', 'example : True := by sorry')
        result = self.scan()
        self.assertEqual({i['path'] for i in result['files']}, {'normal.txt', '.hidden/a b.lean'})
        self.assertIn('lean.sorry', self.codes(result))
        self.assertEqual(result['tracked_count'], 2)
        self.assertEqual(result['read_count'], 2)
        self.assertEqual(result['proof_candidates'], ['.hidden/a b.lean'])

    def test_tracked_worktree_deletion_is_never_skipped(self):
        self.put('gone.lean', 'example := by sorry')
        (self.root / 'gone.lean').unlink()
        self.assertIn('file.unreadable', self.gaps(self.scan()))

    def test_invalid_utf8_in_lean_is_gap(self):
        self.put('A.lean', b'\xffsorry')
        self.assertIn('file.encoding', self.gaps(self.scan()))

    def test_archive_and_other_proof_format_are_explicit_gaps(self):
        self.put('archive.zip', b'PK\x03\x04')
        self.put('Proof.v', 'Admitted.')
        self.put('paper.tex', r'\begin{proof}omitted\end{proof}')
        result = self.scan()
        self.assertTrue({'archive.unsupported', 'proof.unsupported'} <= self.gaps(result))
        self.assertEqual(set(result['proof_candidates']), {'Proof.v', 'paper.tex'})
        self.assertEqual(result['result'], 'INCOMPLETE')

    def test_archive_magic_without_extension_is_gap(self):
        self.put('payload.bin', b'PK\x03\x04not parsed')
        self.assertIn('archive.unsupported', self.gaps(self.scan()))

    def test_symlink_is_gap_and_never_followed(self):
        self.put('target.txt', 'sorry')
        (self.root / 'link.lean').symlink_to('target.txt')
        subprocess.run(['git', '-C', str(self.root), 'add', 'link.lean'], check=True)
        self.assertIn('file.symlink', self.gaps(self.scan()))

    def test_submodule_is_explicit_gap(self):
        subprocess.run(['git', '-C', str(self.root), 'update-index', '--add', '--cacheinfo', '160000,' + 'a' * 40 + ',vendor'], check=True)
        self.assertIn('file.submodule', self.gaps(self.scan()))

    def test_untracked_proof_is_visible_gap(self):
        self.put('A.lean', 'example := by sorry', tracked=False)
        result = self.scan()
        self.assertIn('file.untracked_proof', self.gaps(result))
        self.assertEqual(result['untracked'], ['A.lean'])

    def test_ui_float_is_not_mathematical_certification_finding(self):
        self.put('ui.py', 'opacity = 0.5\n')
        self.assertNotIn('certification.float', self.codes(self.scan()))
        self.assertIn('certification_scope', self.scan()['semantic_receipts_required'])

    def test_certified_python_float_literals_and_calls_are_findings(self):
        for source in ('bound = 0.1', 'bound = float("1")', 'import builtins\nbound = builtins.float("1")'):
            with self.subTest(source=source):
                raw = self.put('bound.py', source)
                self.assertIn('certification.float', self.codes(self.scan(registry=self.registry('bound.py', raw))))

    def test_certified_python_exact_arithmetic_and_string_are_not_float_hits(self):
        raw = self.put('bound.py', 'from fractions import Fraction\nbound = Fraction(1, 10)\nlabel = "float(0.1)"\n')
        result = self.scan(registry=self.registry('bound.py', raw))
        self.assertEqual(result['findings'], [])
        self.assertEqual(result['coverage_gaps'], [])

    def test_certified_lean_float_and_unsupported_certifier(self):
        raw = self.put('Bound.lean', '#check Float\n')
        self.assertIn('certification.float', self.codes(self.scan(registry=self.registry('Bound.lean', raw))))
        raw = self.put('bound.js', 'let bound = 0.1;')
        self.assertIn('certification.unsupported', self.gaps(self.scan(registry=self.registry('bound.js', raw))))

    def test_invalid_python_certifier_is_gap(self):
        raw = self.put('bound.py', 'bound = (')
        self.assertIn('certification.parse', self.gaps(self.scan(registry=self.registry('bound.py', raw))))

    def test_registry_hash_and_unknown_file_refused(self):
        raw = self.put('bound.py', 'bound = 1')
        for registry in (self.registry('bound.py', b'wrong'), self.registry('missing.py', raw)):
            self.assertIn('registry.invalid', self.gaps(self.scan(registry=registry)))

    def test_registry_bad_roles_duplicates_and_traversal_refused(self):
        raw = self.put('bound.py', 'bound = 1')
        bad = self.registry('bound.py', raw, 'whitelist')
        duplicate = self.registry('bound.py', raw)
        duplicate['files'] *= 2
        traversal = self.registry('../bound.py', raw)
        for registry in (bad, duplicate, traversal, {'version': True, 'files': []}):
            self.assertIn('registry.invalid', self.gaps(self.scan(registry=registry)))

    def test_quarantine_preserves_findings_and_requires_independent_isolation(self):
        raw = self.put('Q.lean', 'example := by sorry')
        quarantine = {'version': 1, 'entries': [{'path': 'Q.lean', 'sha256': sha(raw), 'statement': 'example', 'reason': 'unproved', 'attempts': ['kernel rejects'], 'owner': 'none'}]}
        result = self.scan(quarantine=quarantine)
        self.assertIn('lean.sorry', self.codes(result))
        self.assertTrue(result['findings'][0]['quarantine_requested'])
        self.assertEqual(result['result'], 'FAIL')
        self.assertIn('quarantine_isolation_and_closed_dependencies', result['semantic_receipts_required'])
        quarantine['entries'][0]['sha256'] = '0' * 64
        self.assertIn('quarantine.invalid', self.gaps(self.scan(quarantine=quarantine)))

    def test_empty_tree_never_manufactures_layer_a_green(self):
        result = self.scan()
        self.assertEqual(result['result'], 'INCOMPLETE')
        self.assertEqual(result['static_result'], 'PASS')
        self.assertIn('kernel_and_declared_axioms', result['semantic_receipts_required'])
        self.assertIn('status_index_and_nonvacuity', result['semantic_receipts_required'])
        self.assertFalse(result['mathematical_soundness_certified'])

    def test_cli_json_and_failure_exit(self):
        self.put('A.lean', 'example := by sorry')
        proc = subprocess.run([sys.executable, '-B', str(TOOL), 'scan', '--root', str(self.root)], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)['result'], 'FAIL')

    def test_cli_invalid_json_manifest_is_explicit_failure(self):
        path = self.root / 'bad.json'
        path.write_text('{"version":1,"version":1,"files":[]}')
        proc = subprocess.run([sys.executable, '-B', str(TOOL), 'scan', '--root', str(self.root), '--registry', str(path)], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)['result'], 'FAIL')

    def test_kernel_check_bypass_is_a_trust_finding(self):
        self.put('A.lean', 'set_option debug.skipKernelTC true\n')
        self.assertIn('lean.skipKernelTC', self.codes(self.scan()))

    def test_replaced_tracked_file_kind_is_gap(self):
        self.put('A.lean', 'example := by sorry')
        (self.root / 'A.lean').unlink()
        (self.root / 'A.lean').mkdir()
        self.assertIn('file.non_regular', self.gaps(self.scan()))

    def test_symlink_ancestor_is_gap_even_when_destination_is_readable(self):
        self.put('formal/A.lean', 'example := by sorry')
        (self.root / 'formal').rename(self.root / 'moved')
        (self.root / 'formal').symlink_to('moved', target_is_directory=True)
        result = self.scan()
        self.assertIn('file.symlink', self.gaps(result))
        self.assertEqual(result['read_count'], 0)

    def test_python_complex_literals_are_floating_point(self):
        raw = self.put('bound.py', 'bound = 1j')
        self.assertIn('certification.float', self.codes(self.scan(registry=self.registry('bound.py', raw))))

    def test_explicit_noncertifying_float_is_not_a_bound_finding(self):
        raw = self.put('ui.py', 'opacity = 0.25')
        result = self.scan(registry=self.registry('ui.py', raw, 'noncertifying'))
        self.assertEqual(result['findings'], [])
        self.assertEqual(result['files'][0]['classification'], 'noncertifying')

    def test_registry_extends_proof_candidate_inventory(self):
        raw = self.put('custom.data', 'formal proof')
        result = self.scan(registry=self.registry('custom.data', raw, 'proof-source'))
        self.assertIn('custom.data', result['proof_candidates'])
        self.assertIn('proof.unsupported', self.gaps(result))

    def test_archive_declared_signatures_and_suffixes_and_negative(self):
        for signature in (b'PK\x03\x04', b'PK\x05\x06', b'\x1f\x8b', b'BZh', b'\xfd7zXZ\x00', b'7z\xbc\xaf\x27\x1c', b'Rar!', b'\x28\xb5\x2f\xfd', b'_' * 257 + b'ustar'):
            self.assertTrue(audit.is_archive('unknown.bin', signature))
        for suffix in audit.ARCHIVE_SUFFIXES:
            self.assertTrue(audit.is_archive('file' + suffix))
        self.assertFalse(audit.is_archive('normal.txt', b'ordinary text'))

    def test_quarantine_missing_reason_owner_or_attempts_is_invalid(self):
        raw = self.put('Q.lean', 'example := by sorry')
        row = {'path': 'Q.lean', 'sha256': sha(raw), 'statement': 'example', 'reason': 'unproved', 'attempts': ['kernel rejects'], 'owner': 'none'}
        for field, value in (('reason', ''), ('statement', ''), ('owner', 'someone'), ('attempts', [])):
            bad = dict(row, **{field: value})
            result = self.scan(quarantine={'version': 1, 'entries': [bad]})
            self.assertIn('quarantine.invalid', self.gaps(result))
            self.assertIn('lean.sorry', self.codes(result))

    def test_json_numbers_and_duplicate_keys_refused_valid_roundtrip(self):
        path = self.root / 'input.json'
        for raw in ('{"x":1.5}', '{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'):
            path.write_text(raw)
            with self.assertRaises(ValueError):
                audit.load_json(path)
        path.write_text('{"version":1,"files":[]}')
        self.assertEqual(audit.load_json(path), {'version': 1, 'files': []})

    def test_empty_scan_cli_is_incomplete_nonzero(self):
        proc = subprocess.run([sys.executable, '-B', str(TOOL), 'scan', '--root', str(self.root)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)['result'], 'INCOMPLETE')

    def test_subdirectory_request_scans_repository_root(self):
        self.put('outside.lean', 'example := by sorry')
        self.put('inside/ok.txt', 'ordinary')
        result = audit.scan(self.root / 'inside')
        self.assertEqual(result['tracked_count'], 2)
        self.assertEqual(result['root'], str(self.root))
        self.assertIn('lean.sorry', self.codes(result))

    def test_lfs_pointer_is_not_mistaken_for_source(self):
        self.put('A.lean', 'version https://git-lfs.github.com/spec/v1\noid sha256:' + 'a' * 64 + '\nsize 1024\n')
        self.assertIn('file.lfs_pointer', self.gaps(self.scan()))

    def test_compiled_lean_artifacts_are_explicit_coverage_gaps(self):
        self.put('A.olean', b'compiled bytes')
        self.assertIn('proof.unsupported', self.gaps(self.scan()))

    def test_coverage_gap_binds_exact_bytes_and_tool_source(self):
        raw = self.put('archive.zip', b'PK\x03\x04opaque')
        result = self.scan()
        self.assertEqual(len(result['coverage_gaps']), 1)
        self.assertEqual(result['coverage_gaps'][0].get('sha256'), sha(raw))
        self.assertEqual(result.get('scanner_sha256'), sha(TOOL.read_bytes()))
        self.assertEqual(result['coverage_gaps'][0].get('kind'), 'archive')

    def test_unmerged_index_stages_are_coverage_gaps(self):
        self.put('A.lean', 'example := by sorry')
        oid = subprocess.run(['git', '-C', str(self.root), 'rev-parse', ':A.lean'], capture_output=True, text=True, check=True).stdout.strip()
        entries = '0 ' + '0' * 40 + '\tA.lean\n' + '100644 ' + oid + ' 1\tA.lean\n100644 ' + oid + ' 2\tA.lean\n'
        subprocess.run(['git', '-C', str(self.root), 'update-index', '--index-info'], input=entries, text=True, check=True)
        self.assertIn('git.conflict', self.gaps(self.scan()))

    def test_all_declared_trust_tokens_and_nonmatching_names(self):
        values = ('sorryAx', 'unsafe', 'implemented_by', 'extern', 'ofReduceBool', 'ofReduceNat', 'trustCompiler', 'skipKernelTC')
        self.put('A.lean', '\n'.join(values))
        self.assertEqual(self.codes(self.scan()), {'lean.' + value for value in values})
        self.put('A.lean', '\n'.join('prefix_' + value + '_suffix' for value in values))
        self.assertEqual(self.codes(self.scan()), set())

    def test_all_declared_proof_suffixes_are_enumerated(self):
        suffixes = ('.lean', '.olean', '.ilean', '.v', '.agda', '.thy', '.tex', '.typ', '.ipynb', '.smt2', '.pvs', '.dfy', '.fst', '.fsti', '.idr', '.lagda')
        for suffix in suffixes:
            self.put('proof' + suffix, '')
        self.put('ordinary.txt', '')
        result = self.scan()
        self.assertEqual(set(result['proof_candidates']), {'proof' + suffix for suffix in suffixes})
        self.assertEqual(len(result['coverage_gaps']), len(suffixes) - 1)

    def test_non_git_root_is_explicit_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit.scan(Path(directory))
        self.assertIn('git.enumeration', self.gaps(result))


if __name__ == '__main__':
    unittest.main()
