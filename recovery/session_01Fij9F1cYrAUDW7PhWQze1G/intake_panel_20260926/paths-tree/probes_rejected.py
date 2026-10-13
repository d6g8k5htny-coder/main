"""Hypotheses REJECTED: the current checker already behaves correctly. These assert the
correct outcome and pass on orig/ (and must keep passing on patched/)."""
import copy
import json
import unittest
from harness import Probe, REPO, HEAD, BASE, SOURCE, blob, sha


class R_Paths(Probe):
    def test_r01_unicode_and_nfc_nfd_rejected(self):
        for name in ['théorème.md', 'théoreme.md', 'α.md', 'x y.md', 'ｘ.md']:
            with self.subTest(name=name):
                old = self.raw.copy(); self.add_file(name); self.reject('unsafe path'); self.raw = old

    def test_r02_control_and_del_chars_rejected_in_intake(self):
        for ch in ['\x7f', '\x85', '\x9f', ' ', '\t', '\x1f']:
            with self.subTest(ch=repr(ch)):
                old = self.raw.copy(); self.add_file(f'a{ch}b.md'); self.reject('unsafe path|invalid changed path'); self.raw = old

    def test_r03_backslash_rejected_in_both_lanes(self):
        old = self.raw.copy(); self.add_file('a\\b.md'); self.reject('invalid changed path'); self.raw = old
        self.pr['head']['repo']['full_name'] = REPO
        self.route_files()[:] = [{'filename': 'docs\\x.md', 'status': 'modified', 'sha': 'd'*40}]
        self.pr['changed_files'] = 1
        self.reject('invalid changed path')

    def test_r16_dotfile_components_rejected(self):
        for name in ['.gitattributes', '.github/workflows/x.md', '..x.md', 'a/.b/c.md', '.../x.md']:
            with self.subTest(name=name):
                old = self.raw.copy(); self.add_file(name); self.reject('unsafe path'); self.raw = old

    def test_r17_noncanonical_rejected(self):
        for path in ['incoming/example//x.md', 'incoming/example/./x.md', 'incoming/example/x.md/', '/incoming/example/x.md', 'incoming//example/x.md']:
            with self.subTest(path=path):
                rows = copy.deepcopy(self.route_files())
                rows.append({'filename': path, 'status': 'added', 'sha': blob(b'x')})
                self.route_files()[:] = rows
                self.pr['changed_files'] = len(rows)
                self.reject('unsafe path|invalid changed path|noncanonical|package tree')
                self.route_files()[:] = rows[:-1]; self.pr['changed_files'] = len(rows) - 1

    def test_r18_suffix_edges(self):
        for name in ['plot.PNG', 'notes.MD']:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.add_file(name, b'\x89PNG\r\n\x1a\nxx' if name.endswith('PNG') else b'ok\n')
                self.accept(); self.raw = old
        for name in ['x.md ', 'x.', 'x', 'x.md.', 'x.md.bak']:
            with self.subTest(name=name):
                old = self.raw.copy(); self.add_file(name); self.reject('unsupported file extension|unportable path component'); self.raw = old

    def test_r20_identity_and_result_names_exact(self):
        self.files = {'Identity.json': self.files['IDENTITY.json'], 'RESULT.md': self.raw['RESULT.md']}
        self.install(); self.reject('IDENTITY.json required')
        self.files = {'IDENTITY.json': self.files['Identity.json'], 'result.md': self.raw['RESULT.md']}
        self.install(); self.reject('RESULT.md and IDENTITY.json required|exactly cover|unknown artifact')

    def test_r21_protected_names_case_folded(self):
        for name in ['status.md', 'Status.notes.md', 'Claims/x.md', 'proof_index/x.md', 'Landing_Claims.json', 'status/x.md']:
            with self.subTest(name=name):
                old = self.raw.copy(); self.add_file(name); self.reject('protected'); self.raw = old
        for name in ['STATUS_final.md', 'claims.json', 'myclaims/x.md']:   # INFO: not protected names by design
            with self.subTest(name=name):
                old = self.raw.copy(); self.add_file(name, b'{}' if name.endswith('.json') else b'x\n'); self.accept(); self.raw = old

    def test_r15_package_id_regex_boundaries(self):
        for pid, ok in [('a'*64, True), ('a'*65, False), ('Abc', False), ('-abc', False), ('abc-', True), ('a b', False), ('a.b', False), ('claims', True), ('status', True)]:
            with self.subTest(pid=pid):
                self.package = f'incoming/{pid}/'; self.save()
                if ok:
                    self.accept()
                else:
                    self.reject('invalid package id|unportable path component')


class R_RowsAndStatus(Probe):
    def test_r04_previous_filename_with_added_status_rejected(self):
        self.route_files()[0].update(previous_filename='incoming/other/x.md')
        self.reject('add-only')

    def test_r05_every_non_added_status_rejected(self):
        for status in ['removed', 'modified', 'renamed', 'copied', 'changed', 'unchanged', None, 'Added']:
            with self.subTest(status=status):
                self.route_files()[0]['status'] = status
                self.reject('add-only')

    def test_r25_pagination_boundaries(self):
        self.pr['head']['repo']['full_name'] = REPO
        for count, pages in [(100, 1), (101, 2), (200, 2), (201, 3)]:
            with self.subTest(count=count):
                rows = [{'filename': f'docs/i{i}.md', 'status': 'modified', 'sha': 'd'*40} for i in range(count)]
                self.pr['changed_files'] = count
                for k in range(1, 4):
                    self.api.routes.pop(f'/repos/{REPO}/pulls/7/files?per_page=100&page={k}', None)
                for k in range(1, pages+1):
                    self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page={k}'] = rows[(k-1)*100:k*100]
                self.assertEqual(self.run_check()['lane'], 'maintainer-engineering')
                self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page={pages}'] = rows[(pages-1)*100:k*100][:-1]
                self.reject('incomplete file listing')

    def test_r13_empty_and_identity_only_packages_rejected(self):
        self.pr['changed_files'] = 0; self.reject('unsupported changed file count')
        self.files = {'IDENTITY.json': self.files['IDENTITY.json']}; self.install(); self.reject('RESULT.md and IDENTITY.json required')
        self.raw = {'RESULT.md': b' \n'}; self.save(); self.reject('RESULT.md and IDENTITY.json required')


class R_Trees(Probe):
    def test_r06_truncated_variants_rejected(self):
        for value in [True, 'false', None, 0]:
            with self.subTest(value=value):
                t = self.head_tree(); t['truncated'] = value
                self.reject('truncated')
        self.head_tree()['truncated'] = False
        self.head_tree()['tree'] = None; self.reject('truncated')

    def test_r07_modes_as_package_entry_and_as_source(self):
        for mode, kind in [('120000', 'blob'), ('100755', 'blob'), ('160000', 'commit'), ('100644', 'commit'), ('100644', 'tree')]:
            with self.subTest(mode=mode, kind=kind):
                entry = copy.deepcopy(self.head_tree()['tree'][0]); entry.update(mode=mode, type=kind)
                saved = self.head_tree()['tree'][0]; self.head_tree()['tree'][0] = entry
                self.reject('regular non-executable|package tree does not match')
                self.head_tree()['tree'][0] = saved
        route = f'/repos/d6g8k5htny-coder/Math-/git/trees/{SOURCE}?recursive=1'
        for mode, kind in [('120000', 'blob'), ('100755', 'blob'), ('160000', 'commit'), ('040000', 'tree')]:
            with self.subTest(source_mode=mode):
                self.api.routes[route]['tree'][0].update(mode=mode, type=kind)
                self.reject('regular non-executable public file')

    def test_r08_submodule_as_package_or_inside_package_rejected(self):
        # gitlink at incoming/<id> itself
        self.route_files()[:] = [{'filename': 'incoming/example', 'status': 'added', 'sha': 'd'*40}]
        self.pr['changed_files'] = 1
        self.head_tree()['tree'] = [{'path': 'incoming/example', 'mode': '160000', 'type': 'commit', 'sha': 'd'*40}]
        self.reject('incoming-only')
        # gitlink nested inside the package, listed as a changed file
        self.save()
        self.route_files().append({'filename': self.package + 'vendor', 'status': 'added', 'sha': 'd'*40})
        self.pr['changed_files'] += 1
        self.head_tree()['tree'].append({'path': self.package + 'vendor', 'mode': '160000', 'type': 'commit', 'sha': 'd'*40})
        self.reject('regular non-executable')
        # .gitmodules must be outside the package
        self.save()
        self.route_files().append({'filename': '.gitmodules', 'status': 'added', 'sha': 'd'*40})
        self.pr['changed_files'] += 1
        self.reject('incoming-only|unsafe path')

    def test_r09_one_package_rule_and_base_collisions(self):
        self.add_file('nested/deeper/x.md'); self.accept()          # nesting inside one package is legitimate
        self.add_file('incoming/other/y.md'); self.accept()         # 'incoming' as an inner directory name is fine
        rows = self.route_files(); rows[0]['filename'] = 'incoming/other/RESULT.md'
        self.reject('one package')
        self.save()
        for entry in [{'path': 'incoming/example', 'mode': '120000', 'type': 'blob', 'sha': 'd'*40},
                      {'path': 'incoming/example', 'mode': '160000', 'type': 'commit', 'sha': 'd'*40},
                      {'path': 'incoming/example/', 'mode': '040000', 'type': 'tree', 'sha': 'd'*40},
                      {'path': 'incoming/example/deep/x', 'mode': '100644', 'type': 'blob', 'sha': 'd'*40}]:
            with self.subTest(entry=entry['path'] + entry['mode']):
                self.base_tree()['tree'] = [entry]; self.reject('new package id')
        self.base_tree()['tree'] = [{'path': 'incoming/example-2/x', 'mode': '100644', 'type': 'blob', 'sha': 'd'*40},
                                    {'path': 'incoming/exampl/x', 'mode': '100644', 'type': 'blob', 'sha': 'd'*40}]
        self.accept()                                                # sibling ids are not prefix collisions

    def test_r14_rows_vs_tree_duplicates_and_mismatch(self):
        t = self.head_tree()['tree']
        t.append(copy.deepcopy(t[0])); self.reject('invalid tree entries'); t.pop()
        t.append({'path': self.package + 'extra.md', 'mode': '100644', 'type': 'blob', 'sha': blob(b'e'), 'size': 1}); self.reject('package tree does not match'); t.pop()
        t.append({'path': self.package + 'extra', 'mode': '160000', 'type': 'commit', 'sha': 'd'*40}); self.reject('package tree does not match'); t.pop()
        removed = t.pop(); self.reject('package tree does not match'); t.append(removed)
        t.append({'path': 'Incoming/example/z.md', 'mode': '100644', 'type': 'blob', 'sha': blob(b'z'), 'size': 1}); self.accept(); t.pop()  # not under prefix, not a row: ignored
        rows = self.route_files(); rows.append(copy.deepcopy(rows[0])); self.pr['changed_files'] += 1; self.reject('duplicate changed path')

    def test_r23_tree_entry_size_and_sha_validation(self):
        e = self.head_tree()['tree'][0]
        for size in [None, True, '3', 262145, -1]:
            with self.subTest(size=size):
                saved = e['size']; e['size'] = size; self.reject('size limit'); e['size'] = saved
        e['sha'] = 'D' * 40; self.reject('invalid blob SHA')


class R_Limits(Probe):
    def test_r10_max_package_files_50_ok_51_rejected(self):
        for i in range(47):
            self.raw[f'f{i}.txt'] = b'x'
        self.save(); self.assertEqual(len(self.files), 50); self.accept()
        self.raw['f47.txt'] = b'x'; self.save(); self.assertEqual(len(self.files), 51); self.reject('too many package files')

    def test_r11_max_file_bytes_boundary(self):
        self.raw['big.txt'] = b'a' * 262144; self.save(); self.accept()
        self.raw['big.txt'] = b'a' * 262145; self.save(); self.reject('size limit')

    def test_r12_max_package_bytes_boundary(self):
        for i in range(7):
            self.raw[f'b{i}.txt'] = b'a' * 262144
        self.raw['fill.txt'] = b''
        for _ in range(4):   # fixed point: IDENTITY.json size depends on fill size digits
            self.save()
            total = sum(len(v) for v in self.files.values())
            self.raw['fill.txt'] = b'a' * (len(self.raw['fill.txt']) + 2097152 - total)
        self.save(); self.assertEqual(sum(len(v) for v in self.files.values()), 2097152); self.accept()
        self.raw['fill.txt'] += b'a'; self.save(); self.reject('total size limit')

    def test_r22_source_count_and_path_checks(self):
        src = self.manifest['sources'][0]
        self.manifest['sources'] = [dict(src, path=f'coefficients/s{i}.json') for i in range(8)]
        self.save()   # install() rewrites the source tree route with sources[0] only; add the rest afterwards
        for s in self.manifest['sources'][1:]:
            self.api.routes[f'/repos/d6g8k5htny-coder/Math-/git/trees/{SOURCE}?recursive=1']['tree'].append(
                {'path': s['path'], 'mode': '100644', 'type': 'blob', 'size': len(self.source), 'sha': blob(self.source)})
        self.accept()
        self.manifest['sources'].append(dict(src, path='coefficients/s9.json')); self.save(); self.reject('one to eight')
        self.manifest['sources'] = [dict(src, path='coefficients/../x.json')]; self.save(); self.reject('unsafe path')
        self.manifest['sources'] = [dict(src, path='link/example.json')]; self.save()
        self.api.routes[f'/repos/d6g8k5htny-coder/Math-/git/trees/{SOURCE}?recursive=1']['tree'] = [
            {'path': 'link', 'mode': '120000', 'type': 'blob', 'size': 12, 'sha': blob(b'coefficients')}]   # symlinked directory: not traversed by the tree API
        self.reject('regular non-executable public file')


if __name__ == '__main__':
    unittest.main()
