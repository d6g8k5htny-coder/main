

class ArchiveRefusalPins(unittest.TestCase):
    """Refusals that single-point mutants showed unpinned. Reuses the StrictMembers fixture without rerunning its tests."""
    setUp = StrictMembers.setUp
    git = StrictMembers.git
    commit = StrictMembers.commit
    assert_refused_before_output = StrictMembers.assert_refused_before_output
    replace_named_commit_with_pending_changes = StrictMembers.replace_named_commit_with_pending_changes

    def test_checkout_only_member_is_named(self):
        self.git('rm', '-q', 'LICENSE')
        self.commit('drop LICENSE')
        (self.repo / 'LICENSE').write_text('synthetic LICENSE\n')
        self.replace_named_commit_with_pending_changes()
        self.assert_refused_before_output(
            re.escape("commit and checkout disagree on members: commit only=[]; checkout only=['LICENSE']") + '$')

    def test_default_call_refuses_private_and_compiled_paths_under_a_listed_name(self):
        cases = (('__pycache__/module.txt', 'private/generated path refused'),
                 ('Sandbox/note.txt', 'private/generated path refused'),
                 ('compiled.pyc', 'compiled payload refused'))
        for inner, pattern in cases:
            with self.subTest(path='SUPPORT.md/' + inner):
                self.git('rm', '-q', 'SUPPORT.md')
                target = self.repo / 'SUPPORT.md' / inner
                target.parent.mkdir(parents=True)
                target.write_text('synthetic descendant\n')
                self.commit('descendant ' + inner)
                try:
                    with self.assertRaisesRegex(ValueError, '^' + pattern):
                        build_source_archive(self.repo, self.out, 1700000000)
                    self.assertFalse(self.out.parent.exists())
                finally:
                    self.git('reset', '-q', '--hard', 'HEAD~1')

    def test_gitlink_at_a_listed_name_is_refused_as_irregular(self):
        self.git('rm', '-q', 'LICENSE')
        self.commit('drop LICENSE')
        self.git('update-index', '--add', '--cacheinfo', '160000,' + self.git('rev-parse', 'HEAD') + ',LICENSE')
        (self.repo / 'LICENSE').mkdir()
        self.git('commit', '-qm', 'gitlink at LICENSE')
        self.assertEqual(self.git('status', '--porcelain'), '')
        with self.assertRaisesRegex(ValueError, '^tracked payload is not a regular file: LICENSE$'):
            build_source_archive(self.repo, self.out, 1700000000)

    def test_invalid_epoch_and_project_metadata_are_refused(self):
        with self.assertRaisesRegex(ValueError, '^invalid SOURCE_DATE_EPOCH$'):
            build_source_archive(self.repo, self.out, -1)
        (self.repo / 'pyproject.toml').write_text('[project]\nname="universal-law-query"\n')
        self.commit('no version')
        with self.assertRaisesRegex(ValueError, '^project name/version required$'):
            build_source_archive(self.repo, self.out, 1700000000)
        self.assertFalse(self.out.parent.exists())

    def test_failed_git_command_is_reported(self):
        plain = self.base / 'not-a-repository'
        plain.mkdir()
        with self.assertRaisesRegex(ValueError, '^git command failed'):
            build_source_archive(plain, self.out, 1700000000)

    def test_non_sha1_commit_is_refused(self):
        repo = self.base / 'sha256-checkout'
        repo.mkdir()
        subprocess.run(['git', 'init', '-q', '--object-format=sha256', str(repo)], check=True)
        for name in sorted(ARCHIVE_PAYLOADS):
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic ' + name + '\n')
        (repo / 'pyproject.toml').write_text('[project]\nname="universal-law-query"\nversion="0.1.0"\n')
        for args in (('config', 'user.email', 'test@example.com'), ('config', 'user.name', 'test'),
                     ('add', '-A'), ('commit', '-qm', 'sha256 fixture')):
            subprocess.run(['git', '-C', str(repo), *args], check=True)
        with self.assertRaisesRegex(ValueError, '^exact commit required$'):
            build_source_archive(repo, self.out, 1700000000)

    def test_gzip_header_records_the_source_date_epoch(self):
        build_source_archive(self.repo, self.out, 1700000000)
        header = self.out.read_bytes()[:10]
        self.assertEqual(header[:3], b'\x1f\x8b\x08')
        self.assertEqual(int.from_bytes(header[4:8], 'little'), 1700000000)
