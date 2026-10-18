

class MutationGapPins(unittest.TestCase):
    """Each case breaks exactly one rule of an otherwise valid record, so no other rule can mask it."""

    def mutated(self, change):
        data = example()
        change(data)
        return rc.validate(data)

    @staticmethod
    def as_delta(data):
        data['records'][0].update(delta=True)
        data['records'][0]['subject'].update(commit='b' * 40)

    def test_each_rule_rejects_on_its_own(self):
        subject = lambda d, **kw: d['records'][0]['subject'].update(**kw)
        cases = {
            'id with trailing punctuation': (lambda d: d['records'][0].update(id='EX-001!'), 'records[0].id'),
            'unknown repository on a delta record': (
                lambda d: (self.as_delta(d), subject(d, repository='elsewhere')), 'records[0].subject.repository'),
            '41-character commit on a delta record': (
                lambda d: (self.as_delta(d), subject(d, commit='b' * 41)), 'records[0].subject.commit'),
            'backslash in path': (lambda d: subject(d, path='reviews\\REVIEW.md'), 'records[0].subject.path'),
            'not_applicable with evidence': (lambda d: d['records'][0].update(state='not_applicable'),
                                             'records[0]: state not_applicable'),
            'performer with an extra key': (lambda d: d['records'][0]['performer'].update(extra='x'),
                                            'records[0].performer'),
            'whitespace-only exposure': (lambda d: d['records'][0].update(exposure='   '), 'records[0].exposure'),
            'empty exposure': (lambda d: d['records'][0].update(exposure=''), 'records[0].exposure'),
            'whitespace-only author session': (lambda d: d['author'].update(session=' '), 'author.session'),
        }
        for name, (change, prefix) in cases.items():
            with self.subTest(case=name):
                errors = self.mutated(change)
                self.assertTrue(errors, 'accepted')
                self.assertTrue(all(e.startswith(prefix) for e in errors), errors)

    def test_malformed_subject_is_not_sent_to_git(self):
        from unittest import mock
        data = example()
        data['records'][0]['subject']['commit'] = 1
        calls = []

        def fake_blob(repo, commit, path):
            calls.append(commit)
            return None

        with mock.patch.object(rc, '_git_blob', fake_blob):
            errors = rc.validate(data, repos={'Math-': ROOT})
        self.assertNotIn(1, calls)
        self.assertTrue(any(e.startswith('records[0].subject.commit') for e in errors), errors)

    def test_each_evidence_rule_rejects_on_its_own(self):
        comment = {'kind': 'github_comment', 'repository': 'main', 'ref': '6159767786'}
        cases = {
            'unknown evidence repository': (dict(comment, repository='elsewhere'), 'records[0].evidence[0].repository'),
            'ref with trailing text': (dict(comment, ref='123abc'), 'records[0].evidence[0].ref'),
            'ref with a leading zero': (dict(comment, ref='0123'), 'records[0].evidence[0].ref'),
            'non-native conclusion': (dict(FAILED_RUN, conclusion='passed'), 'records[0].evidence[0].conclusion'),
            'neutral expected conclusion': (dict(FAILED_RUN, expected_conclusion='neutral'),
                                            'records[0].evidence[0].expected_conclusion'),
        }
        for item in (comment, FAILED_RUN):
            self.assertEqual(self.mutated(lambda d: d['records'][0].update(evidence=[copy.deepcopy(item)])), [])
        for name, (item, prefix) in cases.items():
            with self.subTest(case=name):
                errors = self.mutated(lambda d: d['records'][0].update(evidence=[copy.deepcopy(item)]))
                self.assertTrue(errors, 'accepted')
                self.assertTrue(all(e.startswith(prefix) for e in errors), errors)

    def test_key_order_does_not_make_evidence_distinct(self):
        data = example()
        before = rc.aggregate([data])['counts']['distinct_evidence_items']
        twin = copy.deepcopy(data['records'][0])
        twin.update(id='EX-003', axis='custody')
        twin['evidence'] = [dict(reversed(list(item.items()))) for item in twin['evidence']]
        data['records'].append(twin)
        self.assertEqual(rc.validate(data), [])
        self.assertEqual(rc.aggregate([data])['counts']['distinct_evidence_items'], before)

    def test_repo_option_does_not_verify_main_records(self):
        data = example()
        data['records'][0]['subject'].update(repository='main', commit=rc.BASELINE['main'], path='README.md',
                                             blob='c' * 40)
        self.assertEqual(rc.validate(data), [])
        flags = ['-O'] if sys.flags.optimize else []
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'main_subject.json'
            path.write_text(json.dumps(data))
            run = subprocess.run([sys.executable, '-B', *flags, '-S', str(TOOL), '--repo', str(ROOT), str(path)],
                                 capture_output=True, text=True, timeout=60)
        self.assertEqual((run.returncode, run.stderr), (0, ''))

    def test_unparseable_files_are_reported_and_later_files_are_still_read(self):
        valid = EXAMPLE.read_text(encoding='utf-8')
        cases = {'not JSON': '{not json', 'duplicate key': '{"schema": 1, ' + valid.lstrip()[1:],
                 'non-finite number': valid.replace('"status_authority": false', '"status_authority": NaN')}
        flags = ['-O'] if sys.flags.optimize else []
        for name, text in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                bad = pathlib.Path(tmp) / 'bad.json'
                bad.write_text(text, encoding='utf-8')
                run = subprocess.run([sys.executable, '-B', *flags, '-S', str(TOOL), str(bad), str(EXAMPLE)],
                                     capture_output=True, text=True, timeout=60)
                self.assertEqual((run.returncode, run.stderr), (1, ''))
                report = json.loads(run.stdout)
                self.assertFalse(report['files'][str(bad)]['valid'])
                self.assertEqual(report['files'][str(EXAMPLE)], {'valid': True, 'errors': []})
