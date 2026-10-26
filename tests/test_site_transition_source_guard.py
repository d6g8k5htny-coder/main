"""Pure merge-identity boundary controls; no Git, network or project imports."""
import hashlib
import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'tools/site_transition_source_guard.py'
SPEC = importlib.util.spec_from_file_location('site_transition_source_guard', SOURCE)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)

BASE, HEAD, OTHER, TREE = ('1' * 40, '2' * 40, '3' * 40, '4' * 40)
AUTHOR = b'author Fixture <fixture@example.invalid> 1700000000 +0000'
COMMITTER = b'committer Fixture <fixture@example.invalid> 1700000000 +0000'


def literal_commit(parents=(BASE, HEAD), extra=(), message=b'Controlled merge fixture\n'):
    lines = [b'tree ' + TREE.encode()]
    lines += [b'parent ' + parent.encode() for parent in parents]
    return b'\n'.join(lines + [AUTHOR, COMMITTER] + list(extra)) + b'\n\n' + message


def arguments(raw=None, **changes):
    """Independent Git object framing constructs the declared fixture identity."""
    raw = literal_commit() if raw is None else raw
    commit = hashlib.sha1(b'commit ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest()
    result = dict(event_name='pull_request', repository='d6g8k5htny-coder/main',
                  event_ref='refs/pull/501/merge', event_sha=commit, pr_number=501,
                  event_head_sha=HEAD, event_base_sha=BASE, checked_commit=commit,
                  actual_commit_raw=raw, frozen_base_commit=BASE)
    result.update(changes)
    return result


class MergeJoinControls(unittest.TestCase):
    def refuse(self, **values):
        with self.assertRaises(guard.SourceJoinError):
            guard.verify_merge_join(**values)

    def test_stale_event_base_preserves_observation_and_accepts_frozen_actual_parents(self):
        # Break caught: the event base is incorrectly substituted for the frozen
        # actual first parent, rejecting a valid currently tested composition.
        values = arguments(event_base_sha=OTHER)
        try:
            record = guard.verify_merge_join(**values)
        except guard.SourceJoinError as error:
            self.fail('Valid frozen ordered parents were refused because of stale event metadata: ' + str(error))
        self.assertEqual(record['pull_request_base_sha'], OTHER)
        self.assertEqual(record['candidate_base_commit'], BASE)
        self.assertEqual(record['actual_merge_parents'], [BASE, HEAD])
        self.assertEqual(record['checked_commit'], values['checked_commit'])
        self.assertEqual(record['actual_tree_sha'], TREE)

    def test_matching_event_and_actual_base_still_passes(self):
        values = arguments()
        expected = {
            'event_name': 'pull_request', 'repository': 'd6g8k5htny-coder/main',
            'event_ref': 'refs/pull/501/merge', 'event_sha': values['event_sha'],
            'pull_request_number': 501, 'pull_request_head_sha': HEAD,
            'pull_request_base_sha': BASE, 'checked_commit': values['checked_commit'],
            'actual_tree_sha': TREE, 'actual_merge_parents': [BASE, HEAD],
            'candidate_base_commit': BASE,
        }
        self.assertEqual(guard.verify_merge_join(**values), expected)

    def test_wrong_frozen_first_parent_refuses(self):
        # Even agreeing event metadata must not authorize a moved frozen base.
        self.refuse(**arguments(frozen_base_commit=OTHER))
        self.refuse(**arguments(literal_commit(parents=(OTHER, HEAD)), event_base_sha=OTHER))

    def test_wrong_second_parent_refuses(self):
        self.refuse(**arguments(literal_commit(parents=(BASE, OTHER))))
        self.refuse(**arguments(event_head_sha=OTHER))

    def test_reversed_parents_refuse(self):
        self.refuse(**arguments(literal_commit(parents=(HEAD, BASE))))

    def test_third_parent_refuses(self):
        self.refuse(**arguments(literal_commit(parents=(BASE, HEAD, OTHER))))

    def test_missing_parent_refuses(self):
        self.refuse(**arguments(literal_commit(parents=(BASE,))))
        self.refuse(**arguments(literal_commit(parents=())))

    def test_wrong_checkout_or_event_sha_refuses(self):
        self.refuse(**arguments(checked_commit=OTHER))
        self.refuse(**arguments(event_sha=OTHER))

    def test_commit_body_hash_mismatch_refuses(self):
        values = arguments()
        values['actual_commit_raw'] += b'Additional message bytes\n'
        self.refuse(**values)

    def test_wrong_repository_event_or_pr_ref_refuses(self):
        for changes in (
            {'repository': 'other/repository'}, {'repository': None},
            {'event_name': 'pull_request_target'}, {'event_name': 'push'},
            {'event_ref': 'refs/heads/main'}, {'event_ref': 'refs/pull/502/merge'},
            {'event_ref': 'refs/pull/501/head'}, {'event_ref': 'refs/pull/0501/merge'},
            {'pr_number': True}, {'pr_number': 0}, {'pr_number': -1},
            {'pr_number': '501'}, {'pr_number': 501.0},
        ):
            with self.subTest(changes=changes):
                self.refuse(**arguments(**changes))

    def test_malformed_sha_values_refuse(self):
        for field in ('event_sha', 'event_head_sha', 'event_base_sha',
                      'checked_commit', 'frozen_base_commit'):
            for value in (None, False, 0, [], 'A' * 40, 'a' * 39, 'a' * 41, 'g' * 40, 'a' * 40 + '\n'):
                with self.subTest(field=field, value=value):
                    self.refuse(**arguments(**{field: value}))

    def test_malformed_commit_header_refuses(self):
        original = literal_commit()
        malformed = {
            'missing separator': original.replace(b'\n\n', b'\n', 1),
            'bad tree': original.replace(TREE.encode(), b'f' * 39, 1),
            'duplicate tree': literal_commit(extra=(b'tree ' + TREE.encode(),)),
            'duplicate author': literal_commit(extra=(AUTHOR,)),
            'duplicate committer': literal_commit(extra=(COMMITTER,)),
            'missing author': original.replace(AUTHOR + b'\n', b'', 1),
            'missing committer': original.replace(COMMITTER + b'\n', b'', 1),
            'header NUL': literal_commit(extra=(b'encoding UTF-8\0',)),
            'header CR': literal_commit(extra=(b'encoding UTF-8\r',)),
            'tab continuation': literal_commit(extra=(b'gpgsig signature', b'\tcontinued')),
            'uppercase header': literal_commit(extra=(b'Encoding UTF-8',)),
            'empty header value': literal_commit(extra=(b'encoding ',)),
            'orphan continuation': original.replace(AUTHOR, b' orphan\n' + AUTHOR, 1),
            'identity continuation': literal_commit(extra=(b' identity continuation',)),
        }
        for name, raw in malformed.items():
            with self.subTest(name=name):
                self.refuse(**arguments(raw))

    def test_multiline_signature_header_preserves_literal_parent_join(self):
        raw = literal_commit(extra=(b'gpgsig -----BEGIN PGP SIGNATURE-----',
                                    b' ', b' encoded-signature',
                                    b' -----END PGP SIGNATURE-----'))
        result = guard.verify_merge_join(**arguments(raw))
        self.assertEqual(result['actual_merge_parents'], [BASE, HEAD])
        self.assertEqual(result['actual_tree_sha'], TREE)

    def test_header_line_bound_accepts_128_and_refuses_129(self):
        # Five required lines plus 123 extension lines is the accepted boundary.
        extension = tuple(b'x-fixture value' for _ in range(123))
        self.assertEqual(guard.verify_merge_join(**arguments(literal_commit(extra=extension)))['actual_tree_sha'], TREE)
        self.refuse(**arguments(literal_commit(extra=extension + (b'x-fixture extra',))))

    def test_commit_byte_bound_accepts_65536_and_refuses_65537(self):
        header = literal_commit(message=b'')
        exact = header + b'x' * (65536 - len(header))
        self.assertEqual(guard.verify_merge_join(**arguments(exact))['actual_tree_sha'], TREE)
        self.refuse(**arguments(exact + b'x'))

    def test_mutable_or_nonbyte_commit_input_refuses(self):
        for value in (None, '', bytearray(literal_commit()), memoryview(literal_commit())):
            with self.subTest(value_type=type(value).__name__):
                values = arguments()
                values['actual_commit_raw'] = value
                self.refuse(**values)

    def test_message_is_hashed_as_bytes_without_becoming_header(self):
        raw = literal_commit(message=b'parent ' + OTHER.encode() + b'\n\x00non-UTF8:\xff\n')
        self.assertEqual(guard.verify_merge_join(**arguments(raw))['actual_merge_parents'], [BASE, HEAD])


if __name__ == '__main__':
    unittest.main()
