# Appended to tests/test_evidence_profile.py to pin properties that single-point mutants showed unpinned.


class MutationGapPins(unittest.TestCase):
    def test_unresolved_or_missing_author_never_yields_yes(self):
        # Survivor P2: provider_distinct must not compare reviewers against an author naming no family.
        for author in ('not disclosed', 'possibly OpenAI', None):
            fields = {'review_providers': ['xAI']}
            if author is not None:
                fields['author_provider'] = author
            with self.subTest(author=author):
                self.assertEqual(ep.provider_distinct(node('PROVED_REVIEWED', **fields)), 'not recorded')

    def test_version_token_must_match_whole_token(self):
        # Survivor F5: a version-shaped prefix does not license trailing text; F12/F13: no disjunction or comma lists.
        for text in ('GPT 5,Anthropic', 'OpenAI 4.7;not', 'OpenAI or ChatGPT', 'OpenAI,ChatGPT', 'OpenAI & Codex'):
            with self.subTest(text=text):
                self.assertIsNone(ep.family(text))

    def test_text_label_reports_each_provider_distinct_value(self):
        # Survivor L6: the text tree must carry the computed value, not only 'not recorded'.
        graph = {'schema_version': 1, 'object': 'LABEL', 'edges': [], 'nodes': {
            'math.yes': node('PROVED_REVIEWED', author_provider='OpenAI', review_providers=['xAI']),
            'math.no': node('PROVED_REVIEWED', author_provider='OpenAI', review_providers=['Codex'])}}
        GATE.validate_graph_fail_closed(graph)
        self.assertEqual(ep.tree_lines(GATE, graph, 'math.yes')[0],
                         'math.yes  PROVED_REVIEWED  [author OpenAI; reviewers xAI; provider-distinct yes]')
        self.assertEqual(ep.tree_lines(GATE, graph, 'math.no')[0],
                         'math.no  PROVED_REVIEWED  [author OpenAI; reviewers Codex; provider-distinct no]')

    def test_identity_mode_opens_the_graph_path_once(self):
        # Survivor C2: a second read before the identity read is invisible to the read_bytes hook; count opens.
        from unittest import mock
        raw = json.dumps(synthetic()).encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'graph.json'
            path.write_bytes(raw)
            opened = []
            real_open = io.open

            def counting_open(file, *args, **kwargs):
                if pathlib.Path(file) == path:
                    opened.append(file)
                return real_open(file, *args, **kwargs)

            with mock.patch('io.open', counting_open), mock.patch('builtins.open', counting_open):
                code, _ = run(['--graph', str(path), '--input-identity', '--json'])
        self.assertEqual(code, 0)
        self.assertEqual(len(opened), 1)

    def test_multiple_roots_are_separated_by_one_blank_line(self):
        # Survivor C4: text output for several selected nodes keeps the documented separator.
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'graph.json'
            path.write_text(json.dumps(synthetic()))
            code, text = run(['--graph', str(path), '--node', 'math.leaf', '--node', 'math.boundary'])
        self.assertEqual(code, 0)
        self.assertIn('Unsatisfied required dependencies: none\n\nmath.boundary  REFUTED', text)
