"""Regression tests for dep_graph_check.py on small synthetic inputs (standard library only).

They encode the coverage findings DG-COV-01..04 of an OpenAI engineering probe (main#389 6325826874):
a duplicate label must not hide a cycle; missing inputs must not pass; the DOT export must not depend on
the hash seed; a change of the directed internal edge set of a KNOWN component must fail.
Every helper invocation runs the real checker in both normal and optimized mode, with site loading
suppressed. Outer unittest optimization and warning flags are not used as child-coverage evidence.

Run from the supplement directory:  python3 -B -m unittest checks.test_dep_graph_check -v
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(HERE, 'dep_graph_check.py')
FILES = ['supp_s1_setting.tex', 'supp_s2_contact.tex', 'supp_s3_coefficient.tex', 'supp_s4_correction.tex',
         'supp_s5_selection.tex', 'supp_s6_limit.tex', 'supp_s7_rate.tex', 'supp_s8_remainder.tex',
         'supp_s9_cusp.tex', 'supp_s10_third.tex', 'supp_s11_ctwo.tex', 'supp_s12_parent.tex', 'supp_s13_cap.tex']


def lemma(label, refs=(), proof_refs=()):
    body = ' '.join(r'\ref{%s}' % r for r in refs)
    pr = ' '.join(r'\ref{%s}' % r for r in proof_refs)
    return r'\begin{lemma}\label{%s} Statement %s.\end{lemma}\begin{proof}Argument %s.\end{proof}' % (label, body, pr) + '\n'


def run(d, *extra, seed='0'):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env['PYTHONHASHSEED'] = seed
    results = []
    for optimize in (False, True):
        command = [sys.executable, '-B'] + (['-O'] if optimize else [])
        command += ['-S', CHECKER, d] + list(extra)
        r = subprocess.run(command, capture_output=True, text=True, env=env, timeout=60)
        if r.stderr:
            raise AssertionError('Unexpected checker stderr (%r): %s' % (command, r.stderr))
        results.append((r.returncode, r.stdout))
    if results[0] != results[1]:
        raise AssertionError('Normal and optimized checker results differ: %r' % (results,))
    return results[0]


class DepGraphCheck(unittest.TestCase):
    def make(self, first, only=None):
        d = tempfile.mkdtemp(prefix='depgraph_')
        self.addCleanup(shutil.rmtree, d, True)
        for fn in (only if only is not None else FILES):
            with open(os.path.join(d, fn), 'w', encoding='utf-8') as f:
                f.write(first if fn == FILES[0] else '')
        return d

    def test_acyclic_passes(self):
        code, out = run(self.make(lemma('lem:a') + lemma('lem:b', proof_refs=['lem:a'])))
        self.assertEqual(code, 0, out)
        self.assertIn('NO_UNEXPLAINED_COMPONENT', out)

    def test_new_cycle_fails(self):
        code, out = run(self.make(lemma('lem:a', proof_refs=['lem:b']) + lemma('lem:b', proof_refs=['lem:a'])))
        self.assertEqual(code, 1, out)
        self.assertIn('UNEXPLAINED_COMPONENT_FOUND', out)

    def test_duplicate_label_is_rejected(self):  # DG-COV-01
        text = lemma('lem:a', proof_refs=['lem:b']) + lemma('lem:b', proof_refs=['lem:a']) + lemma('lem:a')
        code, out = run(self.make(text))
        self.assertEqual(code, 2, out)
        self.assertIn('duplicate labels', out)

    def test_missing_directory_is_rejected(self):  # DG-COV-02
        base = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, base, True)
        code, out = run(os.path.join(base, 'absent'))
        self.assertEqual(code, 2, out)

    def test_missing_section_file_is_rejected(self):  # DG-COV-02
        code, out = run(self.make(lemma('lem:a'), only=FILES[:1]))
        self.assertEqual(code, 2, out)
        self.assertIn('missing section files', out)

    def test_dot_is_seed_independent(self):  # DG-COV-03
        text = lemma('lem:hub', proof_refs=['lem:leaf-%d' % i for i in range(8)])
        text += ''.join(lemma('lem:leaf-%d' % i) for i in range(8))
        d = self.make(text)
        outs = []
        for seed in ('0', '1', '2', '7', '99991'):
            dot = os.path.join(d, 'g%s.dot' % seed)
            code, _ = run(d, '--dot', dot, seed=seed)
            self.assertEqual(code, 0)
            with open(dot, encoding='utf-8') as f:
                outs.append(f.read())
        self.assertEqual(len(set(outs)), 1)

    def test_known_component_exact_edges_pass(self):  # DG-COV-04 control
        # From the committed draft16 report, not from the implementation's snapshot table:
        # S5.26 -> S5.30; S5.29 -> S5.26; S5.30 -> S5.26 and S5.29.
        a, b, c = ('lem:selection-slope', 'lem:selection-capture', 'lem:selection-robust')
        text = lemma(a, proof_refs=[c]) + lemma(b, proof_refs=[a]) + lemma(c, proof_refs=[a, b])
        code, out = run(self.make(text))
        self.assertEqual(code, 0, out)
        self.assertIn('internal edges: 4', out)
        self.assertIn('1 explained in KNOWN, 0 not', out)

    def test_known_component_equal_count_rewire_fails(self):  # DG-COV-04 RED on count-only source
        # Replace C->B by A->B. Same three labels, four edges and one strongly connected component.
        a, b, c = ('lem:selection-slope', 'lem:selection-capture', 'lem:selection-robust')
        text = lemma(a, proof_refs=[b, c]) + lemma(b, proof_refs=[a]) + lemma(c, proof_refs=[a])
        code, out = run(self.make(text))
        self.assertEqual(code, 1, out)
        self.assertIn('CHANGED', out)
        self.assertIn('internal edges: 4', out)
        self.assertIn('0 explained in KNOWN, 1 not', out)

    def test_known_component_with_changed_edges_fails(self):  # DG-COV-04
        # A KNOWN three-member component with all six internal edges (KNOWN_EDGES records four).
        labs = ['lem:selection-slope', 'lem:selection-capture', 'lem:selection-robust']
        text = ''.join(lemma(a, proof_refs=[b for b in labs if b != a]) for a in labs)
        code, out = run(self.make(text))
        self.assertEqual(code, 1, out)
        self.assertIn('CHANGED', out)


    def test_statement_body_references_remain_edges(self):
        text = lemma('lem:a', refs=['lem:b']) + lemma('lem:b', refs=['lem:a'])
        code, out = run(self.make(text))
        self.assertEqual(code, 1, out)
        self.assertIn('proofs: 2; edges: 2', out)
        self.assertIn('UNEXPLAINED_COMPONENT_FOUND', out)

    def test_statement_title_references_cannot_hide_a_cycle(self):
        # Statement-title text is a reference source, not a proof-validity judgment.
        for kind in ('theorem', 'lemma', 'proposition', 'corollary'):
            for command in ('ref', 'eqref'):
                with self.subTest(kind=kind, command=command):
                    title = r'From \%s{lem:b}' % command
                    a = (r'\begin{%s}[%s]\label{lem:a}Statement.\end{%s}'
                         r'\begin{proof}Argument.\end{proof}' % (kind, title, kind))
                    code, out = run(self.make(a + lemma('lem:b', proof_refs=['lem:a'])))
                    self.assertEqual(code, 1, out)
                    self.assertIn('edges: 2', out)
                    self.assertIn('UNEXPLAINED_COMPONENT_FOUND', out)
                    self.assertIn('(statement) -> S1.2 via lem:b:', out)

    def test_statement_title_equation_reference_uses_its_proof_owner(self):
        a = (r'\begin{lemma}[From \eqref{eq:b}]\label{lem:a}Statement.\end{lemma}'
             r'\begin{proof}Argument.\end{proof}')
        b = (r'\begin{lemma}\label{lem:b}Statement.\end{lemma}'
             r'\begin{proof}\begin{equation}\label{eq:b}1=1\end{equation}'
             r'Uses \ref{lem:a}.\end{proof}')
        code, out = run(self.make(a + b))
        self.assertEqual(code, 1, out)
        self.assertIn('edges: 2', out)
        self.assertIn('(statement) -> S1.2 via eq:b:', out)

    def test_acyclic_title_references_are_edges_not_automatic_failure(self):
        title = r'From \ref{lem:b}; again \ref{lem:b}; self \ref{lem:a}; external \ref{outside}'
        a = (r'\begin{lemma}[%s]\label{lem:a}Statement.\end{lemma}'
             r'\begin{proof}Argument.\end{proof}' % title)
        d = self.make(a + lemma('lem:b'))
        dot = os.path.join(d, 'title.dot')
        code, out = run(d, '--dot', dot)
        self.assertEqual(code, 0, out)
        self.assertIn('proofs: 2; edges: 1', out)
        with open(dot, encoding='utf-8') as f:
            edges = [line.strip() for line in f if ' -> ' in line]
        self.assertEqual(edges, ['"S1.1" -> "S1.2";'])

    def test_plain_and_commented_titles_preserve_acyclic_behavior(self):
        for title in ('A descriptive title', 'A title % ignored '+r'\ref{lem:b}'+'\ncontinued'):
            with self.subTest(title=title):
                a = (r'\begin{lemma}[%s]\label{lem:a}Statement.\end{lemma}'
                     r'\begin{proof}Argument.\end{proof}' % title)
                code, out = run(self.make(a + lemma('lem:b', proof_refs=['lem:a'])))
                self.assertEqual(code, 0, out)
                self.assertIn('proofs: 2; edges: 1', out)
                self.assertIn('NO_UNEXPLAINED_COMPONENT', out)



if __name__ == '__main__':
    unittest.main()
