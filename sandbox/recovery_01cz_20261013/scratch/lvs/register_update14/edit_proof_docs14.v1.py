#!/usr/bin/env python3
"""Round-14 hand edit of the proof packet's README.md and SCOPE.md for HEAD 9e305ea5: the round-6 pending forms that
edit_proof_docs13.py introduced (the 'no hosted receipt yet' receipt-paragraph sentence, the PENDING_NOTE 'no hosted receipt
yet, see above' of the two 'formerly listed here (round 6)' rows and of the triage totals sentence, and the SCOPE Side24
cell's 'have no hosted receipt yet ... their first receipt' clause) become their hosted receipt binding: run 38277458954 on
PR #441 head 9e305ea5, actual tested merge 1376175c... with base a0323b94... (Math- main at the merge of PR #440), receipt
theorem_count 78, covered_props 77, leanchecker PASS, artifact lean-specification-proofs-38277458954-1 sha256 1f32d60e...
(GitHub's digest; the artifact bytes were not downloaded). Derived from register_update13/edit_proof_docs13.py (whose new
strings are, verbatim, the old strings here) and register_update11/edit_proof_docs11.py (round 5's binding, committed as
32fc9845: its receipt paragraph is the shape; the receipt core below is asserted to reproduce, with the round-4 and the
round-5 facts, the committed round-4 and round-5 paragraphs up to line breaks, and only then instantiated with the round-6
facts). Exact string replacements over the COMMITTED text (read-only `git show 9e305ea5:...`), each with an asserted
occurrence count of one, refusing unexpected text and a worktree that differs from the commit.

THE TESTED MERGE IS NOT THE HEAD'S TREE for this run (rounds 4 and 5 said 'with the same tree as the head'): the synthetic
merge 1376175c differs from PR head 9e305ea5 only outside both packets (13 files under companions/torus_filtration_20261011/
and .github/workflows/torus-filtration.yml, main's changes from PRs #428 and #440); both packets are byte-identical between
head and merge (asserted by update_register14.py from local git objects). The receipt core's tree clause says that; the
SCOPE cell says it in short; no new 'same tree' is written (asserted).

Scientific effect: NONE. Documentation only: no label, disposition or status changes; claim-level labels stay none /
specified; alignment PENDING_INDEPENDENT_REVIEW; same-account author, organizational-independence credit 0. No triage verdict
is altered. Never touches check.py, tests, the register or Lean. A receipt is a record of a computation about the
registered Props for the formal objects only, never a review, an acceptance or an alignment, and never evidence about an
informal claim; a local kernel check is evidence about no commit. Every mention of a hosted run names the PR head and the
actual tested merge (C422-E1), asserted per paragraph. 'every control classified as intended (negative controls refused,
positive controls accepted)' is what check.py --execute requires before it writes a kernel-checked receipt, as for rounds
4 and 5; the FACTS record negative controls refused for their intended reasons, honest controls accepted and passed true.

usage: python3 -B -S edit_proof_docs14.py      (R14_PDOCS_OUT=<dir> writes the two docs there instead of the worktree)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

CLONE = Path('<SCRATCH>/repos/math_r5')
PACKET_REL = 'specifications/lean_statement_proofs_20261011'
PACKET = CLONE / PACKET_REL
OUT = Path(os.environ['R14_PDOCS_OUT']) if os.environ.get('R14_PDOCS_OUT') else PACKET
HEAD = '9e305ea5141a1d1503560efd0e3deaf25441749e'
# FACTS (coordinator-verified; re-read with read-only `gh api` and read-only git, see update_register14.py)
RUN11, C11, M11F, BASE5F = '38277458954', '9e305ea5', '1376175ce5536a6ac74cb03706d18e65fdb695fa', 'a0323b94ae54a78f3d2878dc9c358dc40cdfe1dc'
SHA11 = '1f32d60e08b0ff11b634c0413291686982f0b02e6809b9e96af2848350cbbf79'
M11, BASE5 = M11F[:8], BASE5F[:8]
RUN_STMT11, RUN_AGG11 = '38277458488', '38277459272'   # companion runs: not cited
# rounds 4 and 5, as bound in the committed docs
RUN10, C10, M10, BASE4, SHA10 = '38195810463', 'c6ce0256', '1320c5b2', 'be014794', 'd8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff'
RUN9, C9, M9, BASE3 = '38187538744', 'be014794', '79d3964c', '12e072d6'
RUN8, C8, M8, SHA8 = '38180705952', '48bfcbae', '9833d6d9', '87309efad9b804c53c543b60fdb466de1f797971417363cc2e8fa1e10eac26bd'
assert HEAD.startswith(C11)
MODULE = 'SpecificationProofs/Side24Images.lean'
BASE_DESC6 = 'Math- main at the merge of PR #440'


def git(*args):
    return subprocess.run(['git', '-C', str(CLONE), *args], check=True, capture_output=True, text=True, encoding='utf-8').stdout


def pristine(name: str) -> str:
    head = git('rev-parse', 'HEAD').strip()
    if head != HEAD:
        sys.exit(f'REFUSED: clone HEAD is {head}, expected {HEAD}')
    text = git('show', f'{HEAD}:{PACKET_REL}/{name}')
    if (PACKET / name).read_text(encoding='utf-8') != text:
        sys.exit(f'REFUSED: worktree {name} differs from {HEAD[:8]} (already edited?)')
    return text


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f'REFUSED: {label}: expected exactly one occurrence of the old text, found {n}')
    return text.replace(old, new)


norm = lambda s: ' '.join(s.split())
# the module this edit describes is committed and imported last by the root
assert (PACKET / MODULE).is_file() and git('ls-files', '--error-unmatch', f'{PACKET_REL}/{MODULE}').strip(), MODULE
_root = [line for line in (PACKET / 'SpecificationProofs.lean').read_text(encoding='utf-8').splitlines() if line.strip()]
assert _root[-1] == 'import SpecificationProofs.Side24Images' and len(_root) == 14, _root[-2:]
# the tested merge differs from the head only outside both packets (read-only; asserted in full by update_register14.py)
assert git('rev-list', '--parents', '-n', '1', M11F).split() == [M11F, BASE5F, HEAD]
assert git('diff', '--name-only', HEAD, M11F, '--', PACKET_REL, 'specifications/lean_statements_20261010').strip() == ''
assert git('rev-parse', f'{HEAD}^{{tree}}').strip() != git('rev-parse', f'{M11F}^{{tree}}').strip()


def receipt_core(subject, run, pr, head, merge, base, base_desc, tree, count, covered, sha):
    """The receipted-tier sentence of the README's receipt paragraph (whitespace-normalised), pinned to rounds 4 and 5 below."""
    return (f'{subject} are **receipted by the hosted `lean-specification-proofs.yml` run {run}** '
            f'(event pull_request on PR #{pr}, head `{head}`; the actually tested checkout, the receipt\'s `checked_commit`, is the synthetic merge '
            f'`{merge}…` of that head into base `{base}…`, {base_desc}, {tree}; job proof-evidence: fresh build, '
            f'`leanchecker` PASS, receipt `formalization_status: "kernel-checked"`, theorem_count {count}, covered_props {covered}, every control '
            f'classified as intended (negative controls refused, positive controls accepted), worktree CLEAN; artifact '
            f'`lean-specification-proofs-{run}-1`, sha256 `{sha[:8]}…` as GitHub reports it), their first receipt; the same run receipts all {count} '
            'theorems of that head.')


SAME_TREE = 'with the same tree as the head'
TREE6 = ('whose tree differs from the head\'s only outside both packets (13 files under `companions/torus_filtration_20261011/` and '
         '`.github/workflows/torus-filtration.yml`, main\'s changes from PRs #428 and #440; this packet and the statement packet are '
         'byte-identical between head and merge)')
R4_CORE = receipt_core('The 13 theorems of the three round-4 modules (`Side24Analysis`, `Side24Interfaces`, `RNCluster`)', RUN8, 434, C8, M8, BASE3,
                       'the head of PR #423', SAME_TREE, 73, 72, SHA8)
R5_CORE = receipt_core('The 2 theorems of the two round-5 modules (`Side24Covariance`, `Side24ConeMoment`)', RUN10, 439, C10, M10, BASE4,
                       'the head of PR #434', SAME_TREE, 75, 74, SHA10)
R6_CORE = receipt_core('The 3 theorems of the round-6 module `Side24Images` (`imageBound`, `entrywiseCovarianceBound`, `covarianceOrdering`)',
                       RUN11, 441, C11, M11, BASE5, BASE_DESC6, TREE6, 78, 77, SHA11)
assert 'same tree' not in R6_CORE and 'only outside both packets' in R6_CORE

# ---- README: edit_proof_docs13's new strings, verbatim (what 9e305ea5 carries and what is replaced)
OLD_RECEIPT = (
    'within `propext`, `Classical.choice`, `Quot.sound`, like any local run, are evidence about no commit. The 3 theorems\n'
    'of the round-6 module `Side24Images` (`imageBound`, `entrywiseCovarianceBound`, `covarianceOrdering`) have **no\n'
    f'hosted receipt yet**: they were added after PR head `{C10}`, whose hosted run {RUN10} (actual tested merge\n'
    f'`{M10}…` with base `{BASE4}…`; receipt theorem_count 75) does not name them; their local checks at the pin\n'
    '(`lake build`, `lake env lean -DwarningAsError=true`, `#check` of each against its registered Prop, `#print axioms`\n'
    'within `propext`, `Classical.choice`, `Quot.sound`), on an unverified worktree, like any local run, are evidence\n'
    'about no commit; the next hosted `lean-specification-proofs.yml` run is their first receipt.\n\n| group |')
NEW_RECEIPT_CORE = (
    'The 3 theorems\n'
    'of the round-6 module `Side24Images` (`imageBound`, `entrywiseCovarianceBound`, `covarianceOrdering`) are **receipted\n'
    f'by the hosted `lean-specification-proofs.yml` run {RUN11}** (event pull_request on PR #441, head `{C11}`; the\n'
    'actually tested checkout, the receipt\'s `checked_commit`, is the synthetic merge `' + M11 + '…` of that head into base\n'
    f'`{BASE5}…`, {BASE_DESC6}, whose tree differs from the head\'s only outside both packets (13\n'
    'files under `companions/torus_filtration_20261011/` and `.github/workflows/torus-filtration.yml`, main\'s changes from\n'
    'PRs #428 and #440; this packet and the statement packet are byte-identical between head and merge); job proof-evidence:\n'
    'fresh build, `leanchecker` PASS, receipt `formalization_status: "kernel-checked"`, theorem_count 78, covered_props 77,\n'
    'every control classified as intended (negative controls refused, positive controls accepted), worktree CLEAN; artifact\n'
    f'`lean-specification-proofs-{RUN11}-1`, sha256 `{SHA11[:8]}…` as GitHub reports it), their first receipt; the same run\n'
    'receipts all 78 theorems of that head.')
assert norm(NEW_RECEIPT_CORE) == R6_CORE
NEW_RECEIPT = (
    'within `propext`, `Classical.choice`, `Quot.sound`, like any local run, are evidence about no commit. ' + NEW_RECEIPT_CORE
    + f' They were added after PR head `{C10}`, whose hosted run {RUN10} (actual\n'
    f'tested merge `{M10}…` with base `{BASE4}…`; receipt theorem_count 75) does not name them; their local checks at the\n'
    'pin (`lake build`, `lake env lean -DwarningAsError=true`, `#check` of each against its registered Prop, `#print axioms`\n'
    'within `propext`, `Classical.choice`, `Quot.sound`), like any local run, are evidence about no commit.\n\n| group |')
PENDING_NOTE = 'no hosted receipt yet, see above'
RECEIPT6_NOTE = f'first receipted by run {RUN11}: PR head `{C11}`, actual tested merge `{M11}…` with base `{BASE5}…`, see above'
RECEIPT5_NOTE = f'first receipted by run {RUN10}: PR head `{C10}`, actual tested merge `{M10}…` with base `{BASE4}…`, see above'
assert RECEIPT6_NOTE == RECEIPT5_NOTE.replace(RUN10, RUN11).replace(C10, C11).replace(M10, M11).replace(BASE4, BASE5)   # round 5's note shape
README_EDITS = [
    ('round-6 receipt status', OLD_RECEIPT, NEW_RECEIPT),
    ('lattice round-6 row: pending note',
     f'(`Side24Images.imageBound`; {PENDING_NOTE})',
     f'(`Side24Images.imageBound`; {RECEIPT6_NOTE})'),
    ('covariance round-6 row: pending note',
     f'(`Side24Images.entrywiseCovarianceBound`, `Side24Images.covarianceOrdering`; {PENDING_NOTE})',
     f'(`Side24Images.entrywiseCovarianceBound`, `Side24Images.covarianceOrdering`; {RECEIPT6_NOTE})'),
    ('triage totals: pending note',
     f'have theorems since round 6 ({PENDING_NOTE}).',
     f'have theorems since round 6 ({RECEIPT6_NOTE}).'),
]
# ---- SCOPE
SCOPE_EDITS = [
    ('Side24 family: round-6 receipt status',
     '; the image bound, the entrywise bound and display (3) (round 6, module Side24Images) have no hosted receipt yet: they were added after PR head '
     f'`{C10}`, that run does not name them, and the next hosted `lean-specification-proofs.yml` run is their first receipt |',
     '; the image bound, the entrywise bound and display (3) (round 6, module Side24Images) are first receipted by run '
     f'{RUN11} (PR head `{C11}`, actual tested merge `{M11}…` with base `{BASE5}…`, the merge differing from the head only outside both packets; '
     'receipt theorem_count 78) |'),
]


def unescaped_pipes(line: str) -> int:
    return len(re.findall(r'(?<!\\)\|', line))


def table_rows(text: str, header: str) -> list:
    lines = text.split('\n')
    i = lines.index(header)
    rows = []
    for line in lines[i + 2:]:
        if not line.startswith('|'):
            break
        rows.append(line)
    return rows


README_HEADER = '| group | Props (names without `UniversalLaw.Spec.<Module>.`) | why no theorem here |'
SCOPE_HEADER = '| family | the theorems establish | not established |'
fr = lambda t: len(re.findall(r'first receipt(?!ed)', norm(t)))


def main() -> None:
    for name, edits in (('README.md', README_EDITS), ('SCOPE.md', SCOPE_EDITS)):
        text = pristine(name)
        orig = text
        assert 'no hosted receipt yet' in orig and RUN10 in orig and RUN11 not in orig and '†' not in orig, name
        if name == 'README.md':      # the receipt core reproduces the committed round-4 and round-5 paragraphs (the shape is pinned, not retyped)
            assert norm(orig).count(R4_CORE) == 1, 'the round-4 receipt paragraph is not the committed shape'
            assert norm(orig).count(R5_CORE) == 1, 'the round-5 receipt paragraph is not the committed shape'
            assert orig.count(PENDING_NOTE) == 3 and orig.count(RECEIPT5_NOTE) == 3
        for label, old, new in edits:
            text = replace_once(text, old, new, f'{name}: {label}')
        # the pending forms are gone
        assert '†' not in text and 'no hosted receipt' not in text and 'the last head with a hosted receipt' not in text, name
        assert 'is their first receipt' not in text and 'unverified worktree' not in text and 'await the' not in text, name
        assert text.count('kernel-checked') == orig.count('kernel-checked') + (1 if name == 'README.md' else 0), name   # one receipt word, in the receipt
        assert len(re.findall(r'\bproved\b', text)) == len(re.findall(r'\bproved\b', orig)), name                         # the label is not added
        assert text.count('same tree') == orig.count('same tree'), name                                                  # no new same-tree claim
        for para in text.split('\n\n'):      # C422-E1: a hosted run is named with its PR head and its tested merge
            if RUN11 in para:
                assert C11 in para and M11 in para and BASE5 in para, (name, para[:120])
                if RUN11 in para.replace(RECEIPT6_NOTE, ''):     # the notes point 'see above' to the receipt paragraph, round 5's note shape
                    assert 'only outside both packets' in para, (name, para[:120])
            if RUN10 in para:
                assert C10 in para and M10 in para and BASE4 in para, (name, para[:120])
            if RUN9 in para:
                assert C9 in para and M9 in para and BASE3 in para, (name, para[:120])
            if RUN8 in para:
                assert C8 in para and M8 in para, (name, para[:120])
        for found in set(re.findall(r'\b38\d{9}\b', text)) - set(re.findall(r'\b38\d{9}\b', orig)):
            assert found == RUN11, (name, found)
        assert set(re.findall(r'\b38\d{9}\b', orig)) <= set(re.findall(r'\b38\d{9}\b', text)), name   # no earlier run dropped
        assert RUN_STMT11 not in text and RUN_AGG11 not in text, name
        for phrase in ('Scientific effect', 'NONE', 'PENDING_INDEPENDENT_REVIEW'):
            assert phrase in text, (name, phrase)
        assert not re.findall(r'<<[^<>\n]*>>', text), name                                         # check.py's PLACEHOLDER
        assert text.count(RUN10) == orig.count(RUN10), name                                        # round 5's receipt stays bound everywhere
        if name == 'README.md':
            assert norm(text).count(R6_CORE) == 1 and norm(text).count(R5_CORE) == 1 and norm(text).count(R4_CORE) == 1
            assert text.count(RECEIPT6_NOTE) == 3 and text.count(RECEIPT5_NOTE) == 3 and text.count(PENDING_NOTE) == 0
            assert text.count('formerly listed here (round 6)') == 2 and text.count('formerly listed here (round 5)') == 2
            assert fr(text) == fr(orig) == 4   # round 4's, round 5's, round 6's ('their first receipt'), the generic 'where the first receipt for a commit exists'
            assert text.count(RUN11) == 2 + 3   # the receipt core (run and artifact name) and the three notes
            rows, rows_o = table_rows(text, README_HEADER), table_rows(orig, README_HEADER)
            assert len(rows) == len(rows_o) and all(unescaped_pipes(r) == 4 for r in rows)
            assert [r.split(' | ')[0] for r in rows] == [r.split(' | ')[0] for r in rows_o]
            assert 'counts for the 64 Props without a theorem are 28 predicates' in text
        else:
            rows, rows_o = table_rows(text, SCOPE_HEADER), table_rows(orig, SCOPE_HEADER)
            assert len(rows) == len(rows_o) == 4 and all(unescaped_pipes(r) == 4 for r in rows)
            side24 = [r for r in rows if r.startswith('| Side24 ')]
            assert len(side24) == 1 and RUN11 in side24[0].split(' | ')[1] and text.count(RUN11) == 1 and text.count(RUN10) == 1
            assert 'nothing here encloses the coefficient' in side24[0].split(' | ')[2]
        (OUT / name).write_text(text, encoding='utf-8')
        print(f'{name}: {len(edits)} replacement(s) applied -> {OUT / name}')


if __name__ == '__main__':
    main()
