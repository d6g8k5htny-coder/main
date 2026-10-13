#!/usr/bin/env python3
"""Round-11 hand edit of the proof packet's README.md and SCOPE.md for HEAD c6ce0256: the round-5 pending forms that
edit_proof_docs10.py introduced (the 'no hosted receipt yet' receipt paragraph, the dagger † on the two round-5 theorems
and its legends) become their hosted receipt binding: run 38195810463 on PR #439 head c6ce0256, actual tested merge
1320c5b2... with base be014794... (the head of PR #434), receipt theorem_count 75, covered_props 74, leanchecker PASS,
artifact lean-specification-proofs-38195810463-1 sha256 d8abd42b... (GitHub's digest). Derived from
register_update10/edit_proof_docs10.py (whose new strings are, verbatim, the old strings here) and
register_update9/edit_proof_docs9.py (the round-4 binding, whose receipt paragraph, as amended by r4amend/apply_r4_fix2.py
and committed, is the shape: the receipt core below is asserted to reproduce, with the round-4 facts, the committed round-4
paragraph up to line breaks). Exact string replacements over the COMMITTED text (read-only `git show c6ce0256:...`),
refusing unexpected text and a worktree that differs from the commit.

Scientific effect: NONE. Documentation only: no label, disposition or status changes; alignment
PENDING_INDEPENDENT_REVIEW; organizational-independence credit 0. Never touches check.py, tests or Lean. A receipt is
evidence about the registered Props for the formal objects only, never about an informal claim; a local kernel check on an
unverified worktree is evidence about no commit. Every mention of a hosted run names the PR head and the actual tested
merge (C422-E1), asserted per paragraph. 'every control classified as intended (negative controls refused, positive
controls accepted)' is what check.py --execute requires before it writes a kernel-checked receipt (negative_controls:
the hidden environment control refused for exactly the expected reasons, the honest control classified exactly, the
justified maxHeartbeats control accepted), as for the round-4 run; the FACTS record negative controls refused for their
intended reasons and passed true.

usage: python3 -B -S edit_proof_docs11.py      (R11_PDOCS_OUT=<dir> writes the two docs there instead of the worktree)
"""
import os
import re
import subprocess
import sys
from pathlib import Path

CLONE = Path('<SCRATCH>/repos/math_r5')
PACKET_REL = 'specifications/lean_statement_proofs_20261011'
PACKET = CLONE / PACKET_REL
OUT = Path(os.environ['R11_PDOCS_OUT']) if os.environ.get('R11_PDOCS_OUT') else PACKET
HEAD = 'c6ce025652f76be6e580fa24947380253450cd3b'
# FACTS (coordinator-verified; re-read with read-only `gh api`)
RUN10, C10, M10F, BASE4F = '38195810463', 'c6ce0256', '1320c5b217e491d301ea140e5cc25b07ff5a21b7', 'be01479435f98f34b802258ee9ec3c3578f37169'
SHA10 = 'd8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff'
M10, BASE4 = M10F[:8], BASE4F[:8]
RUN9, C9, M9, BASE3, SHA9 = '38187538744', 'be014794', '79d3964c', '12e072d6', '60c9be0b49d45931d6c82f04d73fb7e6b6e97d539e7c10a376c3e99e302f2617'
RUN8, C8, M8, SHA8 = '38180705952', '48bfcbae', '9833d6d9', '87309efad9b804c53c543b60fdb466de1f797971417363cc2e8fa1e10eac26bd'
RUN_STMT10, RUN_AGG10 = '38195810435', '38195810649'   # companion runs: not cited
assert HEAD.startswith(C10) and BASE4F.startswith(C9)
NEW_MODULES = ('SpecificationProofs/Side24Covariance.lean', 'SpecificationProofs/Side24ConeMoment.lean')


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
for m in NEW_MODULES:    # the modules this edit describes are committed and imported by the root
    assert (PACKET / m).is_file() and git('ls-files', '--error-unmatch', f'{PACKET_REL}/{m}').strip(), m
    assert ('import ' + m[:-5].replace('/', '.')) in (PACKET / 'SpecificationProofs.lean').read_text(encoding='utf-8'), m


def receipt_core(n, k, r, mods, run, pr, head, merge, base, base_pr, count, covered, sha):
    """The receipted-tier sentence of the README's receipt paragraph (whitespace-normalised), pinned to round 4 below."""
    return (f'The {n} theorems of the {k} round-{r} modules ({mods}) are **receipted by the hosted `lean-specification-proofs.yml` run {run}** '
            f'(event pull_request on PR #{pr}, head `{head}`; the actually tested checkout, the receipt\'s `checked_commit`, is the synthetic merge '
            f'`{merge}…` of that head into base `{base}…`, the head of PR #{base_pr}, with the same tree as the head; job proof-evidence: fresh build, '
            f'`leanchecker` PASS, receipt `formalization_status: "kernel-checked"`, theorem_count {count}, covered_props {covered}, every control '
            f'classified as intended (negative controls refused, positive controls accepted), worktree CLEAN; artifact '
            f'`lean-specification-proofs-{run}-1`, sha256 `{sha[:8]}…` as GitHub reports it), their first receipt; the same run receipts all {count} '
            'theorems of that head.')


R4_CORE = receipt_core(13, 'three', 4, '`Side24Analysis`, `Side24Interfaces`, `RNCluster`', RUN8, 434, C8, M8, BASE3, 423, 73, 72, SHA8)
R5_CORE = receipt_core(2, 'two', 5, '`Side24Covariance`, `Side24ConeMoment`', RUN10, 439, C10, M10, BASE4, 434, 75, 74, SHA10)

# edit_proof_docs10's new strings, verbatim (they are what c6ce0256 carries and what is replaced)
OLD_RECEIPT = (
    'run, is evidence about no commit; later commits need their own run. The 2 theorems of the two round-5 modules\n'
    '(`Side24Covariance`, `Side24ConeMoment`), marked † below, have **no hosted receipt yet**: they were added after PR head\n'
    f'`{C9}`, the last head with a hosted receipt (run {RUN9}, event pull_request on PR #434; the actually tested\n'
    f'checkout is the synthetic merge `{M9}…` of that head into base `{BASE3}…`; receipt theorem_count 73, `leanchecker`\n'
    f'PASS; artifact `lean-specification-proofs-{RUN9}-1`, sha256 `{SHA9[:8]}…` as GitHub reports it), which does not name\n'
    'them. Locally, `lake env lean -DwarningAsError=true` exited 0 on each module at the pin and `#print axioms` stayed\n'
    'within `propext`, `Classical.choice`, `Quot.sound`, on an unverified worktree, which is evidence about no commit; the\n'
    'next hosted `lean-specification-proofs.yml` run is their first receipt.\n\n| group |')
NEW_RECEIPT_CORE = (
    'The 2 theorems of the two round-5 modules\n'
    f'(`Side24Covariance`, `Side24ConeMoment`) are **receipted by the hosted `lean-specification-proofs.yml` run {RUN10}**\n'
    f'(event pull_request on PR #439, head `{C10}`; the actually tested checkout, the receipt\'s `checked_commit`, is the\n'
    f'synthetic merge `{M10}…` of that head into base `{BASE4}…`, the head of PR #434, with the same tree as the head; job\n'
    'proof-evidence: fresh build, `leanchecker` PASS, receipt `formalization_status: "kernel-checked"`, theorem_count 75,\n'
    'covered_props 74, every control classified as intended (negative controls refused, positive controls accepted),\n'
    f'worktree CLEAN; artifact `lean-specification-proofs-{RUN10}-1`, sha256 `{SHA10[:8]}…` as GitHub reports it), their\n'
    'first receipt; the same run receipts all 75 theorems of that head.')
assert norm(NEW_RECEIPT_CORE) == R5_CORE
NEW_RECEIPT = (
    'run, is evidence about no commit; later commits need their own run. ' + NEW_RECEIPT_CORE + f' They were added after PR head `{C9}`,\n'
    f'whose hosted run {RUN9} (actual tested merge `{M9}…` with base `{BASE3}…`; receipt theorem_count 73)\n'
    'does not name them; their local `lake env lean -DwarningAsError=true` checks at the pin, with `#print axioms`\n'
    'within `propext`, `Classical.choice`, `Quot.sound`, like any local run, are evidence about no commit.\n\n| group |')
RECEIPT_NOTE = f'first receipted by run {RUN10}: PR head `{C10}`, actual tested merge `{M10}…` with base `{BASE4}…`, see above'
README_EDITS = [
    ('round-5 receipt status', OLD_RECEIPT, NEW_RECEIPT),
    ('covariance round-5 row: dagger',
     '(`Side24Covariance.referenceCovarianceLowerBound†`; † = no hosted receipt yet, see above)',
     f'(`Side24Covariance.referenceCovarianceLowerBound`; {RECEIPT_NOTE})'),
    ('interface round-5 row: dagger',
     '(`Side24ConeMoment.coneMomentD2Identity†`; † = no hosted receipt yet, see above)',
     f'(`Side24ConeMoment.coneMomentD2Identity`; {RECEIPT_NOTE})'),
    ('triage totals: dagger',
     'have theorems since round 5 (marked †, no hosted receipt yet).',
     f'have theorems since round 5 ({RECEIPT_NOTE}).'),
]
SCOPE_EDITS = [
    ('Side24 family: reference covariance bound dagger',
     'stated for every `0 < d`)†; and, for a GIVEN `I : ConeMomentInterface`,',
     'stated for every `0 < d`); and, for a GIVEN `I : ConeMomentInterface`,'),
    ('Side24 family: D2 cone moment dagger',
     '`∫ coneIntegrand (s, x, y) ∂I.P = (1/2)(25/3 - 20/3 + 8 - 8 √(3/8))`† from the law field `sxy_law` alone',
     '`∫ coneIntegrand (s, x, y) ∂I.P = (1/2)(25/3 - 20/3 + 8 - 8 √(3/8))` from the law field `sxy_law` alone'),
    ('Side24 family: dagger legend -> receipt binding',
     f'the interface structures); † = no hosted receipt yet (added after PR head `{C9}`, the last head with a hosted receipt: run {RUN9}, '
     f'actual tested merge `{M9}…` with base `{BASE3}…`) |',
     'the interface structures); the reference covariance lower bound and the `m = 2` cone moment (round 5) are first receipted by run '
     f'{RUN10} (PR head `{C10}`, actual tested merge `{M10}…` with base `{BASE4}…`; receipt theorem_count 75) |'),
]


def main() -> None:
    for name, edits in (('README.md', README_EDITS), ('SCOPE.md', SCOPE_EDITS)):
        text = pristine(name)
        orig = text
        assert '†' in orig and 'no hosted receipt yet' in orig and RUN9 in orig and RUN10 not in orig, name
        if name == 'README.md':      # the receipt core reproduces the committed round-4 paragraph (the shape is pinned, not retyped)
            assert norm(orig).count(R4_CORE) == 1, 'the round-4 receipt paragraph is not the committed shape'
        for label, old, new in edits:
            text = replace_once(text, old, new, f'{name}: {label}')
        # the pending forms are gone
        assert '†' not in text and 'no hosted receipt' not in text and 'the last head with a hosted receipt' not in text, name
        assert 'is their first receipt' not in text and 'unverified worktree' not in text, name
        assert text.count('kernel-checked') == orig.count('kernel-checked') + (1 if name == 'README.md' else 0), name   # one receipt word, in the receipt
        for para in text.split('\n\n'):      # C422-E1: a hosted run is named with its PR head and its tested merge
            if RUN10 in para:
                assert C10 in para and M10 in para and BASE4 in para, (name, para[:120])
            if RUN9 in para:
                assert C9 in para and M9 in para and BASE3 in para, (name, para[:120])
            if RUN8 in para:
                assert C8 in para and M8 in para, (name, para[:120])
        for found in set(re.findall(r'\b38\d{9}\b', text)) - set(re.findall(r'\b38\d{9}\b', orig)):
            assert found == RUN10, (name, found)
        assert RUN_STMT10 not in text and RUN_AGG10 not in text, name
        for phrase in ('Scientific effect', 'NONE', 'PENDING_INDEPENDENT_REVIEW'):
            assert phrase in text, (name, phrase)
        if name == 'README.md':
            assert norm(text).count(R5_CORE) == 1 and norm(text).count(R4_CORE) == 1
            assert text.count(RECEIPT_NOTE) == 3 and text.count('formerly listed here (round 5)') == 2
            fr = lambda t: len(re.findall(r'first receipt(?!ed)', norm(t)))
            assert fr(text) == fr(orig) == 3   # round 4's, round 5's, the generic 'where the first receipt for a commit exists'
            assert text.count(RUN9) == 1 and orig.count(RUN9) == 2      # the artifact name of run 38187538744 is no longer cited; the run is
            for line in text.split('\n'):    # the "why no theorem here" rows list neither round-5 Prop
                if line.startswith('| Side24 covariance and density |') or line.startswith('| Side24 interface statements |'):
                    assert 'ReferenceCovarianceLowerBound' not in line and 'ConeMomentD2Identity' not in line, line[:80]
        else:
            assert text.count(RUN10) == 1 and RUN9 not in text
        (OUT / name).write_text(text, encoding='utf-8')
        print(f'{name}: {len(edits)} replacement(s) applied -> {OUT / name}')


if __name__ == '__main__':
    main()
