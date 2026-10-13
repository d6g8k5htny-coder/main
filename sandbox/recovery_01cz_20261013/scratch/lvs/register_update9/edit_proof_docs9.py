#!/usr/bin/env python3
"""Round-9 hand edit of the proof packet's README.md (and SCOPE.md, which needs none) for HEAD 48bfcbae: the
'no hosted receipt at this commit' / 'first receipt' statement about the 13 round-4 theorems becomes their hosted
receipt binding (run 38180705952 on PR #434 head 48bfcbae, actual tested merge 9833d6d9... with base 12e072d6...,
receipt theorem_count 73, leanchecker PASS, artifact sha256 87309efa...). Derived from register_update7/pdocs/
edit_proof_docs.py: exact string replacements over the COMMITTED text (read-only `git show`), refusing unexpected text.

Scientific effect: NONE. Documentation only: no label, disposition or status changes; alignment
PENDING_INDEPENDENT_REVIEW; organizational-independence credit 0. Never touches check.py, tests or Lean. A receipt is
evidence about the registered Props for the formal objects only, never about an informal claim.
"""
import subprocess
import sys
from pathlib import Path

CLONE = Path('<SCRATCH>/repos/math')
PACKET_REL = 'specifications/lean_statement_proofs_20261011'
PACKET = CLONE / PACKET_REL
HEAD = '48bfcbae7834c2d6f4d3123d85c69d0c5afedfa6'
# FACTS (coordinator-verified)
RUN8, C8, M8F, BASE3F = '38180705952', '48bfcbae', '9833d6d9f9facc335e1a56e243c3144adefe6225', '12e072d614dbe6e91b66d22a2ab55cf87860d693'
SHA8 = '87309efad9b804c53c543b60fdb466de1f797971417363cc2e8fa1e10eac26bd'
M8, BASE3 = M8F[:8], BASE3F[:8]
assert HEAD.startswith(C8)


def pristine(name: str) -> str:
    """The committed text at HEAD 48bfcbae (read-only `git show`; no checkout)."""
    head = subprocess.run(['git', '-C', str(CLONE), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    if head != HEAD:
        sys.exit(f'REFUSED: clone HEAD is {head}, expected {HEAD}')
    return subprocess.run(['git', '-C', str(CLONE), 'show', f'{HEAD}:{PACKET_REL}/{name}'], check=True,
                          capture_output=True, text=True, encoding='utf-8').stdout


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f'REFUSED: {label}: expected exactly one occurrence of the old text, found {n}')
    return text.replace(old, new)


README_EDITS = [
    (
        'round-4 receipt status',
        "so. The 13 theorems of the three round-4 modules (`Side24Analysis`, `Side24Interfaces`, `RNCluster`) have **no\n"
        "hosted receipt at this commit**: `check.py --allow-dirty` passes in source mode and the round-4 commit records a\n"
        "clean local `lake build` with every theorem of the new modules on `propext`, `Classical.choice`, `Quot.sound`\n"
        "only, but a local run is not evidence about any commit; the first `lean-specification-proofs.yml` run on this\n"
        "branch is their first receipt.\n",
        "so. The 13 theorems of the three round-4 modules (`Side24Analysis`, `Side24Interfaces`, `RNCluster`) are **receipted\n"
        f"by the hosted `lean-specification-proofs.yml` run {RUN8}** (event pull_request on PR #434, head `{C8}`; the actually\n"
        f"tested checkout, the receipt's `checked_commit`, is the synthetic merge `{M8}…` of that head into base `{BASE3}…`, the\n"
        "head of PR #423, with the same tree as the head; job proof-evidence: fresh build, `leanchecker` PASS, receipt\n"
        "`formalization_status: \"kernel-checked\"`, theorem_count 73, covered_props 72, every negative control refused for its\n"
        f"intended reason, worktree CLEAN; artifact `lean-specification-proofs-{RUN8}-1`, sha256 `{SHA8[:8]}…` as GitHub\n"
        "reports it), their first receipt; the same run receipts all 73 theorems of that head. The round-4 commit's clean local\n"
        "`lake build` with every theorem of the new modules on `propext`, `Classical.choice`, `Quot.sound` only, like any local\n"
        "run, is evidence about no commit; later commits need their own run.\n",
    ),
]
SCOPE_EDITS = []   # SCOPE.md states no receipt status for the round-4 theorems: nothing to bind, nothing is changed


def main() -> None:
    for name, edits in (('README.md', README_EDITS), ('SCOPE.md', SCOPE_EDITS)):
        path = PACKET / name
        text = pristine(name)
        for phrase in ('no hosted receipt', 'none yet', 'await the next hosted', '†'):
            assert phrase not in text or name == 'README.md' and phrase == 'no hosted receipt', (name, phrase)
        for label, old, new in edits:
            text = replace_once(text, old, new, f'{name}: {label}')
        if name == 'README.md':
            assert 'no hosted receipt' not in text and 'is their first receipt' not in text and 'their first receipt' in text
            for para in text.split('\n\n'):      # C422-E1: the run is named with its PR head and its tested merge
                if RUN8 in para:
                    assert C8 in para and M8 in para and BASE3 in para, para[:120]
            # the one pre-existing 'first receipt' sentence is generic ("where the first receipt for a commit exists"), not a status
            assert text.count('first receipt') == 2, text.count('first receipt')
        if edits:
            path.write_text(text, encoding='utf-8')
        else:
            assert path.read_text(encoding='utf-8') == text, f'{name}: worktree differs from HEAD although no edit is planned'
        print(f'{name}: {len(edits)} replacement(s) applied')


if __name__ == '__main__':
    main()
