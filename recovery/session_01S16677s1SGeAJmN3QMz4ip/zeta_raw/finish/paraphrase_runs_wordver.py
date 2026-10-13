"""Throwaway (not part of the packet): maximal shared word runs of length >= N between the
private note and README.md / CLAIMS.md (plus .py docstrings and the PR/commit drafts).

Usage: python3 -I paraphrase_runs_wordver.py NOTE PACKET_DIR FINISH_DIR [N]
Prints each maximal shared run once per target file, with its length. Two tokenizations:
(a) unicode word tokens (\\w+), lowercased; (b) the same after folding unicode super/subscripts
and math letters to ASCII-ish forms, so that 'r²' and 'r^2' both become 'r 2'.
"""
import re
import sys
import unicodedata
from pathlib import Path

note_path, pkt, fin = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 6


def toks(text, fold):
    if fold:
        text = unicodedata.normalize("NFKC", text)
    return re.findall(r"\w+", text.lower())


def runs(a, b, n):
    """Maximal common runs (length >= n) of token lists a, b, via n-gram seeding."""
    idx = {}
    for i in range(len(b) - n + 1):
        idx.setdefault(tuple(b[i:i + n]), []).append(i)
    out = set()
    i = 0
    covered = set()
    for i in range(len(a) - n + 1):
        key = tuple(a[i:i + n])
        for j in idx.get(key, ()):
            if (i - 1, j - 1) in covered:
                continue
            # extend backward/forward
            s_i, s_j = i, j
            while s_i > 0 and s_j > 0 and a[s_i - 1] == b[s_j - 1]:
                s_i -= 1
                s_j -= 1
            e_i, e_j = i + n, j + n
            while e_i < len(a) and e_j < len(b) and a[e_i] == b[e_j]:
                e_i += 1
                e_j += 1
            for k in range(e_i - s_i):
                covered.add((s_i + k, s_j + k))
            out.add(" ".join(a[s_i:e_i]))
    return sorted(out, key=lambda r: -len(r.split()))


targets = [pkt / "README.md", pkt / "CLAIMS.md"]
targets += sorted(pkt.rglob("*.py"))
targets += [fin / "pr_body.md", fin / "pr_body_live.md", fin / "commit_msg.txt"]
note = note_path.read_text(encoding="utf-8")
for fold in (False, True):
    print(f"=== tokenization {'NFKC-folded' if fold else 'plain'}; N = {N}")
    a = toks(note, fold)
    for t in targets:
        b = toks(t.read_text(encoding="utf-8"), fold)
        found = runs(a, b, N)
        if found:
            print(f"--- {t.relative_to(t.parents[1]) if t.is_relative_to(pkt) else t.name}: {len(found)} run(s)")
            for r in found:
                print(f"  [{len(r.split())}] {r}")
