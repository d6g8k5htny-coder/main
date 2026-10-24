"""Throwaway regression scan (not part of the packet): maximal shared word runs of length >= N
between the private note and README.md / CLAIMS.md after the slice-B edits.

Usage: python3 -I paraphrase_regress_slB.py NOTE PACKET_DIR [N]
Two tokenizations: plain \\w+ lowercased, and NFKC-folded (r² -> r2). Each maximal run is
classified 'formula' when fewer than half of its tokens are prose words (alphabetic, length
>= 3, not a lone math symbol name); only non-formula runs are flagged as problems.
"""
import re
import sys
import unicodedata
from pathlib import Path

note_path, pkt = Path(sys.argv[1]), Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 6
MATHWORDS = {"sin", "cos", "log", "exp", "mod", "det", "max", "min", "sup", "inf", "lim"}


def toks(text, fold):
    if fold:
        text = unicodedata.normalize("NFKC", text)
    return re.findall(r"\w+", text.lower())


def runs(a, b, n):
    idx = {}
    for j in range(len(b) - n + 1):
        idx.setdefault(tuple(b[j:j + n]), []).append(j)
    seen, out = set(), {}
    for i in range(len(a) - n + 1):
        for j in idx.get(tuple(a[i:i + n]), ()):
            if (i, j) in seen:
                continue
            si, sj = i, j
            while si > 0 and sj > 0 and a[si - 1] == b[sj - 1]:
                si, sj = si - 1, sj - 1
            ei, ej = i + n, j + n
            while ei < len(a) and ej < len(b) and a[ei] == b[ej]:
                ei, ej = ei + 1, ej + 1
            for k in range(ei - si):
                seen.add((si + k, sj + k))
            run = tuple(a[si:ei])
            out[run] = out.get(run, 0) + 1
    return out


def is_formula(run):
    prose = [w for w in run if w.isalpha() and len(w) >= 3 and w not in MATHWORDS]
    return len(prose) * 2 < len(run)


note = note_path.read_text(encoding="utf-8")
problems = 0
for fold in (False, True):
    a = toks(note, fold)
    for t in (pkt / "README.md", pkt / "CLAIMS.md"):
        b = toks(t.read_text(encoding="utf-8"), fold)
        found = runs(a, b, N)
        prose = {r: c for r, c in found.items() if not is_formula(r)}
        form = {r: c for r, c in found.items() if is_formula(r)}
        print(f"[{'nfkc' if fold else 'plain'}] {t.name}: {len(prose)} non-formula run(s), "
              f"{len(form)} formula run(s) of >= {N} words")
        for r, c in sorted(prose.items(), key=lambda x: -len(x[0])):
            problems += 1
            print(f"   PROSE len={len(r)} x{c}: {' '.join(r)}")
        for r, c in sorted(form.items(), key=lambda x: -len(x[0])):
            print(f"   formula len={len(r)} x{c}: {' '.join(r)}")
print("non-formula runs total:", problems)
