"""Throwaway paraphrase check for ZF-AD-02 (not part of the packet; never writes files).

Lists left- and right-maximal shared word runs of length >= MIN_RUN (default 6) between the
private note and each packet text file given. Output: file, run length, packet line(s), and
the run as it appears in the PACKET text (lower-cased tokens). Two tokenizations are run:
'plain' (words only) and 'sym' (Greek letters and math symbols mapped to words), so that
runs broken only by symbols are also caught.

Usage: python3 -I paraphrase_check_ad02.py NOTE FILE [FILE ...] [--min N]
"""
import re
import sys
from collections import defaultdict

argv = sys.argv[1:]
MIN_RUN = 6
if "--min" in argv:
    i = argv.index("--min")
    MIN_RUN = int(argv[i + 1])
    del argv[i:i + 2]
NOTE, FILES = argv[0], argv[1:]

SYM = {
    "ξ": " xi ", "Ξ": " xi ", "ζ": " zeta ", "τ": " tau ", "Λ": " lambda ", "λ": " lambda ",
    "ℓ": " ell ", "θ": " theta ", "Θ": " theta ", "ω": " omega ", "π": " pi ", "Φ": " phi ",
    "φ": " phi ", "Γ": " gamma ", "γ": " gamma ", "ρ": " rho ", "κ": " kappa ", "σ": " sigma ",
    "α": " alpha ", "β": " beta ", "∝": " propto ", "≈": " approx ", "∼": " sim ", "~": " sim ",
    "±": " pm ", "√": " sqrt ", "ℤ": " z ", "≥": " ge ", "≤": " le ", "↔": " ", "→": " to ",
    "−": " minus ", "–": " ", "—": " ", "²": " 2 ", "³": " 3 ", "⁴": " 4 ", "₀": " 0 ",
    "∫": " int ", "∂": " d ", "≠": " ne ",
}
WORD = re.compile(r"[^\W_]+", re.UNICODE)


def toks(text, sym):
    out = []
    for ln, line in enumerate(text.splitlines(), 1):
        if sym:
            for k, v in SYM.items():
                line = line.replace(k, v)
        out.extend((w.lower(), ln) for w in WORD.findall(line))
    return out


def maximal_runs(note_t, file_t):
    note = [w for w, _ in note_t]
    ft = [w for w, _ in file_t]
    index = defaultdict(list)
    for j in range(len(note) - MIN_RUN + 1):
        index[tuple(note[j:j + MIN_RUN])].append(j)
    found = {}
    for i in range(len(ft) - MIN_RUN + 1):
        for j in index.get(tuple(ft[i:i + MIN_RUN]), ()):
            if i > 0 and j > 0 and ft[i - 1] == note[j - 1]:
                continue
            k = MIN_RUN
            while i + k < len(ft) and j + k < len(note) and ft[i + k] == note[j + k]:
                k += 1
            phrase = " ".join(ft[i:i + k])
            rec = found.setdefault(phrase, [k, set()])
            rec[1].add(file_t[i][1])
    return found


with open(NOTE, encoding="utf-8") as fh:
    note_text = fh.read()
for mode in ("plain", "sym"):
    note_t = toks(note_text, mode == "sym")
    print(f"## mode={mode} min_run={MIN_RUN} note_tokens={len(note_t)}")
    for path in FILES:
        with open(path, encoding="utf-8") as fh:
            ft = toks(fh.read(), mode == "sym")
        res = maximal_runs(note_t, ft)
        if not res:
            print(f"{path}\t(no shared run >= {MIN_RUN})")
        for phrase, (k, lns) in sorted(res.items(), key=lambda kv: (-kv[1][0], kv[0])):
            print(f"{path}\t{k}\tL{','.join(map(str, sorted(lns)))}\t{phrase}")
