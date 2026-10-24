"""Throwaway verifier: maximal shared word runs between the private note and packet text files.

Usage: python3 -I verify_overlap_r4.py NOTE PACKET [MIN_RUN] [--greek]
Prints to stdout only. Never writes files.
"""
import os
import re
import sys
from collections import defaultdict

args = [a for a in sys.argv[1:] if not a.startswith("--")]
GREEK = "--greek" in sys.argv
NOTE, PACKET = args[0], args[1]
MIN_RUN = int(args[2]) if len(args) > 2 else 4

GMAP = {
    "ξ": " xi ", "Ξ": " xi ", "ζ": " zeta ", "τ": " tau ", "Λ": " lambda ", "λ": " lambda ",
    "ℓ": " ell ", "θ": " theta ", "Θ": " theta ", "ω": " omega ", "π": " pi ", "Φ": " phi ",
    "φ": " phi ", "Γ": " gamma ", "∝": " ", "≈": " ", "±": " ", "√": " sqrt ", "ℤ": " z ",
    "≥": " ", "≤": " ", "↔": " ", "→": " ", "−": " ", "–": " ", "—": " ",
}

WORD = re.compile(r"[^\W_]+", re.UNICODE)


def toks(text):
    if GREEK:
        for k, v in GMAP.items():
            text = text.replace(k, v)
    return [w.lower() for w in WORD.findall(text)]


with open(NOTE, encoding="utf-8") as fh:
    note = toks(fh.read())

index = defaultdict(list)
for j in range(len(note) - MIN_RUN + 1):
    index[tuple(note[j:j + MIN_RUN])].append(j)


def runs(ft):
    out = {}
    for i in range(len(ft) - MIN_RUN + 1):
        for j in index.get(tuple(ft[i:i + MIN_RUN]), ()):
            if i > 0 and j > 0 and ft[i - 1] == note[j - 1]:
                continue  # not left-maximal
            k = MIN_RUN
            while i + k < len(ft) and j + k < len(note) and ft[i + k] == note[j + k]:
                k += 1
            phrase = " ".join(ft[i:i + k])
            out.setdefault(phrase, [k, 0])
            out[phrase][1] += 1
    return out


for root, dirs, files in os.walk(PACKET):
    dirs.sort()
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for name in sorted(files):
        path = os.path.join(root, name)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except UnicodeDecodeError:
            print("SKIP(binary)", os.path.relpath(path, PACKET))
            continue
        res = runs(toks(text))
        rel = os.path.relpath(path, PACKET)
        for phrase, (k, c) in sorted(res.items(), key=lambda kv: (-kv[1][0], kv[0])):
            print(f"{rel}\t{k}\t{c}\t{phrase}")
