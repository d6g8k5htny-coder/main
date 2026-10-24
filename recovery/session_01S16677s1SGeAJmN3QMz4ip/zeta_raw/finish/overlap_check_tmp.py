"""Throwaway: list shared word runs between the private note and packet text files.

Prints to stdout only. Never writes files.
"""
import os
import re
import sys
from collections import defaultdict

NOTE = sys.argv[1]
PACKET = sys.argv[2]
MIN_RUN = int(sys.argv[3]) if len(sys.argv) > 3 else 4

TEXT_EXT = {".md", ".py", ".out", ".txt", ".json", ".csv", ".log", ".svg", ".gitignore", ""}

WORD = re.compile(r"[^\W_]+", re.UNICODE)


def toks(text):
    return [w.lower() for w in WORD.findall(text)]


with open(NOTE, encoding="utf-8") as fh:
    note = toks(fh.read())

index = defaultdict(list)
for j in range(len(note) - MIN_RUN + 1):
    index[tuple(note[j:j + MIN_RUN])].append(j)


def runs(ftoks):
    out = {}
    for i in range(len(ftoks) - MIN_RUN + 1):
        key = tuple(ftoks[i:i + MIN_RUN])
        for j in index.get(key, ()):
            # only start maximal runs
            if i > 0 and j > 0 and ftoks[i - 1] == note[j - 1]:
                continue
            k = MIN_RUN
            while i + k < len(ftoks) and j + k < len(note) and ftoks[i + k] == note[j + k]:
                k += 1
            phrase = " ".join(ftoks[i:i + k])
            out.setdefault(phrase, [k, 0])
            out[phrase][1] += 1
    return out


for root, dirs, files in os.walk(PACKET):
    dirs.sort()
    for name in sorted(files):
        path = os.path.join(root, name)
        ext = os.path.splitext(name)[1]
        if ext not in TEXT_EXT and name != ".gitignore":
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except UnicodeDecodeError:
            print("SKIP(binary)", path)
            continue
        res = runs(toks(text))
        if not res:
            continue
        rel = os.path.relpath(path, PACKET)
        for phrase, (k, c) in sorted(res.items(), key=lambda kv: -kv[1][0]):
            print(f"{rel}\t{k}\t{c}\t{phrase}")
