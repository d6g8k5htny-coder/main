"""Throwaway verifier (not part of the packet; writes nothing).

Maximal shared word runs (>= MIN tokens) between the private note and packet files.
Two tokenizations: 'plain' (letters/digits only) and 'sym' (Greek/math symbols mapped to words).
Prints: mode, file, run length, packet line range, packet-side tokens.

Usage: python3 -I paraphrase_verify_ad02.py NOTE FILE [FILE ...] [--min N]
"""
import re
import sys

args = sys.argv[1:]
MIN = 6
if "--min" in args:
    i = args.index("--min")
    MIN = int(args[i + 1])
    del args[i:i + 2]
NOTE, FILES = args[0], args[1:]

SYM = {"ξ": "xi", "Ξ": "xi", "ζ": "zeta", "τ": "tau", "Λ": "lambda", "λ": "lambda", "ℓ": "ell",
       "θ": "theta", "Θ": "theta", "π": "pi", "Φ": "phi", "Γ": "gamma", "γ": "gamma",
       "ρ": "rho", "κ": "kappa", "σ": "sigma", "α": "alpha", "β": "beta", "∝": "propto",
       "≈": "approx", "~": "sim", "∼": "sim", "±": "pm", "√": "sqrt", "≥": "ge", "≤": "le",
       "→": "to", "−": "minus", "²": "2", "³": "3", "⁴": "4", "₀": "0", "∫": "int",
       "∂": "d", "≠": "ne", "ℤ": "z", "ν": "nu", "δ": "delta", "ε": "epsilon"}
WORD = re.compile(r"[^\W_]+")


def tokens(text, sym):
    out = []
    for ln, line in enumerate(text.splitlines(), 1):
        if sym:
            line = "".join(f" {SYM[c]} " if c in SYM else c for c in line)
        for w in WORD.findall(line.lower()):
            out.append((w, ln))
    return out


def runs(a, b):
    """Maximal common runs between token lists a (note) and b (packet), length >= MIN."""
    aw = [w for w, _ in a]
    bw = [w for w, _ in b]
    idx = {}
    for i in range(len(aw) - MIN + 1):
        idx.setdefault(tuple(aw[i:i + MIN]), []).append(i)
    found = {}
    for j in range(len(bw) - MIN + 1):
        for i in idx.get(tuple(bw[j:j + MIN]), ()):
            if i > 0 and j > 0 and aw[i - 1] == bw[j - 1]:
                continue                                    # not left-maximal
            k = MIN
            while i + k < len(aw) and j + k < len(bw) and aw[i + k] == bw[j + k]:
                k += 1
            key = (j, k)
            found[key] = i
    return sorted(found)


note_text = open(NOTE, encoding="utf-8").read()
for mode in ("plain", "sym"):
    nt = tokens(note_text, mode == "sym")
    for f in FILES:
        pt = tokens(open(f, encoding="utf-8").read(), mode == "sym")
        rs = runs(nt, pt)
        print(f"== {mode} {f}: {len(rs)} run(s) >= {MIN}")
        for j, k in rs:
            l0, l1 = pt[j][1], pt[j + k - 1][1]
            print(f"  len={k} lines {l0}-{l1}: {' '.join(w for w, _ in pt[j:j + k])}")
