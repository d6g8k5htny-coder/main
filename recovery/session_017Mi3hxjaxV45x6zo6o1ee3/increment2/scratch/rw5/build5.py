import pathlib, re, sys
SP = pathlib.Path("<SCRATCH>")
R = SP / "rw5"
src205 = (SP / "wtw3/frontiers/c8_window_coefficient_d3_torus_20260930/window_coefficient_d3.py").read_text()
src199 = (SP / "wtd4/coefficients/side24_d4_20260930/cone_moment_d4.py").read_text()
imports = src205[src205.index("import argparse"):src205.index("MUTANTS = (")]
toolkit = src205[src205.index("# ---------------- interval arithmetic"):src205.index("def simpson(")]
i3 = src205[src205.index("# ---------------- the d = 3 reference birth integral"):src205.index("def scaled(")]
alg = src199[src199.index("def padd("):src199.index("def psubst(")]
alg = alg[:alg.index("def ppow(")]            # padd, pmul, pscale, pconst, pvar
pins = sys.argv[1] if len(sys.argv) > 1 else "{}"
consts = (R / "consts5.py").read_text().replace("PINS_PLACEHOLDER", pins)
newcode = (R / "newcode5.py").read_text().replace("POLY_ALGEBRA_PLACEHOLDER\n", alg.rstrip() + "\n")
out = (R / "head5.py").read_text() + imports + consts + "\n\n" + toolkit + i3.rstrip() + "\n" + newcode + (R / "maincode5.py").read_text()
target = SP / "wtw5/frontiers/c8_window_coefficient_d5_20261001/window_coefficient_d5.py"
target.write_text(out)
print(len(out.splitlines()), "lines")
