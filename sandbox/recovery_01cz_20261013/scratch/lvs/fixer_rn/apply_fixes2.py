"""Second fixer pass: re-sync the shared docstrings with Field.lean where its new wording is true
in RN.lean too (HessianHasInertia, torusDist), and state the exact scope of identity. Exact-match,
count-asserted replacements. Scientific effect: NONE."""
import pathlib, sys
F = pathlib.Path("<SCRATCH>/repos/math/specifications/lean_statements_20261010/Specifications/RN.lean")
txt = F.read_text(encoding="utf-8")
def rep(old, new):
    global txt
    n = txt.count(old)
    if n != 1: print("ABORT count", n, old[:100]); sys.exit(1)
    txt = txt.replace(old, new)
rep("""positive definite. For a symmetric form this is equivalent to the eigenvalue count.
""", """positive definite. Intended to agree with the eigenvalue count for symmetric (`C^2`)
Hessians; not shown here.
""")
rep("""`|x - y + L n|`. It vanishes exactly when `x` and `y` represent the same torus point.
""", """`|x - y + L n|`. For `L > 0` it is intended to vanish exactly when `x` and `y` represent the
same torus point; not shown here.
""")
rep("""This module is self-contained: it imports only Mathlib and re-declares, verbatim up to the
namespace, the shared objects of `Specifications/Field.lean` that it needs (ambient space,
lattice vectors, Hessian form, critical points, inertia, pins, torus distance, fundamental
domain). The duplication is deliberate at this layer and is recorded as such.""",
"""This module is self-contained: it imports only Mathlib and re-declares, up to the
namespace, the shared objects of `Specifications/Field.lean` that it needs (ambient space,
lattice vectors, Hessian form, critical points, inertia, pins, torus distance, fundamental
domain): the nine bodies are byte-identical to Field.lean's, and so are the docstrings except
the `Does not claim:` lines of `pinM` and `pinS`, which are module-specific (Field.lean
records that no statement there consumes the pins; here the interface and the window
statements consume them). The duplication is deliberate at this layer and is recorded as
such.""")
rep("""/-! ### Shared objects, re-declared from `Specifications/Field.lean` (same text, this
namespace). The duplication is deliberate: modules of this packet import only Mathlib. -/""",
"""/-! ### Shared objects, re-declared from `Specifications/Field.lean` (same bodies, this
namespace; see the module header for the two module-specific `Does not claim:` lines). The
duplication is deliberate: modules of this packet import only Mathlib. -/""")
F.write_text(txt, encoding="utf-8"); print("pass 2 applied;", len(txt.encode()), "bytes")
