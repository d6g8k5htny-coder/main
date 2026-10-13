"""Second pass: reword the prose added by apply_fixes.py so that no comment contains the bare
words `instance`/`theorem` (lowercase) and no docstring line starts with a Lean command word at
column 0 (the author's whole-text static_check.py and similar vocabulary scans flag them; the
packet's comment-stripped check.py does not). Shared block is edited in Field.lean and
re-propagated byte-identically into Lifetime.lean. Scientific effect: NONE."""
import pathlib, sys

PKT = pathlib.Path("<SCRATCH>/repos/math/specifications/lean_statements_20261010")
FIELD = PKT / "Specifications" / "Field.lean"
LIFE = PKT / "Specifications" / "Lifetime.lean"


def rep(text: str, old: str, new: str, count: int = 1, label: str = "") -> str:
    n = text.count(old)
    if n != count:
        sys.exit(f"replacement {label!r}: expected {count} occurrence(s), found {n}")
    return text.replace(old, new)


field = FIELD.read_text(encoding="utf-8")
life = LIFE.read_text(encoding="utf-8")

field = rep(field,
    "  -- Its sigma-algebra (square-bracket field, as in Mathlib's bundled categories). It is not a\n"
    "  -- global instance after this declaration; consumers bring it into scope with `haveI := G.mΩ`.\n",
    "  -- Its sigma-algebra (square-bracket field, as in Mathlib's bundled categories). It is not\n"
    "  -- registered for typeclass resolution after this declaration; consumers bring it into\n"
    "  -- scope with `haveI := G.mΩ`.\n",
    1, "mΩ comment")
field = rep(field,
    "The source's densities are only defined up to such versions. The sigma-algebra of the sample\n"
    "space is brought into scope by the inlined term-level `haveI := G.mΩ` (an instance-implicit\n"
    "structure field is not a global instance after the declaration).\n",
    "The source's densities are only defined up to such versions. The sigma-algebra of the sample\n"
    "space is brought into scope by the inlined term-level `haveI := G.mΩ` (a square-bracket\n"
    "field of a bundled record is not registered for typeclass resolution after the declaration).\n",
    1, "IsLifetimeDensityVersion haveI note")

f0 = field.index("/-- The ambient Euclidean space")
f1 = field.index("\nend UniversalLaw.Spec.Field")
shared = field[f0:f1]
l0 = life.index("/-- The ambient Euclidean space")
l1 = life.index("\n/-! ### Populations of pairs -/")
life = life[:l0] + shared + life[l1:]

life = rep(life,
    "The planar statements (E4) and (E11) carry the parent upper bound (1.1), for the same\n"
    "typed pair law at `d = 2`, as an explicit hypothesis, as the source does (\"If the precise\n"
    "Theorem A upper bound in [LP] is consumed\"; \"the upper consumes the already identified\n"
    "parent upper theorem\"); the direct lower bound (E2) is unconditional, as in the source.\n",
    "The planar statements (E4) and (E11) carry the parent upper bound (1.1), for the same\n"
    "typed pair law at `d = 2`, as an explicit hypothesis, as the source does (\"If the precise\n"
    "Theorem A upper bound in [LP] is consumed\"; its Section 4 says the upper half of (E11)\n"
    "consumes the parent upper bound already identified there); the direct lower bound (E2) is\n"
    "unconditional, as in the source.\n",
    1, "Lifetime header")
life = rep(life,
    "/-- Shape of the planar compact-window density gap (E11): in `d = 2`, if (1.1) holds for the\n"
    "typed pair law on the windows `B`, `K` (the source: \"the upper consumes the already identified\n"
    "parent upper theorem\", identified in its Corollary E2 as \"the precise Theorem A upper bound in\n"
    "[LP]\"), then for windows of positive length with `kLo > 0` there are a distance cutoff\n",
    "/-- Shape of the planar compact-window density gap (E11): in `d = 2`, if (1.1) holds for the\n"
    "typed pair law on the windows `B`, `K` (the source says the upper half consumes the parent\n"
    "upper bound already identified in its Corollary E2 as \"the precise Theorem A upper bound in\n"
    "[LP]\"), then for windows of positive length with `kLo > 0` there are a distance cutoff\n",
    1, "E11 docstring")

FIELD.write_text(field, encoding="utf-8")
LIFE.write_text(life, encoding="utf-8")
print("Field.lean", len(field.encode()), "bytes; Lifetime.lean", len(life.encode()), "bytes; written")
