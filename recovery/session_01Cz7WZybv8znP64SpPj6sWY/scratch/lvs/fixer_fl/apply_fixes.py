"""Fixer for item spec_field_lifetime: applies the verifier's findings to the two statement
modules by exact string replacement. Scientific effect: NONE. Statement-only modules; nothing
here proves anything. Every replacement asserts its expected occurrence count so that a drifted
module fails loudly instead of being silently half-edited."""
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

# ---------------------------------------------------------------- shared block (edited in Field.lean)
# minor 7a: periodicCovariance docstring hedge
field = rep(field,
    "sum at `z = 0`, so `K_L(0) = 1` (variance one).\n",
    "sum at `z = 0`, so `K_L(0) = 1` (variance one) whenever that denominator is nonzero (the\n"
    "summability field is meant to provide this; not shown here).\n",
    1, "periodicCovariance hedge")
# minor 7b: HessianHasInertia docstring hedge
field = rep(field,
    "positive definite. For a symmetric form this is equivalent to the eigenvalue count.\n",
    "positive definite. Intended to agree with the eigenvalue count for symmetric (`C^2`)\n"
    "Hessians; not shown here.\n",
    1, "HessianHasInertia hedge")
# minor 7c: torusDist docstring hedge
field = rep(field,
    "`|x - y + L n|`. It vanishes exactly when `x` and `y` represent the same torus point.\n",
    "`|x - y + L n|`. For `L > 0` it is intended to vanish exactly when `x` and `y` represent the\n"
    "same torus point; not shown here.\n",
    1, "torusDist hedge")
# minor 4: pinM / pinS are auxiliaries consumed by no statement (both docstrings carry the same line)
field = rep(field,
    "Does not claim: that `u` is a unit vector (a hypothesis where used). -/\n",
    "Does not claim: that `u` is a unit vector; this pin is an auxiliary definition consumed by no statement of this packet (the typed pair law `p` is a primitive), so it is attached to no claim. -/\n",
    2, "pinM/pinS does-not-claim")
# minor 6: IsLifetimeDensityVersion does-not-claim wording (measurability is required, not disclaimed)
field = rep(field,
    "Does not claim: existence of a version; the weighted Kac-Rice identity; the radial ledger (10.2); measurability of the pair count as a function of the sample. -/\n",
    "Does not claim: existence of a version; the weighted Kac-Rice identity; the radial ledger (10.2). Measurability of the pair count as a function of the sample is REQUIRED by the predicate (through `Integrable`, hence `AEStronglyMeasurable`), not established here. -/\n",
    1, "IsLifetimeDensityVersion does-not-claim")
# minor 9: bring the bundled sigma-algebra into scope with an inlined term-level haveI
field = rep(field,
    "The source's densities are only defined up to such versions.\n\nSource:",
    "The source's densities are only defined up to such versions. The sigma-algebra of the sample\n"
    "space is brought into scope by the inlined term-level `haveI := G.mΩ` (an instance-implicit\n"
    "structure field is not a global instance after the declaration).\n\nSource:",
    1, "IsLifetimeDensityVersion docstring haveI note")
field = rep(field,
    "    (Pair : (E d → ℝ) → E d → E d → Prop) (ν : ℝ → ℝ) : Prop :=\n"
    "  (∀ ℓ : ℝ, 0 ≤ ν ℓ) ∧ Measurable ν ∧\n",
    "    (Pair : (E d → ℝ) → E d → E d → Prop) (ν : ℝ → ℝ) : Prop :=\n"
    "  haveI := G.mΩ\n"
    "  (∀ ℓ : ℝ, 0 ≤ ν ℓ) ∧ Measurable ν ∧\n",
    1, "IsLifetimeDensityVersion haveI")
field = rep(field,
    "  -- Its sigma-algebra (square-bracket field, as in Mathlib's bundled categories).\n",
    "  -- Its sigma-algebra (square-bracket field, as in Mathlib's bundled categories). It is not a\n"
    "  -- global instance after this declaration; consumers bring it into scope with `haveI := G.mΩ`.\n",
    1, "GaussianFieldInterface mΩ comment")

# ---------------------------------------------------------------- propagate the shared block to Lifetime.lean
f0 = field.index("/-- The ambient Euclidean space")
f1 = field.index("\nend UniversalLaw.Spec.Field")
shared = field[f0:f1]
l0 = life.index("/-- The ambient Euclidean space")
l1 = life.index("\n/-! ### Populations of pairs -/")
old_shared = life[l0:l1]
# sanity: the old shared block of Lifetime must equal the old shared block of Field (before edits)
old_field = FIELD.read_text(encoding="utf-8")
assert old_field[old_field.index("/-- The ambient Euclidean space"):old_field.index("\nend UniversalLaw.Spec.Field")] == old_shared, \
    "pre-existing shared blocks differ; refusing to regenerate"
life = life[:l0] + shared + life[l1:]

# ---------------------------------------------------------------- Lifetime-specific edits
# header: the planar statements carry (1.1) as a hypothesis
life = rep(life,
    "coefficient value; it does not state Theorem A as consumed by Theorem R; and it does\n"
    "not state the compact-window `O(ell^(2/3))` difference for the unrestricted densities.\n",
    "coefficient value; it does not state Theorem A as consumed by Theorem R; and it does\n"
    "not state the compact-window `O(ell^(2/3))` difference for the unrestricted densities.\n"
    "The planar statements (E4) and (E11) carry the parent upper bound (1.1), for the same\n"
    "typed pair law at `d = 2`, as an explicit hypothesis, as the source does (\"If the precise\n"
    "Theorem A upper bound in [LP] is consumed\"; \"the upper consumes the already identified\n"
    "parent upper theorem\"); the direct lower bound (E2) is unconditional, as in the source.\n",
    1, "Lifetime header")
# minor 5: PairLawInterface does-not-claim names the phantom parameters
life = rep(life,
    "Does not claim: existence of the regression law, the typed weight, the full normalizer `Z_r > 0`, or that `p` is the probability of any event under any measure declared here. -/\n",
    "Does not claim: existence of the regression law, the typed weight, the full normalizer `Z_r > 0`, or that `p` is the probability of any event under any measure declared here; that `p` depends on `G` or `I` in any way (no field mentions them; they only record which field and elder relation the law is meant to refer to). -/\n",
    1, "PairLawInterface does-not-claim")
# blocking: E4 conditional on (1.1), as the source states Corollary E2
e4_start = life.index("/-- Shape of the planar two-sided selection rate (E4)")
e4_end = life.index("\n\n/-- Shape of the planar compact-window density gap (E11)")
new_e4 = '''/-- Shape of the planar two-sided selection rate (E4), Corollary E2 of the source: if (1.1)
holds for the same typed pair law at `d = 2` (the source: "If the precise Theorem A upper
bound in [LP] is consumed"), then `c r^3 ≤ 1 - p_r(b,k,R) ≤ C r^3` for `0 < r ≤ r_*`,
uniformly on the compact planar model. The parent upper input is an explicit hypothesis of
the statement, not folded into its conclusion.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/window_multiplicity_laws_20260928/ELDER_LOWER_AND_DENSITY_GAP.md sha256 8ce05a43bde8e0d718f09e2da1ffe75d3ec5893489254afdbf5b58b602cdba0c
Anchor: "c r^3 <= 1-p_r(b,k,R) <= C r^3."
Interface: `G`, `I`, `PL` are given objects as in `PlanarElderLowerE2`; the hypothesis `ParentUniformSelectionA 2 L G I PL bLo bHi kLo kHi` is the consumed upper input (a statement shape, not the parent result).
Does not claim: the upper half independently of the parent (1.1) (the source says the upper half consumes it, and the statement is conditional on it); the parent (1.1) itself; the result or its proof; numerical constants; nonauthor review or acceptance. -/
def PlanarTwoSidedRateE4 (L : ℝ) (G : GaussianFieldInterface 2 L) (I : SuperlevelBarInterface 2 L)
    (PL : PairLawInterface 2 L G I) (bLo bHi kLo kHi : ℝ) : Prop :=
  ParentUniformSelectionA 2 L G I PL bLo bHi kLo kHi →
    (0 < L → bLo ≤ bHi → 0 < kLo → kLo ≤ kHi →
      ∃ c : ℝ, 0 < c ∧ ∃ C : ℝ, ∃ rStar : ℝ, 0 < rStar ∧
        ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
          ∀ R : OrthonormalBasis (Fin 2) ℝ (E 2),
            c * r ^ 3 ≤ 1 - PL.p r b k R ∧ 1 - PL.p r b k R ≤ C * r ^ 3)'''
life = life[:e4_start] + new_e4 + life[e4_end:]
# minor 2: E11 carries the consumed parent upper input (1.1) as a hypothesis
e11_start = life.index("/-- Shape of the planar compact-window density gap (E11)")
e11_end = life.index("\n\nend UniversalLaw.Spec.Lifetime")
new_e11 = '''/-- Shape of the planar compact-window density gap (E11): in `d = 2`, if (1.1) holds for the
typed pair law on the windows `B`, `K` (the source: "the upper consumes the already identified
parent upper theorem", identified in its Corollary E2 as "the precise Theorem A upper bound in
[LP]"), then for windows of positive length with `kLo > 0` there are a distance cutoff
`r_* > 0`, constants `c_{B,K} > 0`, `C_{B,K}` and `ell_* > 0` such that versions of the
compact-window candidate and ordinary elder densities satisfy
`c_{B,K} ℓ^(2/3) ≤ νc(ℓ) - νe(ℓ) ≤ C_{B,K} ℓ^(2/3)` for all `0 < ℓ ≤ ell_*`. The parent upper
input is an explicit hypothesis, not folded into the conclusion. The lower half, which the
source derives from (E2) alone, is stated under the same hypothesis because the source displays
(E11) as one two-sided inequality; that is a hypothesis the source does not need for the lower
half, recorded here as such.

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/window_multiplicity_laws_20260928/ELDER_LOWER_AND_DENSITY_GAP.md sha256 8ce05a43bde8e0d718f09e2da1ffe75d3ec5893489254afdbf5b58b602cdba0c
Anchor: "       <= C_{B,K} ell^(2/3).                             (E11)"
Interface: `G`, `I` and `PL` are given; the hypothesis `ParentUniformSelectionA 2 L G I PL bLo bHi kLo kHi` is the consumed parent upper input (a statement shape, not the parent result); versions are characterised by the integral identity only; the weighted Kac-Rice / radial ledger of [LP, Sections 9-11] that the source consumes is not transcribed.
Does not claim: the result or its proof; the upper half independently of the parent (1.1); the parent (1.1) itself; the lower half without the hypothesis (the source has it from (E2) alone); sharpness of the `O(1)` remainder of (R1); removal of the compact-mark restriction; that the cutoff `r_*` is the one of (1.1); any numerical constant; nonauthor review or acceptance. -/
def PlanarDensityGapE11 (L : ℝ) (G : GaussianFieldInterface 2 L) (I : SuperlevelBarInterface 2 L)
    (PL : PairLawInterface 2 L G I) (bLo bHi kLo kHi : ℝ) : Prop :=
  ParentUniformSelectionA 2 L G I PL bLo bHi kLo kHi →
    (0 < L → bLo < bHi → 0 < kLo → kLo < kHi →
      ∃ rStar : ℝ, 0 < rStar ∧ ∃ c : ℝ, 0 < c ∧ ∃ C : ℝ, ∃ ℓStar : ℝ, 0 < ℓStar ∧
        ∃ νc νe : ℝ → ℝ,
          IsLifetimeDensityVersion 2 L G (IsWindowPair 2 L rStar bLo bHi kLo kHi) νc ∧
          IsLifetimeDensityVersion 2 L G (IsWindowElderPair 2 L rStar bLo bHi kLo kHi I) νe ∧
          ∀ ℓ : ℝ, 0 < ℓ → ℓ ≤ ℓStar →
            c * ℓ ^ ((2 : ℝ) / 3) ≤ νc ℓ - νe ℓ ∧ νc ℓ - νe ℓ ≤ C * ℓ ^ ((2 : ℝ) / 3))'''
life = life[:e11_start] + new_e11 + life[e11_end:]

FIELD.write_text(field, encoding="utf-8")
LIFE.write_text(life, encoding="utf-8")
print("Field.lean", len(field.encode()), "bytes; Lifetime.lean", len(life.encode()), "bytes; written")
