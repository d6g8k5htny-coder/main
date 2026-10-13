"""Fixer edits for Specifications/RN.lean (item spec_rn). Every replacement is exact-match and
must occur exactly once; otherwise the script aborts without writing. Scientific effect: NONE."""
import pathlib, sys

F = pathlib.Path("<SCRATCH>/repos/math/specifications/lean_statements_20261010/Specifications/RN.lean")
txt = F.read_text(encoding="utf-8")
orig = txt
edits = []

def rep(old, new, count=1):
    global txt
    n = txt.count(old)
    if n != count:
        print("ABORT: expected", count, "occurrence(s), found", n, "for:\n", old[:160])
        sys.exit(1)
    txt = txt.replace(old, new)
    edits.append((old[:60], n))

# ---------------------------------------------------------------- blocking 1: pin regime gate
rep("""`Q_r`, the typed endpoint weight `W_r`, the tilted law `Q_r^W = (W_r/Z_r) Q_r` and the
contact kernel `Lambda_j` are GIVEN; their Gaussian-regression construction, the
determinant form of the weight and the Kac-Rice representation of the kernel are informal
in the sources and are not transcribed.""",
"""`Q_r`, the typed endpoint weight `W_r`, the tilted law `Q_r^W = (W_r/Z_r) Q_r` and the
contact kernel `Lambda_j` are GIVEN; their Gaussian-regression construction, the
determinant form of the weight and the Kac-Rice representation of the kernel are informal
in the sources and are not transcribed. The pin equations, the positivity of the normalizer
and the exact tilt are required only in the pin regime `0 < r < L`, this module's reading of
the sources' "For small r>0 the pins are" (for `0 < r < L` the two pins are distinct modulo
`L Z^d`, so the pin equations are consistent with `L`-periodicity); outside that regime the
laws are unconstrained given probability measures.""")

rep("""paths; `Q_r` is a probability measure under which the six/`2(d+1)` pin equations hold almost
surely; `W_r ≥ 0` vanishes unless `H_M` is negative definite and `H_S` has index `d-1`;
`Z_r = E_(Q_r) W_r > 0`; `dQ_r^W = (W_r/Z_r) dQ_r` exactly (`Measure.withDensity`); and
`Q_r^W` is a probability measure. The Gaussian-regression construction of `Q_r`, the
determinant factors `|det H_M| |det H_S|` of `W_r` and the Kac-Rice contact representation
(13) of `Lambda_j` are informal in the sources and are NOT transcribed. Existence of such an
object is not asserted.""",
"""paths; `Q_r` is a probability measure under which, in the pin regime `0 < r < L`, the
`2(d+1)` pin equations hold almost surely; `W_r ≥ 0` vanishes unless `H_M` is negative
definite and `H_S` has index `d-1`; in the same regime `Z_r = E_(Q_r) W_r > 0` and
`dQ_r^W = (W_r/Z_r) dQ_r` exactly (`Measure.withDensity`); and `Q_r^W` is a probability
measure. The regime `0 < r < L` is this module's reading of the source clause "For small r>0
the pins are": for `0 < r < L` the pins `M`, `S` are distinct modulo `L Z^d` (`‖S - M‖ = r`
while every nonzero lattice vector has norm at least `L`), so the pin equations are
consistent with `L`-periodicity; without the gate the structure would have no inhabitant for
`L > 0` (at `r = L` along a coordinate frame the two pins coincide on the torus, so the pin
values `b` and `b - k r^3` would be prescribed at one point). The Gaussian-regression
construction of `Q_r`, the determinant factors `|det H_M| |det H_S|` of `W_r` and the
Kac-Rice contact representation (13) of `Lambda_j` are informal in the sources and are NOT
transcribed. Existence of such an object is not asserted (a Dirac mass at one smooth
`L`-periodic sample path with the prescribed pin values, a nondegenerate maximum at `M` and a
nondegenerate index-`(d-1)` saddle at `S`, with `W = 1` on that path, would satisfy the gated
fields; this consistency remark is not a statement of this module).""")

rep("""Interface: every field is a given property of the law, never derived; in particular `Λ` is only a given kernel (its continuity and positivity are part of the Theorem A statement, not of this interface).
Does not claim: that `Q_r` is the Gaussian regression of any field; that `W_r` equals `F_d(H_M) F_(d-1)(H_S)`; the normalizer floor `Z_r >= z_* r^2`; that `Λ` is given by (13); existence or uniqueness of an object satisfying the interface. -/""",
"""Interface: every field is a given property of the law, never derived; in particular `Λ` is only a given kernel (its continuity and positivity are part of the Theorem A statement, not of this interface); the pin regime `0 < r < L` gating `Q_pinned`, `Z_pos` and `QW_eq` is part of the hypothesis fields, not a derived fact.
Does not claim: that `Q_r` is the Gaussian regression of any field; that `W_r` equals `F_d(H_M) F_(d-1)(H_S)`; the normalizer floor `Z_r >= z_* r^2`; that `Λ` is given by (13); existence or uniqueness of an object satisfying the interface (an inhabitant is not asserted; the consistency remark above is not stated); that `0 < r < L` is the sources' unspecified `small r` threshold. -/""")

rep("""  /-- Section 1 clause: the pins "f(M)=b, f(S)=b-k r^3, grad f(M)=grad f(S)=0" hold
  `Q_r`-almost surely for `r > 0`. -/
  Q_pinned : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r →
    ∀ᵐ ω ∂(Q r b k R),""",
"""  /-- Section 1 clause: the pins "f(M)=b, f(S)=b-k r^3, grad f(M)=grad f(S)=0" hold
  `Q_r`-almost surely in the pin regime `0 < r < L` (source: "For small r>0 the pins are"). -/
  Q_pinned : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ᵐ ω ∂(Q r b k R),""")

rep("""  /-- Section 1 clause: the "FULL normalizer" `Z_r=E_(Q_r) W_r` is positive for `r > 0`. -/
  Z_pos : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r →
    0 < ∫ ω, W r b k R ω ∂(Q r b k R)""",
"""  /-- Section 1 clause: the "FULL normalizer" `Z_r=E_(Q_r) W_r` is positive in the pin
  regime `0 < r < L`. -/
  Z_pos : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    0 < ∫ ω, W r b k R ω ∂(Q r b k R)""")

rep("""  /-- Section 1 clause: `dQ_r^W=(W_r/Z_r)dQ_r`, exactly, for `r > 0`. -/
  QW_eq : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r →
    QW r b k R = (Q r b k R).withDensity""",
"""  /-- Section 1 clause: `dQ_r^W=(W_r/Z_r)dQ_r`, exactly, in the pin regime `0 < r < L`. -/
  QW_eq : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    QW r b k R = (Q r b k R).withDensity""")

# ---------------------------------------------------------------- blocking 2: Theorem O small-r form
rep("""/-- Shape of Theorem O (sharp ordinary-Poisson obstruction): for a family of laws
`μ_r` on `ℕ` (the law of the window count `N_r`) satisfying only (H1) `E N_r ≤ A r^3` and
(H2) `P(N_r ≥ 2) ≥ a r^3` with `a > 0`, for all sufficiently small `r`,
`(a/2) r^3 ≤ inf_{lambda ≥ 0} dTV(Law N_r, Pois(lambda)) ≤ A r^3`: every nonnegative
intensity is at distance at least `(a/2) r^3`, and some intensity is within `A r^3`.""",
"""/-- Shape of Theorem O (sharp ordinary-Poisson obstruction): for a family of laws
`μ_r` on `ℕ` (the law of the window count `N_r`) which, for all sufficiently small `r`
(that is, for `0 < r ≤ r_0` for some `r_0 > 0`), are probability mass functions with
summable mean satisfying only (H1) `E N_r ≤ A r^3` and (H2) `P(N_r ≥ 2) ≥ a r^3` with
`a > 0`: then, for all sufficiently small `r`,
`(a/2) r^3 ≤ inf_{lambda ≥ 0} dTV(Law N_r, Pois(lambda)) ≤ A r^3`: every nonnegative
intensity is at distance at least `(a/2) r^3`, and some intensity is within `A r^3`. The
hypotheses are imposed only on `0 < r ≤ r_0`, as in the source ("for all sufficiently small
r"); imposing (H2) for every `r > 0` would be unsatisfiable (`a r^3 ≤ 1` fails for large
`r`) and would make the statement hold for every family. The source's remark that the
premises imply `2a ≤ A` is not taken as a hypothesis.""")

rep("""Interface: the laws `μ_r` are given probability mass functions on `ℕ` (nonnegative, summing to one, with summable mean); the hypotheses (H1), (H2) are hypotheses; nothing ties `μ_r` to the Gaussian model here (the source says the only analytic inputs are H1, H2, Hq).
Does not claim: that H1 or H2 hold for the Gaussian window count""",
"""Interface: the laws `μ_r` are given probability mass functions on `ℕ` (nonnegative, summing to one, with summable mean) for `0 < r ≤ r_0`; the hypotheses (H1), (H2) are hypotheses on that range only; nothing ties `μ_r` to the Gaussian model here (the source says the only analytic inputs are H1, H2, Hq).
Does not claim: the result (Theorem O) or its proof; that `2a ≤ A` (derived in the source, not a hypothesis here); that H1 or H2 hold for the Gaussian window count""")

rep("""def RareClusterPoissonObstructionO (μ : ℝ → ℕ → ℝ) (a A : ℝ) : Prop :=
  0 < a → a ≤ A →
  (∀ r : ℝ, 0 < r →
    (∀ n : ℕ, 0 ≤ μ r n) ∧ Summable (μ r) ∧ ∑' n : ℕ, μ r n = 1 ∧
    Summable (fun n : ℕ => (n : ℝ) * μ r n) ∧ ∑' n : ℕ, (n : ℝ) * μ r n ≤ A * r ^ 3 ∧
    a * r ^ 3 ≤ 1 - μ r 0 - μ r 1) →
  ∃ rStar : ℝ, 0 < rStar ∧ ∀ r : ℝ, 0 < r → r ≤ rStar →""",
"""def RareClusterPoissonObstructionO (μ : ℝ → ℕ → ℝ) (a A : ℝ) : Prop :=
  0 < a →
  (∃ r0 : ℝ, 0 < r0 ∧ ∀ r : ℝ, 0 < r → r ≤ r0 →
    (∀ n : ℕ, 0 ≤ μ r n) ∧ Summable (μ r) ∧ ∑' n : ℕ, μ r n = 1 ∧
    Summable (fun n : ℕ => (n : ℝ) * μ r n) ∧ ∑' n : ℕ, (n : ℝ) * μ r n ≤ A * r ^ 3 ∧
    a * r ^ 3 ≤ 1 - μ r 0 - μ r 1) →
  ∃ rStar : ℝ, 0 < rStar ∧ ∀ r : ℝ, 0 < r → r ≤ rStar →""")

# ---------------------------------------------------------------- minor: Theorem N does-not-claim lines
HOLD = "; nonauthor review or acceptance (the landing disposition HOLD_WITH_DOMAIN of rn-count-interface is transcribed by the register, not here). -/"
rep("""Does not claim: that any count of the Gaussian model satisfies the moment hypothesis; the result or its proof; any rate in `r`. -/""",
    """Does not claim: that any count of the Gaussian model satisfies the moment hypothesis; the result or its proof; any rate in `r`""" + HOLD)
rep("""Does not claim: that the exponent can be rounded up to `3`; that any Gaussian count satisfies the hypotheses; the result or its proof. -/""",
    """Does not claim: that the exponent can be rounded up to `3`; that any Gaussian count satisfies the hypotheses; the result or its proof""" + HOLD)
rep("""Does not claim: the equivalence with a bounded conditional mean, the lower comparison `q ≥ c r^3`, or anything about the Gaussian model. -/""",
    """Does not claim: the equivalence with a bounded conditional mean, the lower comparison `q ≥ c r^3`, or anything about the Gaussian model; the result or its proof""" + HOLD)
rep("""the exponential-tail remark after (N5); that the construction says anything about the Gaussian field. -/""",
    """the exponential-tail remark after (N5); that the construction says anything about the Gaussian field; the result or its proof""" + HOLD)
rep("""that the logarithmic loss is attained (the source's remark about (N4)); anything about the Gaussian model. -/""",
    """that the logarithmic loss is attained (the source's remark about (N4)); anything about the Gaussian model; the result or its proof""" + HOLD)

# ---------------------------------------------------------------- minor: Corollary D uses the source's N(E) = sum_j N_j(E)
rep("""/-- `N_j(E)`: the number of index-`j` critical points of `g` in `A` at all heights.
""",
"""/-- `N(E) = sum_j N_j(E)`: the all-index window count of the remote-collision note, the sum
over `j = 0, ..., d` of the index-`j` window counts (critical points with nonsingular Hessian
of index `j`; an index above `d` does not occur in `R^d`).

Source: d6g8k5htny-coder/Math- commit 6c020d6a72936632f2055122e71a7818a4ff497e path frontiers/remote_collision_20260928/PROOF.md sha256 b9b8b58fd8266db7ffe6537078003888ef138b445d3289ec5588f116af9050c2
Anchor: "N(E)   = sum_j N_j(E)."
Interface: none.
Does not claim: finiteness of the counted sets; equality with `windowCountAll` (the two agree only off the degenerate locus). -/
noncomputable def windowCountIndexSum (d : ℕ) (g : E d → ℝ) (b k r : ℝ) (A : Set (E d)) : ℕ :=
  ∑ j ∈ Finset.range (d + 1), windowCount d g b k r j A

/-- `N_j(E)`: the number of index-`j` critical points of `g` in `A` at all heights.
""")

rep("""`E_(Q_r^W) N(E)(N(E)-1) ≤ C r^5 |E|` and `Q_r^W{ N(E) ≥ 2 } ≤ C r^5 |E|`, where `N(E)` is
the all-index window count in `E`. Pairs at every mutual separation inside `D_ρ`, down to
coincidence, are included.""",
"""`E_(Q_r^W) N(E)(N(E)-1) ≤ C r^5 |E|` and `Q_r^W{ N(E) ≥ 2 } ≤ C r^5 |E|`, where
`N(E) = sum_j N_j(E)` is the sum of the index-`j` window counts in `E`
(`windowCountIndexSum`, the source's definition). Pairs at every mutual separation inside
`D_ρ`, down to coincidence, are included.""")

rep("""pairs with a witness near the pins; any numerical constant; nonauthor review or acceptance. -/
def RemoteCollisionSecondFactorialD""",
"""pairs with a witness near the pins; equality of `N(E)` with the all-critical-point count `windowCountAll` (they agree only off the degenerate locus); any numerical constant; nonauthor review or acceptance. -/
def RemoteCollisionSecondFactorialD""")

rep("""windowCountAll d (I.F ω) b k r A""", """windowCountIndexSum d (I.F ω) b k r A""", count=2)

# ---------------------------------------------------------------- minor: torusWindowCount caveat on its four Props
CAV = "; that the count `torusWindowCount` (every critical point off the pins, degenerate ones included) equals the sum of the index counts: it dominates the source's all-index `N`, and the two agree only off the degenerate locus"
rep("""that the logarithm is sharp or removable; any numerical constant; nonauthor review or acceptance. -/""",
    """that the logarithm is sharp or removable; any numerical constant; nonauthor review or acceptance""" + CAV + ". -/")
rep("""a lower bound for `q ≥ 3`; Corollary P; any numerical constant; nonauthor review or acceptance. -/""",
    """a lower bound for `q ≥ 3`; Corollary P; any numerical constant; nonauthor review or acceptance""" + CAV + ". -/")
rep("""any statement about the positions of the pairs; any numerical constant; nonauthor review or acceptance. -/""",
    """any statement about the positions of the pairs; any numerical constant; nonauthor review or acceptance""" + CAV + ". -/")
rep("""any numerical constant; nonauthor review or acceptance; any register transition proposed by the reconciliation. -/""",
    """any numerical constant; nonauthor review or acceptance; any register transition proposed by the reconciliation""" + CAV + ". -/")

# ---------------------------------------------------------------- header: mention the new helper
rep("""contact kernel `Lambda_j` are GIVEN; their Gaussian-regression construction, the""",
    """contact kernel `Lambda_j` are GIVEN; their Gaussian-regression construction, the""")  # no-op guard: header still present

assert txt != orig
F.write_text(txt, encoding="utf-8")
print("applied", len(edits), "replacements;", len(txt.encode()), "bytes")
