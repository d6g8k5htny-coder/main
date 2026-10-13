import Mathlib

/-!
# SIDE24 coefficient in dimensions two and three (statements only)

Scientific effect: NONE. Formal progress of every declaration here: `specified`.
Organizational-independence credit of this transcription: 0 (Anthropic / Claude).

This module transcribes, as definitions, interface structures and `Prop`-valued
specifications, the displayed statements of the Layer 0 note
`coefficients/side24_v1/PROOF.md` (object SIDE24-COEFFICIENT-D23-20260924-v1) and of its
`ENCLOSURE.json`: the exact cone moments `D_1 = 4/3` and `D_2 = 29/6 - sqrt 6`, the
reference coefficient `c_{d,ref}` of display (1), the image constant `E` of display (2), the
relative covariance allowance `epsilon = 10^(-108)` of display (3), the twenty-digit
published endpoints, the headline two-dimensional and three-dimensional enclosures, the
reference comparison (4), the Section 2 derivative and lattice-image bounds, the Section 3
joint-covariance ordering in the positive-semidefinite order, the Section 4 Gaussian density
comparison and integrand-ratio bounds, and the Section 1 cone-moment identities with the
Gaussian laws of the displayed variances `8/3`, `5/3`, `1`.

The exact periodic coefficient `c_{d,24}` is "precisely its equation (15.2)" of the parent
(UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1); that formula involves the Gaussian cone expectation
`D_u` of (15.1), which this module cannot write without a conditional-Gaussian construction.
The coefficient is therefore an INTERFACE parameter (`PeriodicCoefficientInterface`): its
defining property (15.2) is a hypothesis field, the cone moment `D_u` is a given field, and the
densities and the conditional variance are pinned to derivatives of `K_24` by hypothesis fields.

What this module does not establish. It consists only of definitions and structures; it
states nothing as a result and proves nothing. It does not enclose the coefficient, does not
evaluate `Gamma(7/6)`, does not prove any of the displayed inequalities, does not identify an
object satisfying an interface here with the object of the source, and does not touch the
parent's results. The Section 5 Stirling/`Gamma(7/6)` remainder procedure is described in the
source as a method, not displayed as a precise statement, and is NOT specified here. The
arithmetic skeleton of these displays (pairing counts, the image constant, `60E < epsilon`,
odd-block minors, exponent inequalities, the `D_2` algebra, endpoint adjacency) is already
kernel-checked on main as the 31 `UniversalLaw.Side24` targets of `formal/` at origin/main
`f6deeba7`; nothing here re-proves it and nothing here encloses the coefficient. Nothing here
promotes, reclassifies or discharges any claim, premise or obligation of the registers;
alignment of these statements with the source is PENDING_INDEPENDENT_REVIEW.

Primary sources (Layer 0, byte-pinned): `d6g8k5htny-coder/Math-` commit
`760340e921ac4ceda296b8118da936f1133e956e`, paths `coefficients/side24_v1/PROOF.md`
(sha256 `c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769`, source id
`side24-proof`) and `coefficients/side24_v1/ENCLOSURE.json` (sha256
`72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811`, source id
`side24-enclosure`). Parent clause (15.2) quoted from source id `lifetime-parent`.
-/

namespace UniversalLaw.Spec.Side24

open MeasureTheory

/-- The ambient Euclidean space `R^d` of the field; `d ∈ {2, 3}` in the source.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Use its centered variance-one field with the exact normalized covariance"
Interface: none; this is the ambient space `EuclideanSpace ℝ (Fin d)`.
Does not claim: anything about the field or the torus `R^d/(24 Z^d)`. -/
abbrev E (d : ℕ) : Type := EuclideanSpace ℝ (Fin d)

/-! ## Exact constants of the note -/

/-- The cone moment `D_1 = 4/3` of the reference law for `m = d - 1 = 1`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "D_1=E[A^2 1{A<0}]=4/3."
Interface: none; a rational constant.
Does not claim: the identity with the Gaussian expectation (that is `ConeMomentD1Identity`); the arithmetic `(1/2)(8/3) = 4/3` is kernel-checked on main (`UniversalLaw.Side24.cone_moment_m1_thirds`), not here. -/
noncomputable def coneMomentD1 : ℝ := 4 / 3

/-- The cone moment `D_2 = 29/6 - sqrt 6` of the reference law for `m = d - 1 = 2`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "        = 29/6-sqrt(6)."
Interface: none; an explicit real constant.
Does not claim: the identity with the Gaussian cone expectation (that is `ConeMomentD2Identity` and `ConeMomentD2Algebra`); the rational part `29/6` and `4 sqrt(3/8) = sqrt 6` are kernel-checked on main (`UniversalLaw.Side24.cone_moment_m2_algebra`), not here. -/
noncomputable def coneMomentD2 : ℝ := 29 / 6 - Real.sqrt 6

/-- The reference cone moment `D_m` as a function of `m = d - 1`: `D_1` for `m = 1`, `D_2` for
`m = 2`, and `0` (unspecified by the source) otherwise.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "c_d,ref = Gamma(7/6)*(3/2)^(1/3)*D_(d-1)"
Interface: none.
Does not claim: any value outside `m ∈ {1, 2}`; the value `0` there is a placeholder, not a cone moment. -/
noncomputable def coneMomentRef (m : ℕ) : ℝ :=
  if m = 1 then coneMomentD1 else if m = 2 then coneMomentD2 else 0

/-- The nonperiodic reference coefficient of display (1):
`c_{d,ref} = Gamma(7/6) (3/2)^(1/3) D_{d-1} / [2 sqrt 3 pi^(d-1) sqrt pi]`, `d = 2, 3`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "/ [2sqrt(3)*pi^(d-1)*sqrt(pi)], d=2,3."
Interface: none; `Real.Gamma`, `Real.sqrt`, `Real.pi` and real powers are Mathlib's.
Does not claim: that this equals the periodic coefficient (the source says it is NOT claimed equal); any value for `d ∉ {2, 3}`; the derivation from the parent formula (that is `ReferenceCoefficientIdentity`); any numerical value of `Gamma(7/6)`. -/
noncomputable def cRef (d : ℕ) : ℝ :=
  Real.Gamma (7 / 6) * (3 / 2 : ℝ) ^ ((1 : ℝ) / 3) * coneMomentRef (d - 1) /
    (2 * Real.sqrt 3 * Real.pi ^ (d - 1) * Real.sqrt Real.pi)

/-- The image constant `E = 21175738586478 * 10^(-125)` of display (2), as a rational over a
power of ten.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "= 21175738586478 *10^(-125)."
Interface: none.
Does not claim: the factorisation `1458*(76*24^6+15) = 21175738586478`, kernel-checked on main (`UniversalLaw.Side24.image_constant_value`), not here. -/
noncomputable def imageConstE : ℝ := 21175738586478 / 10 ^ 125

/-- The relative covariance allowance `epsilon = 10^(-108)` of display (3).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "epsilon=10^(-108),   since60E<epsilon."
Interface: none.
Does not claim: the inequality `60E < epsilon`, kernel-checked on main (`UniversalLaw.Side24.covariance_relative_bound`), not here. -/
noncomputable def epsilon : ℝ := 1 / 10 ^ 108

/-- Published lower endpoint for `d = 2`: `0.07340691930603427103` as a rational over `10^20`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/ENCLOSURE.json sha256 72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811
Anchor: ""lower": "0.07340691930603427103""
Interface: none.
Does not claim: that the coefficient exceeds this number (that is `CoefficientEnclosure`). -/
noncomputable def d2Lower : ℝ := 7340691930603427103 / 10 ^ 20

/-- Published upper endpoint for `d = 2`: `0.07340691930603427104` as a rational over `10^20`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/ENCLOSURE.json sha256 72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811
Anchor: ""upper": "0.07340691930603427104""
Interface: none.
Does not claim: that the coefficient is below this number; adjacency of the two endpoints is kernel-checked on main (`UniversalLaw.Side24.d2_endpoints_adjacent`), not here. -/
noncomputable def d2Upper : ℝ := 7340691930603427104 / 10 ^ 20

/-- Published lower endpoint for `d = 3`: `0.04177593184059834334` as a rational over `10^20`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/ENCLOSURE.json sha256 72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811
Anchor: ""lower": "0.04177593184059834334""
Interface: none.
Does not claim: that the coefficient exceeds this number (that is `CoefficientEnclosure`). -/
noncomputable def d3Lower : ℝ := 4177593184059834334 / 10 ^ 20

/-- Published upper endpoint for `d = 3`: `0.04177593184059834335` as a rational over `10^20`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/ENCLOSURE.json sha256 72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811
Anchor: ""upper": "0.04177593184059834335""
Interface: none.
Does not claim: that the coefficient is below this number; adjacency is kernel-checked on main (`UniversalLaw.Side24.d3_endpoints_adjacent`), not here. -/
noncomputable def d3Upper : ℝ := 4177593184059834335 / 10 ^ 20

/-! ## The exact normalized periodic covariance `K_24` and the reference kernel `K_infty` -/

/-- The lattice vector `24 n` for `n ∈ Z^d`, in the standard coordinate vectors.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "K_24(z) = sum_{n in Z^d} exp(-|z+24n|^2/2)"
Interface: none.
Does not claim: anything beyond the definition of the vector `24 n`. -/
noncomputable def latticeVec (d : ℕ) (n : Fin d → ℤ) : E d :=
  ∑ i, EuclideanSpace.single i (24 * (n i : ℝ))

/-- The unnormalised theta sum `sum_{n in Z^d} exp(-|z+24n|^2/2)` as a `tsum`; it is `0` if the
family is not summable, and summability is the hypothesis `ThetaSummable`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "K_24(z) = sum_{n in Z^d} exp(-|z+24n|^2/2)"
Interface: summability is the separate hypothesis `ThetaSummable d`, never derived here.
Does not claim: convergence of the sum. -/
noncomputable def thetaSum24 (d : ℕ) (z : E d) : ℝ :=
  ∑' n : Fin d → ℤ, Real.exp (-(‖z + latticeVec d n‖ ^ 2) / 2)

/-- The exact normalized covariance
`K_24(z) = sum_n exp(-|z+24n|^2/2) / sum_n exp(-|24n|^2/2)` (variance one: `K_24(0) = 1`).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "/ sum_{n in Z^d} exp(-|24n|^2/2),  d in {2,3}."
Interface: summability of both theta sums is the hypothesis `ThetaSummable d`.
Does not claim: smoothness or positivity of `K_24`, or any property beyond its definition. -/
noncomputable def K24 (d : ℕ) (z : E d) : ℝ :=
  thetaSum24 d z / thetaSum24 d 0

/-- The reference kernel `K_infty(z) = exp(-|z|^2/2)`, used only to define the reference
contact covariance and its coefficient expression.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Use K_infty(z)=exp(-|z|^2/2) only to define a reference CONTACT covariance and its"
Interface: none.
Does not claim: any infinite-volume persistence statement (the source asserts none). -/
noncomputable def Kinf (d : ℕ) (z : E d) : ℝ :=
  Real.exp (-(‖z‖ ^ 2) / 2)

/-- Summability of the Gaussian theta family at every point (so that `thetaSum24` is the sum
the source writes). Taken as a hypothesis wherever `K_24` enters a specification.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "K_24(z) = sum_{n in Z^d} exp(-|z+24n|^2/2)"
Interface: this predicate is itself the hypothesis; the source takes convergence for granted.
Does not claim: that the family is summable (true, but not shown here). -/
def ThetaSummable (d : ℕ) : Prop :=
  ∀ z : E d, Summable (fun n : Fin d → ℤ => Real.exp (-(‖z + latticeVec d n‖ ^ 2) / 2))

/-! ## Section 2: the uniform derivative bound and the lattice-image bound -/

/-- A tuple of `q` unit directions in `R^d` (the "unit directional differentiations", mixed
directions allowed).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "For q<=6 unit directional differentiations of phi(x)=exp(-|x|^2/2), product-rule"
Interface: none.
Does not claim: anything. -/
def IsUnitTuple (d q : ℕ) (u : Fin q → E d) : Prop :=
  ∀ i, ‖u i‖ = 1

/-- The product-rule pairing sum `sum_{j=0}^{floor(q/2)} q! r^(q-2j) / [2^j j! (q-2j)!]`
evaluated at `r = |x|`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "q! |x|^(q-2j) / [2^j j! (q-2j)!]."
Interface: none.
Does not claim: that the sum of the coefficients at `q = 6` is `76` (kernel-checked on main, `UniversalLaw.Side24.pairing_sum_six`), nor the bound itself (`DerivativePairingBound`). -/
noncomputable def pairingSum (q : ℕ) (r : ℝ) : ℝ :=
  ∑ j ∈ Finset.range (q / 2 + 1),
    (Nat.factorial q : ℝ) * r ^ (q - 2 * j) /
      ((2 : ℝ) ^ j * (Nat.factorial j : ℝ) * (Nat.factorial (q - 2 * j) : ℝ))

/-- Section 2, first display: for `q ≤ 6` and unit directions, the absolute `q`-th directional
derivative of `phi(x) = exp(-|x|^2/2)` is at most `exp(-|x|^2/2)` times the pairing sum.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "pairings bound the absolute derivative by"
Interface: none; the derivative is Mathlib's `iteratedFDeriv ℝ q` applied to the direction tuple.
Does not claim: the bound (statement only); validity for `q > 6` (the source restricts to `q ≤ 6`); nothing here encloses the coefficient. -/
def DerivativePairingBound : Prop :=
  ∀ (d q : ℕ), q ≤ 6 → ∀ (x : E d) (u : Fin q → E d), IsUnitTuple d q u →
    |iteratedFDeriv ℝ q (Kinf d) x u| ≤ Real.exp (-(‖x‖ ^ 2) / 2) * pairingSum q ‖x‖

/-- Section 2: at `|x| ≥ 1` every unit-direction contraction of order `q ≤ 6` of
`phi(x) = exp(-|x|^2/2)` is bounded by `76 |x|^6 exp(-|x|^2/2)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "At |x|>=1 this is at most76 |x|^6 e^(-|x|^2/2)."
Interface: none.
Does not claim: the bound (statement only); the coefficient arithmetic `1+15+45+15=76` and `76` bounding the smaller orders are kernel-checked on main (`UniversalLaw.Side24.pairing_sum_six`, `pairing_sum_le_six`), not here; nothing here encloses the coefficient. -/
def DerivativeBoundAtLeastOne : Prop :=
  ∀ (d q : ℕ), q ≤ 6 → ∀ (x : E d), 1 ≤ ‖x‖ → ∀ (u : Fin q → E d), IsUnitTuple d q u →
    |iteratedFDeriv ℝ q (Kinf d) x u| ≤ 76 * ‖x‖ ^ 6 * Real.exp (-(‖x‖ ^ 2) / 2)

/-- Section 2: at zero every unit-direction contraction of order `q ≤ 6` of `phi` has absolute
value at most `15`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "contraction has absolute value at most15."
Interface: none.
Does not claim: the bound (statement only); `5!! = 15` is kernel-checked on main (`UniversalLaw.Side24.sixth_moment_double_factorial`), not here. -/
def DerivativeBoundAtZero : Prop :=
  ∀ (d q : ℕ), q ≤ 6 → ∀ (u : Fin q → E d), IsUnitTuple d q u →
    |iteratedFDeriv ℝ q (Kinf d) 0 u| ≤ 15

/-- The squared Euclidean length `|n|^2 = sum_i n_i^2` of an integer point.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "sum_{n!=0} |n|^6 e^(-288|n|^2)"
Interface: none.
Does not claim: anything. -/
noncomputable def intNormSq (d : ℕ) (n : Fin d → ℤ) : ℝ :=
  ∑ i, ((n i : ℤ) : ℝ) ^ 2

/-- The lattice-image term `|n|^6 exp(-288 |n|^2)` for `n ≠ 0`, and `0` at `n = 0`
(`288 = 24^2/2`).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "sum_{n!=0} |n|^6 e^(-288|n|^2)"
Interface: none.
Does not claim: `24^2/2 = 288`, kernel-checked on main (`UniversalLaw.Side24.period_exponent`), not here. -/
noncomputable def imageTerm (d : ℕ) (n : Fin d → ℤ) : ℝ :=
  if n = 0 then 0 else intNormSq d n ^ 3 * Real.exp (-288 * intNormSq d n)

/-- Section 2 lattice bound: for `d ≤ 3`, the image family is summable and
`sum_{n ≠ 0} |n|^6 exp(-288|n|^2) ≤ 1458 exp(-288)`. Summability is stated so the `tsum` is
not the vacuous `0`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "<=1458 e^(-288)."
Interface: none.
Does not claim: the bound (statement only); the intermediate `729 sum_j j^9 e^(-288 j^2)` shell count is not specified; `27*27 = 729`, `729*2 = 1458` are kernel-checked on main (`UniversalLaw.Side24.lattice_shell_constant`), not here. -/
def LatticeImageSumBound : Prop :=
  ∀ d : ℕ, d ≤ 3 →
    Summable (fun n : Fin d → ℤ => imageTerm d n) ∧
    (∑' n : Fin d → ℤ, imageTerm d n) ≤ 1458 * Real.exp (-288)

/-- Section 2: successive terms `j^9 exp(-288 j^2)`, `j ≥ 1`, have ratio at most
`512 exp(-864)`, and that constant is below `1/2`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "successive terms have ratio at most512e^(-864)<1/2."
Interface: none.
Does not claim: the bounds (statement only); `2^9 = 512`, `288*3 = 864` are kernel-checked on main (`UniversalLaw.Side24.geometric_ratio_constants`), not here. -/
def SuccessiveTermRatio : Prop :=
  (∀ j : ℕ, 1 ≤ j →
    ((j : ℝ) + 1) ^ 9 * Real.exp (-288 * ((j : ℝ) + 1) ^ 2) ≤
      512 * Real.exp (-864) * ((j : ℝ) ^ 9 * Real.exp (-288 * (j : ℝ) ^ 2))) ∧
  512 * Real.exp (-864) < 1 / 2

/-- Section 2 exponential facts: `exp(288/125) > 10`, hence `exp(-288) < 10^(-125)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "hence e^(-288)<10^(-125) and the geometric-ratio inequality follows too."
Interface: none; `Real.exp` is Mathlib's.
Does not claim: the inequalities (statement only); the rational Taylor sum through degree 20 exceeding `10` is kernel-checked on main (the `UniversalLaw.Side24` Taylor-sum target in `Ledger.lean`), not here. -/
def ExponentialBounds : Prop :=
  10 < Real.exp (288 / 125) ∧ Real.exp (-288) < 1 / 10 ^ 125

/-- Display (2): for `d ∈ {2, 3}`, every unit-direction contraction through total order `6`,
mixed directions included, satisfies `|D^q K_24(0) - D^q K_infty(0)| < E`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "|D^q K_24(0)-D^q K_infty(0)|"
Interface: summability of the theta family is the premise `ThetaSummable d`; derivatives are Mathlib's `iteratedFDeriv ℝ q`.
Does not claim: the bound (statement only); differentiability of `K_24` (if `K_24` is not `C^q` the derivative is `0` in Lean); nothing here encloses the coefficient; nonauthor review or acceptance. -/
def ImageBound : Prop :=
  ∀ d : ℕ, d = 2 ∨ d = 3 → ThetaSummable d →
    ∀ q : ℕ, q ≤ 6 → ∀ u : Fin q → E d, IsUnitTuple d q u →
      |iteratedFDeriv ℝ q (K24 d) 0 u - iteratedFDeriv ℝ q (Kinf d) 0 u| < imageConstE

/-! ## Section 3: the joint covariance of `(G, t, svec H)` in a frame -/

/-- Index set of the `svec` coordinates: ordered pairs `(i, j)` with `i ≤ j`, of cardinality
`d(d+1)/2` (`3` for `d = 2`, `6` for `d = 3`).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Use the joint vector (G,t,svec H), where svec lists diagonal Hessian entries and"
Interface: none.
Does not claim: anything. -/
abbrev SymIndex (d : ℕ) : Type := {p : Fin d × Fin d // p.1 ≤ p.2}

/-- Index set of the joint vector `(G, t, svec H)`: `d` gradient coordinates, one third
derivative `t`, and `d(d+1)/2` `svec` coordinates; dimension `6` for `d = 2` and `10` for `d = 3`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "sqrt(2) times independent off-diagonal entries. Its dimension is at most10."
Interface: none.
Does not claim: the dimension count `6`, `10`, kernel-checked on main (`UniversalLaw.Side24.pin_determinant_and_joint_dimension`), not here. -/
abbrev JointIndex (d : ℕ) : Type := Fin d ⊕ Unit ⊕ SymIndex d

/-- The list of differentiation directions attached to a joint coordinate, in the orthonormal
frame `R` with `u = R 0`: `G_i = ∂_{R i} f`, `t = ∂_u^3 f`, `H_{ij} = ∂_{R i} ∂_{R j} f`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Use the joint vector (G,t,svec H), where svec lists diagonal Hessian entries and"
Interface: none.
Does not claim: anything. -/
def jetDirs (d : ℕ) (u : E d) (R : Fin d → E d) : JointIndex d → List (E d)
  | Sum.inl i => [R i]
  | Sum.inr (Sum.inl _) => [u, u, u]
  | Sum.inr (Sum.inr p) => [R p.1.1, R p.1.2]

/-- The `svec` scaling: `1` on gradient, `t` and diagonal Hessian coordinates, `sqrt 2` on
off-diagonal Hessian coordinates.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "sqrt(2) times independent off-diagonal entries. Its dimension is at most10."
Interface: none.
Does not claim: anything. -/
noncomputable def svecScale (d : ℕ) : JointIndex d → ℝ
  | Sum.inr (Sum.inr p) => if p.1.1 = p.1.2 then 1 else Real.sqrt 2
  | _ => 1

/-- The derivative of `K` at `x` contracted against a list of directions:
`D^{|l|} K(x)[l_1, …, l_{|l|}]`, through Mathlib's `iteratedFDeriv`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Each covariance entry differs from the reference by at most2E, by (2)."
Interface: none.
Does not claim: differentiability of `K` (if absent the derivative is `0` in Lean). -/
noncomputable def derivContraction (d : ℕ) (K : E d → ℝ) (x : E d) (l : List (E d)) : ℝ :=
  iteratedFDeriv ℝ l.length K x (fun i => l.get i)

/-- The joint covariance matrix of `(G, t, svec H)` at a point of a centred stationary field
with covariance `K`, in the frame `R` with `u = R 0`, through the stationary rule
`Cov(∂^α f, ∂^β f) = (-1)^{|β|} ∂^{α+β} K(0)`, with the `svec` scaling. With `K = K_infty`
this is `C_ref`; with `K = K_24` it is `C_24`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Each covariance entry differs from the reference by at most2E, by (2)."
Interface: none; the covariance is DEFINED by the derivative rule, no field is constructed.
Does not claim: that this matrix is the covariance of any random vector (no field is constructed here); symmetry or positive semidefiniteness of the matrix. -/
noncomputable def jointCov (d : ℕ) (K : E d → ℝ) (u : E d) (R : Fin d → E d) :
    Matrix (JointIndex d) (JointIndex d) ℝ :=
  Matrix.of fun a b =>
    svecScale d a * svecScale d b * (-1 : ℝ) ^ (jetDirs d u R b).length *
      derivContraction d K 0 (jetDirs d u R a ++ jetDirs d u R b)

/-- The gradient covariance block `Cov(G_i, G_j) = -∂_{R i} ∂_{R j} K(0)` in the frame `R`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "covariance differentiation gives"
Interface: none.
Does not claim: that it equals the identity for `K_infty` (that is `ReferenceOddAndPinBlocks`). -/
noncomputable def gradCov (d : ℕ) (K : E d → ℝ) (R : Fin d → E d) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => - derivContraction d K 0 [R i, R j]

/-- The pin covariance block of `V = H u`, coordinates `V_i = ∂_u ∂_{R i} f`:
`Cov(V_i, V_j) = ∂_u ∂_{R i} ∂_u ∂_{R j} K(0)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Thus V=Hu has covariance diag(3,1,...,1), tau^2=Var(t|G=0)=6, and"
Interface: none.
Does not claim: that it equals `diag(3,1,...,1)` for `K_infty` (that is `ReferenceOddAndPinBlocks`). -/
noncomputable def pinCov (d : ℕ) (K : E d → ℝ) (u : E d) (R : Fin d → E d) :
    Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => derivContraction d K 0 [u, R i, u, R j]

/-- The covariance vector `Cov(t, G_j) = -∂_u^3 ∂_{R j} K(0)` of `t = ∂_u^3 f` with the gradient.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Var(t)=15,  Cov(t,G)=(-3,0,...,0),"
Interface: none.
Does not claim: its reference value `(-3, 0, …, 0)` (that is `ReferenceOddAndPinBlocks`). -/
noncomputable def thirdGradCov (d : ℕ) (K : E d → ℝ) (u : E d) (R : Fin d → E d) : Fin d → ℝ :=
  fun j => - derivContraction d K 0 [u, u, u, R j]

/-- The variance `Var(t) = -∂_u^6 K(0)` of `t = ∂_u^3 f`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Var(t)=15,  Cov(t,G)=(-3,0,...,0),"
Interface: none.
Does not claim: its reference value `15` (that is `ReferenceOddAndPinBlocks`). -/
noncomputable def thirdVar (d : ℕ) (K : E d → ℝ) (u : E d) : ℝ :=
  - derivContraction d K 0 [u, u, u, u, u, u]

/-- The conditional variance `tau^2 = Var(t | G = 0)` as the Schur complement
`Var(t) - Cov(t,G) Cov(G)^(-1) Cov(G,t)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "tau^2=Var(t|G=0)=6"
Interface: none; the Gaussian conditional variance is written as the Schur complement (a standard identity, used here as the definition).
Does not claim: invertibility of `Cov(G)` (Mathlib's matrix inverse is `0` on a singular matrix); the value `6` (that is `ReferenceOddAndPinBlocks`). -/
noncomputable def condThirdVariance (d : ℕ) (K : E d → ℝ) (u : E d) (R : Fin d → E d) : ℝ :=
  thirdVar d K u -
    dotProduct (thirdGradCov d K u R) (Matrix.mulVec (gradCov d K R)⁻¹ (thirdGradCov d K u R))

/-- The reference joint covariance `C_ref` (kernel `K_infty`) in the frame `R`, `u = R 0`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "(1-epsilon) C_ref <= C_24 <= (1+epsilon) C_ref,"
Interface: none.
Does not claim: its displayed block values (`ReferenceOddAndPinBlocks`), or that it is frame independent. -/
noncomputable def Cref (d : ℕ) (u : E d) (R : Fin d → E d) :
    Matrix (JointIndex d) (JointIndex d) ℝ :=
  jointCov d (Kinf d) u R

/-- The exact periodic joint covariance `C_24` (kernel `K_24`) in the frame `R`, `u = R 0`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "(1-epsilon) C_ref <= C_24 <= (1+epsilon) C_ref,"
Interface: summability of the theta family is a premise wherever this matrix is used.
Does not claim: rotational invariance (the source asserts none for `C_24`). -/
noncomputable def C24 (d : ℕ) (u : E d) (R : Fin d → E d) :
    Matrix (JointIndex d) (JointIndex d) ℝ :=
  jointCov d (K24 d) u R

/-- Display (3): for `d ∈ {2, 3}`, in every orthonormal frame `R` (with `u = R 0`),
`(1-epsilon) C_ref ≤ C_24 ≤ (1+epsilon) C_ref` in the positive-semidefinite order.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "The inequalities are in the positive-semidefinite ordering and hold in every"
Interface: summability of the theta family is the premise `ThetaSummable d`; the matrices are defined by the derivative rule of `jointCov`.
Does not claim: the ordering (statement only); the passage to marginal and conditional covariances (not specified); `60E < epsilon` is kernel-checked on main (`UniversalLaw.Side24.covariance_relative_bound`), not here; nothing here encloses the coefficient; nonauthor review or acceptance. -/
def CovarianceOrdering : Prop :=
  ∀ d : ℕ, d = 2 ∨ d = 3 → ThetaSummable d →
    ∀ (hd : 0 < d) (R : Fin d → E d), Orthonormal ℝ R →
      (C24 d (R ⟨0, hd⟩) R - (1 - epsilon) • Cref d (R ⟨0, hd⟩) R).PosSemidef ∧
      ((1 + epsilon) • Cref d (R ⟨0, hd⟩) R - C24 d (R ⟨0, hd⟩) R).PosSemidef

/-- Section 3: each entry of `C_24` differs from the corresponding entry of `C_ref` by at most
`2E`, in every orthonormal frame.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Each covariance entry differs from the reference by at most2E, by (2)."
Interface: summability of the theta family is the premise `ThetaSummable d`.
Does not claim: the bound (statement only); the spectral-norm bound `20E` (not specified); nonauthor review or acceptance. -/
def EntrywiseCovarianceBound : Prop :=
  ∀ d : ℕ, d = 2 ∨ d = 3 → ThetaSummable d →
    ∀ (hd : 0 < d) (R : Fin d → E d), Orthonormal ℝ R →
      ∀ a b : JointIndex d,
        |C24 d (R ⟨0, hd⟩) R a b - Cref d (R ⟨0, hd⟩) R a b| ≤ 2 * imageConstE

/-- Section 3: `C_ref ≥ I/3` in the positive-semidefinite order, in every orthonormal frame.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Thus C_ref >= I/3 and"
Interface: none.
Does not claim: the bound (statement only); the odd-block eigenvalues `8 ± sqrt 58` and the minors `2/3`, `7/9` are kernel-checked on main (`UniversalLaw.Side24.odd_block_eigenvalues`, `odd_block_shifted_minors`, `smaller_eigenvalue_exceeds_third`), not here. -/
def ReferenceCovarianceLowerBound : Prop :=
  ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ (hd : 0 < d) (R : Fin d → E d), Orthonormal ℝ R →
    (Cref d (R ⟨0, hd⟩) R - (1 / 3 : ℝ) • (1 : Matrix (JointIndex d) (JointIndex d) ℝ)).PosSemidef

/-- Section 1 reference blocks, in every orthonormal frame with `u = R 0`: `Cov(G) = I_d`,
`Var(t) = 15`, `Cov(t, G) = (-3, 0, …, 0)`, `Cov(V) = diag(3, 1, …, 1)` and `tau^2 = 6`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Var(t)=15,  Cov(t,G)=(-3,0,...,0),"
Interface: none; the blocks are the derivative-rule matrices of `K_infty`.
Does not claim: the identities (statement only); `15 - 9 = 6` is kernel-checked on main (`UniversalLaw.Side24.conditional_third_derivative_variance`), not here. -/
def ReferenceOddAndPinBlocks : Prop :=
  ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ (hd : 0 < d) (R : Fin d → E d), Orthonormal ℝ R →
    gradCov d (Kinf d) R = 1 ∧
    thirdVar d (Kinf d) (R ⟨0, hd⟩) = 15 ∧
    thirdGradCov d (Kinf d) (R ⟨0, hd⟩) R = (fun j => if j = ⟨0, hd⟩ then -3 else 0) ∧
    pinCov d (Kinf d) (R ⟨0, hd⟩) R = Matrix.diagonal (fun i => if i = ⟨0, hd⟩ then 3 else 1) ∧
    condThirdVariance d (Kinf d) (R ⟨0, hd⟩) R = 6

/-! ## Section 4: Gaussian density comparison -/

/-- The centred Gaussian density on `R^n` with covariance `C`:
`(2 pi)^(-n/2) det(C)^(-1/2) exp(-x^T C^(-1) x / 2)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Comparing Gaussian determinants"
Interface: none.
Does not claim: that this is a probability density when `C` is singular (the matrix inverse is then `0` in Lean). -/
noncomputable def gaussDensity (n : ℕ) (C : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) : ℝ :=
  (2 * Real.pi) ^ (-(n : ℝ) / 2) * C.det ^ (-(1 : ℝ) / 2) *
    Real.exp (-(dotProduct x (Matrix.mulVec C⁻¹ x)) / 2)

/-- The number `n = m(m+1)/2` of independent symmetric-matrix coordinates.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Let C be a centered covariance in n=m(m+1)/2 independent symmetric-matrix"
Interface: none.
Does not claim: anything. -/
def symCoordDim (m : ℕ) : ℕ := m * (m + 1) / 2

/-- Section 4 density comparison: if `(1-epsilon) C0 ≤ C ≤ (1+epsilon) C0` in the
positive-semidefinite order, with `C0` positive definite, then pointwise
`[(1-epsilon)/(1+epsilon)]^(n/2) phi_{(1-epsilon)C0} ≤ phi_C ≤ [(1+epsilon)/(1-epsilon)]^(n/2) phi_{(1+epsilon)C0}`,
`n = m(m+1)/2`, `epsilon = 10^(-108)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "[(1-epsilon)/(1+epsilon)]^(n/2) phi_((1-epsilon)C0)"
Interface: none; `C.PosDef` is stated as a premise although it follows from `C0.PosDef` and the ordering.
Does not claim: the comparison (statement only); the cone-moment consequence `(1 ± epsilon)^m D0` and the scaling `a^m` of `h(A) = det(A)^2 1{A<0}` (not specified); nothing here encloses the coefficient; nonauthor review or acceptance. -/
def DensityComparison : Prop :=
  ∀ (m : ℕ) (C0 C : Matrix (Fin (symCoordDim m)) (Fin (symCoordDim m)) ℝ),
    C0.PosDef → C.PosDef →
    (C - (1 - epsilon) • C0).PosSemidef → ((1 + epsilon) • C0 - C).PosSemidef →
    ∀ x : Fin (symCoordDim m) → ℝ,
      ((1 - epsilon) / (1 + epsilon)) ^ ((symCoordDim m : ℝ) / 2) *
          gaussDensity (symCoordDim m) ((1 - epsilon) • C0) x ≤
        gaussDensity (symCoordDim m) C x ∧
      gaussDensity (symCoordDim m) C x ≤
        ((1 + epsilon) / (1 - epsilon)) ^ ((symCoordDim m : ℝ) / 2) *
          gaussDensity (symCoordDim m) ((1 + epsilon) • C0) x

/-- The exponent `a = m + 2/3 + n/2` of Section 4, with `m = d - 1`, `n = m(m+1)/2`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "a=m+2/3+n/2,  b=d+n/2."
Interface: none.
Does not claim: the values `13/6`, `25/6`, kernel-checked on main (`UniversalLaw.Side24.exponent_values`), not here. -/
noncomputable def exponentA (d : ℕ) : ℝ :=
  ((d - 1 : ℕ) : ℝ) + 2 / 3 + (symCoordDim (d - 1) : ℝ) / 2

/-- The exponent `b = d + n/2` of Section 4, with `n = (d-1)d/2`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "a=m+2/3+n/2,  b=d+n/2."
Interface: none.
Does not claim: the inequalities `a+2b<14`, `2a+b<13`, kernel-checked on main (`UniversalLaw.Side24.exponent_bounds`), not here. -/
noncomputable def exponentB (d : ℕ) : ℝ :=
  (d : ℝ) + (symCoordDim (d - 1) : ℝ) / 2

/-- Section 4 ratio bounds: for `d ∈ {2, 3}`,
`1 - 13 epsilon ≤ (1-epsilon)^a / (1+epsilon)^b` and `(1+epsilon)^a / (1-epsilon)^b ≤ 1 + 28 epsilon`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "ratio between1-13epsilon and"
Interface: none.
Does not claim: the bounds (statement only); that the integrand ratio of the source equals these power quotients (not specified); `13 ≤ 32`, `28 ≤ 32` and `32*10^(-108) < 10^(-106)` are kernel-checked on main (`UniversalLaw.Side24.ratio_constant_ordering`, `density_comparison_below_reported`), not here. -/
def IntegrandRatioBounds : Prop :=
  ∀ d : ℕ, d = 2 ∨ d = 3 →
    1 - 13 * epsilon ≤ (1 - epsilon) ^ exponentA d / (1 + epsilon) ^ exponentB d ∧
    (1 + epsilon) ^ exponentA d / (1 - epsilon) ^ exponentB d ≤ 1 + 28 * epsilon

/-! ## Section 1: reference densities and the reference coefficient identity -/

/-- Section 1: in every orthonormal frame with `u = R 0`, the reference gradient and pin
densities at zero satisfy `p_G(0) p_V(0) = (2 pi)^(-d) / sqrt 3`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "p_G(0) p_V(0) = (2pi)^(-d) / sqrt(3)."
Interface: none; the densities are `gaussDensity` of the derivative-rule blocks of `K_infty`.
Does not claim: the identity (statement only); `det diag(3,1,…,1) = 3` is kernel-checked on main (`UniversalLaw.Side24.pin_determinant_and_joint_dimension`), not here. -/
def ReferencePinDensityProduct : Prop :=
  ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ (hd : 0 < d) (R : Fin d → E d), Orthonormal ℝ R →
    gaussDensity d (gradCov d (Kinf d) R) 0 *
        gaussDensity d (pinCov d (Kinf d) (R ⟨0, hd⟩) R) 0 =
      (2 * Real.pi) ^ (-(d : ℝ)) / Real.sqrt 3

/-- Ordinary sphere area `|S^(d-1)|`: `2 pi` for `d = 2`, `4 pi` for `d = 3`, `0` otherwise.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Substituting these moments and |S^1|=2pi, |S^2|=4pi into the exact parent coefficient"
Interface: none.
Does not claim: any value outside `d ∈ {2, 3}`. -/
noncomputable def sphereArea (d : ℕ) : ℝ :=
  if d = 2 then 2 * Real.pi else if d = 3 then 4 * Real.pi else 0

/-- Display (1) as an identity: the parent prefactor `Gamma(7/6) / [24^(1/3) sqrt pi]` times
`p_G(0) p_V(0) = (2 pi)^(-d) / sqrt 3`, times `tau^(4/3)` with `tau = sqrt 6`, times `D_{d-1}`,
times `|S^(d-1)|`, equals `c_{d,ref}` (using `6^(2/3)/24^(1/3) = (3/2)^(1/3)`).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "This eliminates BOTH the angular and cone integrals for the reference expression."
Interface: none.
Does not claim: the identity (statement only); that the left side is the parent formula (15.2) evaluated at the reference law (the angular integrand is constant by rotational invariance of `K_infty`, a fact not specified here); `6^2*2 = 3*24` is kernel-checked on main (`UniversalLaw.Side24.cube_root_simplification`), not here; nonauthor review or acceptance. -/
def ReferenceCoefficientIdentity : Prop :=
  ∀ d : ℕ, d = 2 ∨ d = 3 →
    Real.Gamma (7 / 6) / ((24 : ℝ) ^ ((1 : ℝ) / 3) * Real.sqrt Real.pi) *
        ((2 * Real.pi) ^ (-(d : ℝ)) / Real.sqrt 3) * (Real.sqrt 6) ^ ((4 : ℝ) / 3) *
        coneMomentRef (d - 1) * sphereArea d =
      cRef d

/-! ## Section 1: the cone moments as Gaussian expectations -/

/-- Interface for the Gaussian variables of the reference cone moments. GIVEN: a probability
space; the `m = 1` transverse entry `A` with law `N(0, 8/3)`; the `m = 2` coordinates
`s, x, y` (shared trace shift and traceless part) jointly independent with laws `N(0, 5/3)`,
`N(0, 1)`, `N(0, 1)`, stated as a product law. Existence of such variables is not asserted.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Var(s)=5/3 and Var(x)=Var(y)=1. The negative-definite cone is s<-R,"
Interface: every field is a given property; the laws are `ProbabilityTheory.gaussianReal` with the displayed variances.
Does not claim: that these laws are the conditional law `A | V = 0` of the reference field (the source derives them; here they are hypotheses); the derivation of `Var(s) = 5/3` from the `(2/3) delta delta + …` formula (kernel-checked arithmetic on main, `UniversalLaw.Side24.trace_and_traceless_variances`). -/
structure ConeMomentInterface where
  /-- The sample space. -/
  Ω : Type
  /-- Its sigma-algebra. -/
  [mΩ : MeasurableSpace Ω]
  /-- The probability measure. -/
  P : Measure Ω
  /-- `P` is a probability measure. -/
  [isProb : IsProbabilityMeasure P]
  /-- Section 1 clause "For m=1 the variance of A is8/3.": the scalar transverse Hessian. -/
  A : Ω → ℝ
  /-- Law of `A`: centred Gaussian of variance `8/3`. -/
  A_law : Measure.map A P = ProbabilityTheory.gaussianReal 0 (8 / 3)
  /-- Section 1 clause "write A=[[s+x,y],[y,s-x]]": the shared scalar shift `s`. -/
  s : Ω → ℝ
  /-- The traceless coordinate `x`. -/
  x : Ω → ℝ
  /-- The traceless coordinate `y`. -/
  y : Ω → ℝ
  /-- Section 1 clause "s,x,y are independent, Var(s)=5/3 and Var(x)=Var(y)=1": the joint law
  of `(s, x, y)` is the product of `N(0,5/3)`, `N(0,1)`, `N(0,1)`. -/
  sxy_law : Measure.map (fun ω => (s ω, x ω, y ω)) P =
    (ProbabilityTheory.gaussianReal 0 (5 / 3)).prod
      ((ProbabilityTheory.gaussianReal 0 1).prod (ProbabilityTheory.gaussianReal 0 1))

/-- The `m = 2` transverse Hessian `A = [[s+x, y], [y, s-x]]`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "For m=2 write A=[[s+x,y],[y,s-x]]. Then s,x,y are independent,"
Interface: none.
Does not claim: anything. -/
noncomputable def transverseMatrix (s x y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.of ![![s + x, y], ![y, s - x]]

/-- The negative-definite cone `A < 0` for the `m = 2` matrix, as positive definiteness of `-A`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "R=sqrt(x^2+y^2), and det A=s^2-R^2."
Interface: none.
Does not claim: the source's reduction of the cone to `s < -R` (not specified; the cone is taken by its definition). -/
def NegDefCone (q : ℝ × ℝ × ℝ) : Prop :=
  (-(transverseMatrix q.1 q.2.1 q.2.2)).PosDef

/-- The cone-moment integrand `h(A) = det(A)^2 1{A<0}` for the `m = 2` matrix.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "R=sqrt(x^2+y^2), and det A=s^2-R^2."
Interface: none.
Does not claim: the identity `det A = s^2 - R^2` (not specified separately). -/
noncomputable def coneIntegrand (q : ℝ × ℝ × ℝ) : ℝ :=
  Set.indicator {q : ℝ × ℝ × ℝ | NegDefCone q}
    (fun q => (transverseMatrix q.1 q.2.1 q.2.2).det ^ 2) q

/-- Section 1 identity `D_1 = E[A^2 1{A<0}] = 4/3` for `A ~ N(0, 8/3)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "D_1=E[A^2 1{A<0}]=4/3."
Interface: `I : ConeMomentInterface` is given; the law of `A` is its hypothesis field.
Does not claim: the identity (statement only); that `D_1` is the parent's cone moment `D_u` of (15.1) for the reference law; nothing here encloses the coefficient. -/
def ConeMomentD1Identity (I : ConeMomentInterface) : Prop :=
  ∫ ω, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) (I.A ω) ∂I.P = coneMomentD1

/-- Section 1 identity `D_2 = E[det(A)^2 1{A<0}] = (1/2)[25/3 - 20/3 + 8 - 8 sqrt(3/8)]` for
`A = [[s+x, y], [y, s-x]]` with the product Gaussian law.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "D_2 = (1/2)[25/3-20/3+8-8sqrt(3/8)]"
Interface: `I : ConeMomentInterface` is given; the joint law is its hypothesis field.
Does not claim: the identity (statement only); the Rayleigh integration route; that `D_2` is the parent's `D_u` for the reference law; nothing here encloses the coefficient; nonauthor review or acceptance. -/
def ConeMomentD2Identity (I : ConeMomentInterface) : Prop :=
  ∫ ω, coneIntegrand (I.s ω, I.x ω, I.y ω) ∂I.P =
    (1 / 2) * (25 / 3 - 20 / 3 + 8 - 8 * Real.sqrt (3 / 8))

/-- Section 1 algebra `(1/2)[25/3 - 20/3 + 8 - 8 sqrt(3/8)] = 29/6 - sqrt 6`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "        = 29/6-sqrt(6)."
Interface: none.
Does not claim: the identity (statement only); its rational part and `4 sqrt(3/8) = sqrt 6` are kernel-checked on main (`UniversalLaw.Side24.cone_moment_m2_algebra`), not here. -/
def ConeMomentD2Algebra : Prop :=
  (1 / 2 : ℝ) * (25 / 3 - 20 / 3 + 8 - 8 * Real.sqrt (3 / 8)) = coneMomentD2

/-- Section 1 moments of the trace shift `s ~ N(0, 5/3)`: `E s^2 = 5/3`, `E s^4 = 25/3`,
`E exp(-s^2/2) = sqrt(3/8)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "E s^2=5/3, E s^4=25/3 and"
Interface: `I : ConeMomentInterface` is given.
Does not claim: the identities (statement only); `3 (5/3)^2 = 25/3` and `1 + 5/3 = 8/3` are kernel-checked on main (`UniversalLaw.Side24.fourth_moment_of_trace`, `exponential_moment_of_trace`), not here. -/
def TraceMoments (I : ConeMomentInterface) : Prop :=
  ∫ ω, I.s ω ^ 2 ∂I.P = 5 / 3 ∧
  ∫ ω, I.s ω ^ 4 ∂I.P = 25 / 3 ∧
  ∫ ω, Real.exp (-(I.s ω ^ 2) / 2) ∂I.P = Real.sqrt (3 / 8)

/-- Section 1: `R^2 = x^2 + y^2` has density `exp(-z/2)/2` on `(0, ∞)`, written through its
distribution function `P(R^2 ≤ a) = ∫_0^a exp(-z/2)/2 dz` for `a ≥ 0`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "The variable R^2 has density e^(-z/2)/2."
Interface: `I : ConeMomentInterface` is given.
Does not claim: the law (statement only); measurability of the event (if the event is not measurable, `P` is its outer measure). -/
def RayleighSquareLaw (I : ConeMomentInterface) : Prop :=
  ∀ a : ℝ, 0 ≤ a →
    (I.P {ω | I.x ω ^ 2 + I.y ω ^ 2 ≤ a}).toReal = ∫ z in (0 : ℝ)..a, Real.exp (-z / 2) / 2

/-- Section 1 elementary integral: for `a ≥ 0`,
`∫_0^a (a-z)^2 exp(-z/2) dz/2 = a^2 - 4a + 8 - 8 exp(-a/2)`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "integral_0^a (a-z)^2 e^(-z/2) dz/2 = a^2-4a+8-8e^(-a/2)."
Interface: none; the integral is Mathlib's interval integral against Lebesgue measure.
Does not claim: the identity (statement only). -/
def ElementaryConeIntegral : Prop :=
  ∀ a : ℝ, 0 ≤ a →
    ∫ z in (0 : ℝ)..a, (a - z) ^ 2 * Real.exp (-z / 2) / 2 =
      a ^ 2 - 4 * a + 8 - 8 * Real.exp (-a / 2)

/-! ## The exact periodic coefficient as an interface, and the headline statements -/

/-- Interface for the exact periodic coefficient `c_{d,24}`, "precisely its equation (15.2)"
of the parent. GIVEN: the coefficient `c24`; the ordinary sphere-area measure on `S^(d-1)`;
the gradient density `p_G(0)`, the pin density `p_{V_u}(0)`, the conditional standard
deviation `tau_u` and the cone moment `D_u` of the parent's (15.1). Hypothesis fields: the
sphere measure is carried by the unit sphere with the areas `2 pi`, `4 pi`; the parent's
positivity clause "Finite-jet rank proves tau_u>0, positive densities p_G(0),p_Vu(0), and
D_u>0"; the densities and `tau_u^2` agree with the Gaussian density formula and the Schur
complement of the derivative-rule blocks of `K_24`; summability of the theta family; and
the parent's display (15.2)
`c_{d,L}= Gamma(7/6) / [24^(1/3) sqrt(pi)] * integral_{S^(d-1)} p_G(0) p_Vu(0) tau_u^(4/3) D_u d sigma(u)`
(source id `lifetime-parent`, Section 15). The cone moment `D_u` is NOT pinned to `K_24`
here: writing it needs the conditional Gaussian law `A_u | V_u = 0`, which this module does
not construct.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Let c_d,24 be precisely its equation (15.2), using ordinary sphere area, ordered"
Interface: every field is a given object or a hypothesis; nothing is derived; in particular `coneMoment` is characterised only by its docstring.
Does not claim: existence of an object satisfying this interface; that `c24` is the finite-bar asymptotic constant (that reading is conditional on the parent's results, which are not touched); that the parent's "ordered maximum/saddle roles and the full height/gradient pin Jacobian" are encoded beyond the prefactor of (15.2). -/
structure PeriodicCoefficientInterface where
  /-- The exact periodic coefficient `c_{d,24}`, as a function of the dimension. -/
  c24 : ℕ → ℝ
  /-- Scope clause "using ordinary sphere area": the surface measure `sigma` on `S^(d-1)`. -/
  sphereMeasure : (d : ℕ) → Measure (E d)
  /-- The sphere measure is carried by the unit sphere. -/
  sphereMeasure_support : ∀ d : ℕ, sphereMeasure d ((Metric.sphere (0 : E d) 1)ᶜ) = 0
  /-- Section 1 clause "|S^1|=2pi, |S^2|=4pi": total areas in dimensions `2` and `3`. -/
  sphereMeasure_area :
    (sphereMeasure 2 Set.univ).toReal = 2 * Real.pi ∧ (sphereMeasure 3 Set.univ).toReal = 4 * Real.pi
  /-- Parent (15.2) ingredient: the gradient density at zero `p_G(0)`. -/
  pG0 : ℕ → ℝ
  /-- Parent (15.2) ingredient: the pin density at zero `p_{V_u}(0)` for the direction `u`. -/
  pV0 : (d : ℕ) → E d → ℝ
  /-- Parent (15.1) ingredient: `tau_u`, with `tau_u^2 = Var(t_u | G = 0)`. -/
  tau : (d : ℕ) → E d → ℝ
  /-- Parent (15.1) ingredient: `D_u = E[(det A_u)^2 1{A_u<0} | V_u=0]`, the Gaussian cone
  expectation under the conditional law of the transverse Hessian; GIVEN, not constructed. -/
  coneMoment : (d : ℕ) → E d → ℝ
  /-- Parent clause: positive density `p_G(0)`. -/
  pG0_pos : ∀ d : ℕ, d = 2 ∨ d = 3 → 0 < pG0 d
  /-- Parent clause: positive density `p_{V_u}(0)` for every unit direction. -/
  pV0_pos : ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ u : E d, ‖u‖ = 1 → 0 < pV0 d u
  /-- Parent clause: `tau_u > 0` for every unit direction. -/
  tau_pos : ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ u : E d, ‖u‖ = 1 → 0 < tau d u
  /-- Parent clause: `D_u > 0` for every unit direction. -/
  coneMoment_pos : ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ u : E d, ‖u‖ = 1 → 0 < coneMoment d u
  /-- The theta family of `K_24` is summable. -/
  theta_summable : ∀ d : ℕ, d = 2 ∨ d = 3 → ThetaSummable d
  /-- `p_G(0)` is the centred Gaussian density at zero with the gradient block of `K_24`, in
  every orthonormal frame. -/
  pG0_def : ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ R : Fin d → E d, Orthonormal ℝ R →
    pG0 d = gaussDensity d (gradCov d (K24 d) R) 0
  /-- `p_{V_u}(0)` is the centred Gaussian density at zero with the pin block of `K_24`, in
  every orthonormal frame with `u = R 0`. -/
  pV0_def : ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ (hd : 0 < d) (R : Fin d → E d), Orthonormal ℝ R →
    pV0 d (R ⟨0, hd⟩) = gaussDensity d (pinCov d (K24 d) (R ⟨0, hd⟩) R) 0
  /-- `tau_u^2` is the Schur complement `Var(t_u) - Cov(t_u,G) Cov(G)^(-1) Cov(G,t_u)` of the
  derivative-rule blocks of `K_24`, in every orthonormal frame with `u = R 0`. -/
  tau_sq : ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ (hd : 0 < d) (R : Fin d → E d), Orthonormal ℝ R →
    tau d (R ⟨0, hd⟩) ^ 2 = condThirdVariance d (K24 d) (R ⟨0, hd⟩) R
  /-- Parent display (15.2), for `d ∈ {2, 3}`. -/
  eq_15_2 : ∀ d : ℕ, d = 2 ∨ d = 3 →
    c24 d = Real.Gamma (7 / 6) / ((24 : ℝ) ^ ((1 : ℝ) / 3) * Real.sqrt Real.pi) *
      ∫ u, pG0 d * pV0 d u * tau d u ^ ((4 : ℝ) / 3) * coneMoment d u ∂(sphereMeasure d)

/-- The headline statement of the source: the two enclosures
`0.07340691930603427103 < c_{2,24} < 0.07340691930603427104` and
`0.04177593184059834334 < c_{3,24} < 0.04177593184059834335`, for any coefficient satisfying
the interface.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "0.07340691930603427103 < c_2,24 < 0.07340691930603427104,"
Interface: `I : PeriodicCoefficientInterface` is given; its `c24` is pinned only by the hypothesis fields of the interface (parent (15.2) with a GIVEN cone moment). The interface does not determine `c24`: the Prop is a predicate on the interface object, not a closed statement; its universal closure over all inhabitants is false whenever the interface is inhabited (scaling `coneMoment` by a positive constant scales `c24` through `eq_15_2` and gives another inhabitant), and `specified` refers to the statement shape at the source's object, which is not constructed.
Does not claim: the enclosures (statement only: nothing here encloses the coefficient); that an object satisfying the interface exists or is the coefficient of the source; the parent's results or the persistence interpretation; any finite-radius error estimate, unrestricted remainder or RN/24-jet statement; the arithmetic skeleton (endpoint adjacency, `d3 < d2`, endpoints in `(0, 1/10)`) is kernel-checked on main (31 `UniversalLaw.Side24` targets at origin/main f6deeba7), not here; nonauthor review or acceptance. -/
def CoefficientEnclosure (I : PeriodicCoefficientInterface) : Prop :=
  d2Lower < I.c24 2 ∧ I.c24 2 < d2Upper ∧ d3Lower < I.c24 3 ∧ I.c24 3 < d3Upper

/-- Display (4): `|c_{d,24} / c_{d,ref} - 1| < 10^(-106)` for `d = 2, 3`, for any coefficient
satisfying the interface.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "|c_d,24/c_d,ref -1| < 10^(-106), d=2,3."
Interface: `I : PeriodicCoefficientInterface` is given; `c_{d,ref}` is the explicit `cRef`. The interface does not determine `c24`: the Prop is a predicate on the interface object, not a closed statement; its universal closure over all inhabitants is false whenever the interface is inhabited (scaling `coneMoment` by a positive constant scales `c24` through `eq_15_2` and gives another inhabitant), and `specified` refers to the statement shape at the source's object, which is not constructed.
Does not claim: the bound (statement only); equality of the periodic and reference constants (the source says it is NOT claimed); `32*10^(-108) < 10^(-106)` is kernel-checked on main (`UniversalLaw.Side24.density_comparison_below_reported`), not here; nothing here encloses the coefficient. -/
def ReferenceComparison (I : PeriodicCoefficientInterface) : Prop :=
  ∀ d : ℕ, d = 2 ∨ d = 3 → |I.c24 d / cRef d - 1| < 1 / 10 ^ 106

end UniversalLaw.Spec.Side24
