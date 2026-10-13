
/-! ## Round 7 (draft under HOLD, not a packet change): the successor interface that pins the cone moment

Everything above this line is byte-identical to the module at Math- `8acc4856`. The declarations below
transcribe the parent's (15.1) `D_u = E[(det A_u)^2 1{A_u<0} | V_u=0]` (source id `lifetime-parent`,
Section 15) for the formal kernel `K_24`: the conditional law of the transverse Hessian block `A_u`
given the pin `V_u = 0` is written, for a centred Gaussian vector, through the Schur complement of the
derivative-rule blocks (the SIDE24 note's Section 3 reading of conditioning), and the cone expectation
is the Lebesgue integral of `det(A)^2 1{A<0}` against Mathlib's `multivariateGaussian` with that
covariance. The successor `PeriodicCoefficientInterface2` keeps every field of
`PeriodicCoefficientInterface` and adds the parent's nondegeneracy clause for the pin block and the
conditional block and the definition of `coneMoment`; the two successor Props restate the headline
displays over it. Formal progress of every declaration here: `specified`. Nothing here is proved,
nothing above changes, no label moves, and no inhabitant of either interface is constructed.

What the successor does NOT pin (recorded for the design review): `sphereMeasure` stays a given measure
constrained only by its support and total area, so `c24` is determined by the fields only up to that
choice; frame invariance of `coneMomentOf` under transverse-frame changes (the parent asserts it) is
not a field, so the frame-quantified `coneMoment_def` is a consistency burden on the interface rather
than a theorem here. -/

/-- Transverse coordinates of the frame: the indices `i ≠ 0` (the axial direction is `u = R 0`), the
`m = d - 1` directions of `u`-perp.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Writing m=d-1 and A for the transverse Hessian, Gaussian conditioning on V=0 gives"
Interface: none; a subtype of `Fin d`.
Does not claim: anything. -/
abbrev TransRow (d : ℕ) : Type := {i : Fin d // 0 < i.val}

/-- Transverse `svec` coordinates: the `svec` indices `(i, j)`, `i ≤ j`, with both entries transverse
(`0 < i`), of cardinality `m(m+1)/2` (`1` for `d = 2`, `3` for `d = 3`).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Use the joint vector (G,t,svec H), where svec lists diagonal Hessian entries and"
Interface: none; a subtype of `SymIndex d`.
Does not claim: the cardinalities (not specified). -/
abbrev TransIndex (d : ℕ) : Type := {p : SymIndex d // 0 < p.1.1.val}

/-- The `svec` scaling restricted to transverse coordinates: `1` on the diagonal, `sqrt 2` off it
(the restriction of `svecScale` to the Hessian block).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "sqrt(2) times independent off-diagonal entries. Its dimension is at most10."
Interface: none.
Does not claim: anything. -/
noncomputable def transScale (d : ℕ) (a : TransIndex d) : ℝ :=
  if a.1.1.1 = a.1.1.2 then 1 else Real.sqrt 2

/-- The covariance block of the transverse Hessian entries `A_ij = ∂_{R i} ∂_{R j} f`, `i, j ≠ 0`, by the
stationary rule `Cov(∂^α f, ∂^β f) = (-1)^{|β|} ∂^{α+β} K(0)` at even total order (sign `+`), with the
`svec` scaling: the restriction of `jointCov` to the transverse `svec` coordinates.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Cov(H_ij,H_kl)=delta_ij delta_kl+delta_ik delta_jl+delta_il delta_jk,"
Interface: none; the matrix is DEFINED by the derivative rule, no field is constructed.
Does not claim: that this matrix is the covariance of any random vector; its reference value (the display is for `K_infty`); symmetry or positive semidefiniteness. -/
noncomputable def transCov (d : ℕ) (K : E d → ℝ) (R : Fin d → E d) :
    Matrix (TransIndex d) (TransIndex d) ℝ :=
  Matrix.of fun a b => transScale d a * transScale d b *
    derivContraction d K 0 [R a.1.1.1, R a.1.1.2, R b.1.1.1, R b.1.1.2]

/-- The cross-covariance block of the transverse Hessian entries with the pin coordinates
`V_i = ∂_u ∂_{R i} f`: `Cov(A_ab, V_i) = ∂_{R a} ∂_{R b} ∂_u ∂_{R i} K(0)`, with the `svec` scaling.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Thus V=Hu has covariance diag(3,1,...,1), tau^2=Var(t|G=0)=6, and"
Interface: none; defined by the derivative rule.
Does not claim: that this matrix is a covariance; its reference value. -/
noncomputable def transPinCov (d : ℕ) (K : E d → ℝ) (u : E d) (R : Fin d → E d) :
    Matrix (TransIndex d) (Fin d) ℝ :=
  Matrix.of fun a i => transScale d a * derivContraction d K 0 [R a.1.1.1, R a.1.1.2, u, R i]

/-- The conditional covariance of the transverse Hessian given `V_u = 0`, as the Schur complement
`Cov(A) - Cov(A,V) Cov(V)^(-1) Cov(V,A)` of the derivative-rule blocks (the same pattern as
`condThirdVariance`, with `pinCov` as `Cov(V)`): the Gaussian conditional covariance when `Cov(V)` is
nondegenerate.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "to their conditional covariances. For the latter, the Schur-complement quadratic"
Interface: none; the Gaussian conditional covariance is written as the Schur complement (a standard identity for a nondegenerate conditioning block, used here as the definition).
Does not claim: invertibility of `pinCov` (Mathlib's matrix inverse is `0` on a singular matrix, and the expression then degenerates to `transCov`; the successor interface carries nondegeneracy as the field `pinCov_posDef`); that it is the covariance of any conditional law (no field is constructed); the reference value `(2/3) delta delta + delta delta + delta delta` of Section 1. -/
noncomputable def condHessianCov (d : ℕ) (K : E d → ℝ) (u : E d) (R : Fin d → E d) :
    Matrix (TransIndex d) (TransIndex d) ℝ :=
  transCov d K R - transPinCov d K u R * (pinCov d K u R)⁻¹ * (transPinCov d K u R).transpose

/-- The symmetric transverse matrix read off an `svec` coordinate vector, undoing the `sqrt 2` scaling
of the off-diagonal entries: entry `(i, j)` is `v_(i,j)` on the diagonal and `v_(min,max) / sqrt 2` off it.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Let C be a centered covariance in n=m(m+1)/2 independent symmetric-matrix"
Interface: none.
Does not claim: anything. -/
noncomputable def symOf (d : ℕ) (v : EuclideanSpace ℝ (TransIndex d)) :
    Matrix (TransRow d) (TransRow d) ℝ :=
  Matrix.of fun i j =>
    if h : i.1 ≤ j.1 then v ⟨⟨(i.1, j.1), h⟩, i.2⟩ / (if i = j then 1 else Real.sqrt 2)
    else v ⟨⟨(j.1, i.1), le_of_not_ge h⟩, j.2⟩ / Real.sqrt 2

/-- The cone-moment integrand `h(A) = det(A)^2 1{A<0}` on `svec` coordinates, the negative-definite
cone taken as positive definiteness of `-A` (as in `NegDefCone`).

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "Integrate the nonnegative function h(A)=det(A)^2 1{A<0}. Under covariance scaling"
Interface: none.
Does not claim: measurability or integrability of the integrand (if it is not integrable, Mathlib's integral is `0`). -/
noncomputable def coneIntegrandTrans (d : ℕ) (v : EuclideanSpace ℝ (TransIndex d)) : ℝ :=
  Set.indicator {v : EuclideanSpace ℝ (TransIndex d) | (-(symOf d v)).PosDef}
    (fun v => (symOf d v).det ^ 2) v

/-- The Gaussian cone expectation of the parent's (15.1), `D_u = E[(det A_u)^2 1{A_u<0} | V_u=0]`,
written for a kernel `K` as the integral of the cone integrand against the centred multivariate
Gaussian law with the conditional covariance `condHessianCov d K u R`, in the frame `R` with `u = R 0`.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "D_u=E[(det A_u)^2 1{A_u<0} | V_u=0]."
Interface: none; Mathlib's `ProbabilityTheory.multivariateGaussian` is the law (it is the Dirac mass at `0` when the covariance is not positive semidefinite, which is why the successor interface carries `condHessianCov_posDef`).
Does not claim: that this number is the parent's `D_u` for any field (no field and no conditional law is constructed: the Gaussian law is written from the formal covariance matrix); invariance under transverse-frame changes (the parent asserts it; not stated here); its reference values `4/3`, `29/6 - sqrt 6` (those are `ConeMomentD1Identity`, `ConeMomentD2Identity` over `ConeMomentInterface`). -/
noncomputable def coneMomentOf (d : ℕ) (K : E d → ℝ) (u : E d) (R : Fin d → E d) : ℝ :=
  ∫ v, coneIntegrandTrans d v ∂(ProbabilityTheory.multivariateGaussian 0 (condHessianCov d K u R))

/-- Successor of `PeriodicCoefficientInterface` (numbered, the predecessor unchanged): every field of
the predecessor, plus the parent's Section 15 clause that `A_u` conditional on `V_u = 0` is a centred
NONDEGENERATE Gaussian symmetric matrix (as positive definiteness of the pin block and of the
conditional block of `K_24`, in every orthonormal frame with `u = R 0`), plus the definition of the
cone moment: `coneMoment d u` is the Gaussian cone expectation `coneMomentOf d (K24 d) u R`. With these
fields `coneMoment` is no longer free, so the doubling inhabitant that makes the universal closure of
`CoefficientEnclosure` and `ReferenceComparison` false is excluded.

Source: d6g8k5htny-coder/Math- commit c39164254ad1529ed49543743598d5dcb24bab98 path imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md sha256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7
Anchor: "Here A_u conditional on V_u=0 is an actual centered nondegenerate Gaussian symmetric matrix"
Interface: every field is a given object or a hypothesis; nothing is derived. Still GIVEN and not pinned beyond support and total area: `sphereMeasure`, so `c24` is determined by the fields only up to the choice of that measure among measures carried by the unit sphere with the displayed areas. Frame invariance of `coneMomentOf` is not a field: `coneMoment_def` is required in every orthonormal frame with `u = R 0`, so an interface inhabitant exists only if the formal `coneMomentOf` is frame invariant (the parent asserts this for the actual object; it is not shown here).
Does not claim: existence of an object satisfying this interface; that the formal Schur complement and `multivariateGaussian` are the conditional law of any field (no field is constructed); that `c24` is the finite-bar asymptotic constant (conditional on the parent's results, which are not touched); frame invariance; anything about the predecessor interface, which is unchanged. -/
structure PeriodicCoefficientInterface2 extends PeriodicCoefficientInterface where
  /-- Parent clause "an actual centered nondegenerate Gaussian symmetric matrix": the pin block
  `Cov(V_u)` of `K_24` is positive definite in every orthonormal frame with `u = R 0`, so the Schur
  complement `condHessianCov` is the Gaussian conditional covariance. -/
  pinCov_posDef : ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ (hd : 0 < d) (R : Fin d → E d), Orthonormal ℝ R →
    (pinCov d (K24 d) (R ⟨0, hd⟩) R).PosDef
  /-- The same clause for the conditional transverse block: positive definite, so the law in
  `coneMomentOf` is a nondegenerate Gaussian and not the Dirac case of `multivariateGaussian`. -/
  condHessianCov_posDef : ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ (hd : 0 < d) (R : Fin d → E d), Orthonormal ℝ R →
    (condHessianCov d (K24 d) (R ⟨0, hd⟩) R).PosDef
  /-- Parent (15.1): `D_u` is the Gaussian cone expectation under the conditional law of the
  transverse Hessian of `K_24` given `V_u = 0`, in every orthonormal frame with `u = R 0`. -/
  coneMoment_def : ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ (hd : 0 < d) (R : Fin d → E d), Orthonormal ℝ R →
    coneMoment d (R ⟨0, hd⟩) = coneMomentOf d (K24 d) (R ⟨0, hd⟩) R

/-- Successor of `CoefficientEnclosure` over the successor interface: the same two enclosures
`0.07340691930603427103 < c_{2,24} < 0.07340691930603427104` and
`0.04177593184059834334 < c_{3,24} < 0.04177593184059834335`, for any object satisfying
`PeriodicCoefficientInterface2`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "0.04177593184059834334 < c_3,24 < 0.04177593184059834335."
Interface: `I : PeriodicCoefficientInterface2` is given; `c24` is pinned by `eq_15_2` with the cone moment now defined by `coneMoment_def`, the densities by `pG0_def`, `pV0_def`, `tau_sq`, and the sphere measure still only by support and total area; the Prop is a predicate on the interface object, stated for every inhabitant.
Does not claim: the enclosures (statement only: nothing here encloses the coefficient; a theorem would need a proof-producing interval pipeline for `Gamma(7/6)`, `pi`, `exp`, the theta sums and the Gaussian cone integral); existence of an inhabitant; that the formal objects are the source's; anything about `CoefficientEnclosure`, which is unchanged; nonauthor review or acceptance. -/
def CoefficientEnclosure2 (I : PeriodicCoefficientInterface2) : Prop :=
  CoefficientEnclosure I.toPeriodicCoefficientInterface

/-- Successor of `ReferenceComparison` over the successor interface: display (4),
`|c_{d,24} / c_{d,ref} - 1| < 10^(-106)` for `d = 2, 3`, for any object satisfying
`PeriodicCoefficientInterface2`.

Source: d6g8k5htny-coder/Math- commit 760340e921ac4ceda296b8118da936f1133e956e path coefficients/side24_v1/PROOF.md sha256 c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769
Anchor: "The exact periodic constant is NOT claimed equal to the reference constant."
Interface: `I : PeriodicCoefficientInterface2` is given; `c_{d,ref}` is the explicit `cRef`; the cone moment is pinned by `coneMoment_def` and the sphere measure only by support and total area (the source's bounds are uniform in `u`, which is what would make the choice immaterial; not shown here).
Does not claim: the bound (statement only; a theorem would need the Section 4 comparison of the Gaussian cone expectation under the ordered covariances, the Section 3 passage of (3) to the conditional covariance, and the frame-uniform integrand ratio, over the proof packet's `CovarianceOrdering`, `ReferenceCovarianceLowerBound`, `DensityComparison` and `IntegrandRatioBounds`); equality of the periodic and reference constants (the source says it is NOT claimed); anything about `ReferenceComparison`, which is unchanged; nothing here encloses the coefficient. -/
def ReferenceComparison2 (I : PeriodicCoefficientInterface2) : Prop :=
  ReferenceComparison I.toPeriodicCoefficientInterface

