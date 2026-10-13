import Mathlib
import Specifications.Side24

/-!
Feasibility probe (scratch only, not a packet file): can the Side24 successor interface of the
design note (main#307 6136538069 §3) be stated against the pinned Mathlib `d13f23b7`?
Defines the transverse symmetric-matrix coordinates, the conditional covariance of the transverse
Hessian given the pin block (Schur complement), the cone expectation under
`ProbabilityTheory.multivariateGaussian`, the successor structure with `coneMoment_def`, and the two
successor Props. `#check`s only; no theorem, no sorry.
-/

namespace Probe.Side24B

open UniversalLaw.Spec.Side24 MeasureTheory

/-- Transverse coordinates: frame indices other than `0` (the axial direction `u = R 0`). -/
abbrev TransRow (d : ℕ) : Type := {i : Fin d // 0 < i.val}

/-- Transverse symmetric-matrix coordinates: `svec` indices with both entries transverse. -/
abbrev TransIndex (d : ℕ) : Type := {p : SymIndex d // 0 < p.1.1.val}

/-- `svec` scaling on transverse coordinates: `1` on the diagonal, `sqrt 2` off it. -/
noncomputable def transScale (d : ℕ) (a : TransIndex d) : ℝ :=
  if a.1.1.1 = a.1.1.2 then 1 else Real.sqrt 2

/-- Covariance of the transverse Hessian entries (stationary rule, even order: sign `+`). -/
noncomputable def transCov (d : ℕ) (K : E d → ℝ) (R : Fin d → E d) :
    Matrix (TransIndex d) (TransIndex d) ℝ :=
  Matrix.of fun a b => transScale d a * transScale d b *
    derivContraction d K 0 [R a.1.1.1, R a.1.1.2, R b.1.1.1, R b.1.1.2]

/-- Cross-covariance of the transverse Hessian entries with the pin block `V_i = ∂_u ∂_{R i} f`. -/
noncomputable def transPinCov (d : ℕ) (K : E d → ℝ) (u : E d) (R : Fin d → E d) :
    Matrix (TransIndex d) (Fin d) ℝ :=
  Matrix.of fun a i => transScale d a * derivContraction d K 0 [R a.1.1.1, R a.1.1.2, u, R i]

/-- Conditional covariance of the transverse Hessian given `V_u = 0`: the Schur complement
`Σ_TT - Σ_TV Σ_VV⁻¹ Σ_VT` (the same pattern as `condThirdVariance`, with `pinCov` as `Σ_VV`). -/
noncomputable def condHessianCov (d : ℕ) (K : E d → ℝ) (u : E d) (R : Fin d → E d) :
    Matrix (TransIndex d) (TransIndex d) ℝ :=
  transCov d K R - transPinCov d K u R * (pinCov d K u R)⁻¹ * (transPinCov d K u R).transpose

/-- The symmetric transverse matrix read off an `svec` vector (undoing the `sqrt 2` scaling). -/
noncomputable def symOf (d : ℕ) (v : EuclideanSpace ℝ (TransIndex d)) :
    Matrix (TransRow d) (TransRow d) ℝ :=
  Matrix.of fun i j =>
    if h : i.1 ≤ j.1 then v ⟨⟨(i.1, j.1), h⟩, i.2⟩ / (if i = j then 1 else Real.sqrt 2)
    else v ⟨⟨(j.1, i.1), le_of_not_ge h⟩, j.2⟩ / Real.sqrt 2

/-- The cone integrand `h(A) = det(A)^2 1{A < 0}` on `svec` coordinates. -/
noncomputable def coneIntegrandTrans (d : ℕ) (v : EuclideanSpace ℝ (TransIndex d)) : ℝ :=
  Set.indicator {v : EuclideanSpace ℝ (TransIndex d) | (-(symOf d v)).PosDef}
    (fun v => (symOf d v).det ^ 2) v

/-- The cone moment `D_u = E[(det A_u)^2 1{A_u < 0} | V_u = 0]` as an integral against the
centred multivariate Gaussian with the conditional covariance. -/
noncomputable def coneMomentOf (d : ℕ) (K : E d → ℝ) (u : E d) (R : Fin d → E d) : ℝ :=
  ∫ v, coneIntegrandTrans d v ∂(ProbabilityTheory.multivariateGaussian 0 (condHessianCov d K u R))

/-- Successor interface: the present fields plus the definition of the cone moment. -/
structure PeriodicCoefficientInterface2 extends PeriodicCoefficientInterface where
  /-- `D_u` is the Gaussian cone expectation under the conditional law of the transverse
  Hessian of `K_24` given `V_u = 0`, in every orthonormal frame with `u = R 0`. -/
  coneMoment_def : ∀ d : ℕ, d = 2 ∨ d = 3 → ∀ (hd : 0 < d) (R : Fin d → E d), Orthonormal ℝ R →
    coneMoment d (R ⟨0, hd⟩) = coneMomentOf d (K24 d) (R ⟨0, hd⟩) R

/-- Successor of `CoefficientEnclosure`, over the determined interface. -/
def CoefficientEnclosure2 (I : PeriodicCoefficientInterface2) : Prop :=
  CoefficientEnclosure I.toPeriodicCoefficientInterface

/-- Successor of `ReferenceComparison`, over the determined interface. -/
def ReferenceComparison2 (I : PeriodicCoefficientInterface2) : Prop :=
  ReferenceComparison I.toPeriodicCoefficientInterface

#check @condHessianCov
#check @coneMomentOf
#check @PeriodicCoefficientInterface2.coneMoment_def
#check @CoefficientEnclosure2
#check @ReferenceComparison2
-- the pinned Mathlib facts the design note leans on
#check @ProbabilityTheory.multivariateGaussian
#check @ProbabilityTheory.multivariateGaussian_of_not_posSemidef
#check @ProbabilityTheory.isGaussian_multivariateGaussian
#check @ProbabilityTheory.covarianceBilin_multivariateGaussian

end Probe.Side24B
