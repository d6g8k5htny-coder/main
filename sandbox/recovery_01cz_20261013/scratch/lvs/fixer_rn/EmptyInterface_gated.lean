import Mathlib.Probability.ConditionalProbability
import Mathlib.MeasureTheory.Measure.WithDensity
import Mathlib.MeasureTheory.Measure.Haar.OfBasis
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Normed.Lp.MeasurableSpace
import Mathlib.LinearAlgebra.Dimension.Finrank

/-! NON-CERTIFYING verifier scratch. The structure below is `PinnedLawInterface` of
Specifications/RN.lean copied verbatim except for the three calculus stand-ins the author
also used (hessian body, IsCriticalPoint body, `smooth` field type); none of the three is
used by the argument. Goal: a kernel check that no term of `PinnedLawInterface d L Ω`
exists when `0 < L`. -/

namespace VerifyRN
open MeasureTheory

abbrev E (d : ℕ) : Type := EuclideanSpace ℝ (Fin d)

noncomputable def latticeVec (d : ℕ) (L : ℝ) (n : Fin d → ℤ) : E d :=
  ∑ i, EuclideanSpace.single i (L * (n i : ℝ))

noncomputable def hessian (d : ℕ) (g : E d → ℝ) (x v w : E d) : ℝ :=
  g x * g v * g w

def IsCriticalPoint (d : ℕ) (g : E d → ℝ) (x : E d) : Prop :=
  g x = 0

def HessianHasInertia (d : ℕ) (g : E d → ℝ) (x : E d) (j : ℕ) : Prop :=
  ∃ V W : Submodule ℝ (E d), IsCompl V W ∧
    Module.finrank ℝ V = j ∧ Module.finrank ℝ W = d - j ∧
    (∀ v ∈ V, v ≠ 0 → hessian d g x v v < 0) ∧
    (∀ w ∈ W, w ≠ 0 → 0 < hessian d g x w w)

noncomputable def pinM (d : ℕ) (r : ℝ) (u : E d) : E d :=
  -((r / 2) • u)

noncomputable def pinS (d : ℕ) (r : ℝ) (u : E d) : E d :=
  (r / 2) • u

structure PinnedLawInterface (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω] where
  F : Ω → E d → ℝ
  smooth : ∀ (_ω : Ω) (n : ℕ), n = n
  periodic : ∀ (ω : Ω) (x : E d) (n : Fin d → ℤ), F ω (x + latticeVec d L n) = F ω x
  Q : ℝ → ℝ → ℝ → OrthonormalBasis (Fin d) ℝ (E d) → Measure Ω
  Q_prob : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), IsProbabilityMeasure (Q r b k R)
  Q_pinned : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ᵐ ω ∂(Q r b k R),
      F ω (pinM d r (R 0)) = b ∧ F ω (pinS d r (R 0)) = b - k * r ^ 3 ∧
      IsCriticalPoint d (F ω) (pinM d r (R 0)) ∧ IsCriticalPoint d (F ω) (pinS d r (R 0))
  W : ℝ → ℝ → ℝ → OrthonormalBasis (Fin d) ℝ (E d) → Ω → ℝ
  W_nonneg : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (ω : Ω), 0 ≤ W r b k R ω
  W_typed : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (ω : Ω), W r b k R ω ≠ 0 →
    HessianHasInertia d (F ω) (pinM d r (R 0)) d ∧ HessianHasInertia d (F ω) (pinS d r (R 0)) (d - 1)
  Z_pos : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    0 < ∫ ω, W r b k R ω ∂(Q r b k R)
  QW : ℝ → ℝ → ℝ → OrthonormalBasis (Fin d) ℝ (E d) → Measure Ω
  QW_eq : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    QW r b k R = (Q r b k R).withDensity
      (fun ω => ENNReal.ofReal (W r b k R ω / ∫ ω', W r b k R ω' ∂(Q r b k R)))
  QW_prob : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), IsProbabilityMeasure (QW r b k R)
  Λ : ℕ → ℝ → ℝ → E d → E d → ℝ

theorem latticeVec_single (d : ℕ) [NeZero d] (L : ℝ) :
    latticeVec d L (Pi.single (0 : Fin d) (1 : ℤ)) = EuclideanSpace.single (0 : Fin d) L := by
  classical
  unfold latticeVec
  rw [Finset.sum_eq_single (0 : Fin d)]
  · simp
  · intro i _ hi
    ext j
    simp [EuclideanSpace.single_apply, Pi.single_apply, hi]
  · intro h; exact absurd (Finset.mem_univ _) h

theorem pinS_eq_pinM_add (d : ℕ) [NeZero d] (L : ℝ) :
    pinS d L ((EuclideanSpace.basisFun (Fin d) ℝ) 0) =
      pinM d L ((EuclideanSpace.basisFun (Fin d) ℝ) 0) +
        latticeVec d L (Pi.single (0 : Fin d) (1 : ℤ)) := by
  classical
  rw [latticeVec_single, EuclideanSpace.basisFun_apply]
  unfold pinS pinM
  ext j
  by_cases hj : j = 0
  · subst hj; simp [EuclideanSpace.single_apply]; ring
  · simp [EuclideanSpace.single_apply, hj]

theorem pinnedLawInterface_empty (d : ℕ) [NeZero d] (L : ℝ) (hL : 0 < L)
    (Ω : Type) [MeasurableSpace Ω] (I : PinnedLawInterface d L Ω) : False := by
  classical
  have hprob : IsProbabilityMeasure (I.Q L 0 1 (EuclideanSpace.basisFun (Fin d) ℝ)) :=
    I.Q_prob L 0 1 _
  have hae := I.Q_pinned L 0 1 (EuclideanSpace.basisFun (Fin d) ℝ) hL
  have hfalse : ∀ᵐ ω ∂(I.Q L 0 1 (EuclideanSpace.basisFun (Fin d) ℝ)), False := by
    filter_upwards [hae] with ω hω
    obtain ⟨hM, hS, -, -⟩ := hω
    have hper := I.periodic ω (pinM d L ((EuclideanSpace.basisFun (Fin d) ℝ) 0))
      (Pi.single (0 : Fin d) (1 : ℤ))
    rw [← pinS_eq_pinM_add d L, hS, hM] at hper
    have h3 : (0 : ℝ) < L ^ 3 := pow_pos hL 3
    linarith
  have hzero : I.Q L 0 1 (EuclideanSpace.basisFun (Fin d) ℝ) = 0 :=
    ae_eq_bot.mp (Filter.eventually_false_iff_eq_bot.mp hfalse)
  have h1 : (I.Q L 0 1 (EuclideanSpace.basisFun (Fin d) ℝ)) Set.univ = 1 := measure_univ
  rw [hzero] at h1
  simp at h1

end VerifyRN
