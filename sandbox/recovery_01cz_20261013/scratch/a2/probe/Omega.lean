import Specifications.Side24
open UniversalLaw.Spec.Side24
open scoped ContDiff
example (d : ℕ) : ContDiff ℝ ω (Kinf d) := by
  unfold Kinf
  exact Real.contDiff_exp.comp ((contDiff_norm_sq ℝ).neg.div_const 2)
example (d n : ℕ) (y : E d) (v : Fin n → E d) (σ : Equiv.Perm (Fin n)) :
    iteratedFDeriv ℝ n (Kinf d) y (v ∘ σ) = iteratedFDeriv ℝ n (Kinf d) y v := by
  have h : ContDiff ℝ ω (Kinf d) := by
    unfold Kinf; exact Real.contDiff_exp.comp ((contDiff_norm_sq ℝ).neg.div_const 2)
  exact h.contDiffAt.iteratedFDeriv_comp_perm v σ
