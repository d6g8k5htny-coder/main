import Mathlib.Topology.Instances.Matrix
import Mathlib.Topology.Algebra.GroupWithZero
import Mathlib.Topology.Algebra.Ring.Real
import Mathlib.Topology.Order.LeftRightNhds
import Mathlib.Analysis.Normed.Group.Real
import Mathlib.Analysis.Normed.Group.Constructions
import Mathlib.Analysis.Normed.Group.Continuity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FunProp

/-!
# Regression parameter limits and uniform small-radius bounds

Only the limiting pin covariance is assumed invertible. Current inverses,
regression means/covariances and their uniform bounds are consequences of the
input block limits. No target covariance inverse or zero-mean premise occurs.
See SOURCE_REGRESSION_LIMITS.md for the frozen source contract.
-/

noncomputable section
namespace I4GaussianRegressionLimits
open Matrix Filter
open scoped Topology
set_option autoImplicit false

/-- The actual regression mean for a centered prior and target d. -/
def regressionMean {p t : Type*} [Fintype p] [DecidableEq p]
    (G : Matrix p p ℝ) (C : Matrix t p ℝ) (d : p → ℝ) : t → ℝ :=
  (C * G⁻¹) *ᵥ d

/-- The Schur covariance; no inverse of this target block is taken. -/
def regressionCovariance {p t : Type*} [Fintype p] [DecidableEq p]
    (G : Matrix p p ℝ) (C : Matrix t p ℝ) (B : Matrix t t ℝ) : Matrix t t ℝ :=
  B - C * G⁻¹ * C.transpose

private theorem tendsto_matrix_mul_explicit {α p t q : Type*} [Fintype t]
    {u : Filter α} {A : α → Matrix p t ℝ} {A0 : Matrix p t ℝ}
    {B : α → Matrix t q ℝ} {B0 : Matrix t q ℝ}
    (hA : Tendsto A u (𝓝 A0)) (hB : Tendsto B u (𝓝 B0)) :
    Tendsto (fun r => A r * B r) u (𝓝 (A0 * B0)) := by
  have hAij (i : p) (j : t) : Tendsto (fun r => A r i j) u (𝓝 (A0 i j)) :=
    ((continuous_apply j).tendsto (A0 i)).comp
      (((continuous_apply i).tendsto A0).comp hA)
  have hBij (i : t) (j : q) : Tendsto (fun r => B r i j) u (𝓝 (B0 i j)) :=
    ((continuous_apply j).tendsto (B0 i)).comp
      (((continuous_apply i).tendsto B0).comp hB)
  apply tendsto_pi_nhds.mpr
  intro i
  apply tendsto_pi_nhds.mpr
  intro j
  exact tendsto_finsetSum Finset.univ (fun k _ => (hAij i k).mul (hBij k j))

private theorem tendsto_matrix_mulVec_explicit {α p t : Type*} [Fintype t]
    {u : Filter α} {A : α → Matrix p t ℝ} {A0 : Matrix p t ℝ}
    {b : α → (t → ℝ)} {b0 : t → ℝ}
    (hA : Tendsto A u (𝓝 A0)) (hb : Tendsto b u (𝓝 b0)) :
    Tendsto (fun r => A r *ᵥ b r) u (𝓝 (A0 *ᵥ b0)) := by
  have hAij (i : p) (j : t) : Tendsto (fun r => A r i j) u (𝓝 (A0 i j)) :=
    ((continuous_apply j).tendsto (A0 i)).comp
      (((continuous_apply i).tendsto A0).comp hA)
  have hbi (i : t) : Tendsto (fun r => b r i) u (𝓝 (b0 i)) :=
    ((continuous_apply i).tendsto b0).comp hb
  apply tendsto_pi_nhds.mpr
  intro i
  exact tendsto_finsetSum Finset.univ (fun j _ => (hAij i j).mul (hbi j))

/-- Continuous pin/target blocks give current invertibility and convergence
of the inverse, noncentered regression mean and possibly singular covariance. -/
theorem regression_parameter_limits
    {p t a : Type*} [Fintype p] [DecidableEq p] [Fintype t]
    (u : Filter a)
    (G : a → Matrix p p ℝ) (C : a → Matrix t p ℝ)
    (B : a → Matrix t t ℝ) (d : a → p → ℝ)
    (G0 : Matrix p p ℝ) (C0 : Matrix t p ℝ) (B0 : Matrix t t ℝ) (d0 : p → ℝ)
    (hG : Tendsto G u (𝓝 G0)) (hC : Tendsto C u (𝓝 C0))
    (hB : Tendsto B u (𝓝 B0)) (hd : Tendsto d u (𝓝 d0)) (hG0 : IsUnit G0) :
    (∀ᶠ r in u, IsUnit (G r)) ∧
      Tendsto (fun r => (G r)⁻¹) u (𝓝 G0⁻¹) ∧
      Tendsto (fun r => regressionMean (G r) (C r) (d r)) u
        (𝓝 (regressionMean G0 C0 d0)) ∧
      Tendsto (fun r => regressionCovariance (G r) (C r) (B r)) u
        (𝓝 (regressionCovariance G0 C0 B0)) := by
  have hdet0 : G0.det ≠ 0 := ((Matrix.isUnit_iff_isUnit_det G0).mp hG0).ne_zero
  have hdet : Tendsto (fun r => (G r).det) u (𝓝 G0.det) :=
    (continuous_id.matrix_det.tendsto G0).comp hG
  have hunit : ∀ᶠ r in u, IsUnit (G r) := by
    filter_upwards [hdet.eventually_ne hdet0] with r hr
    exact (Matrix.isUnit_iff_isUnit_det (G r)).mpr (isUnit_iff_ne_zero.mpr hr)
  have hInv : Tendsto (fun r => (G r)⁻¹) u (𝓝 G0⁻¹) := by
    have hc : ContinuousAt (Ring.inverse : ℝ → ℝ) G0.det := by
      simpa only [Ring.inverse_eq_inv'] using (continuousAt_inv₀ hdet0)
    exact (continuousAt_matrix_inv G0 hc).tendsto.comp hG
  have hK : Tendsto (fun r => C r * (G r)⁻¹) u (𝓝 (C0 * G0⁻¹)) :=
    tendsto_matrix_mul_explicit hC hInv
  have hm : Tendsto (fun r => regressionMean (G r) (C r) (d r)) u
      (𝓝 (regressionMean G0 C0 d0)) :=
    tendsto_matrix_mulVec_explicit hK hd
  have htrans : Tendsto (fun r => (C r).transpose) u (𝓝 C0.transpose) := by
    have hc : Continuous (fun D : Matrix t p ℝ => D.transpose) := by fun_prop
    exact (hc.tendsto C0).comp hC
  have hcov := hB.sub (tendsto_matrix_mul_explicit hK htrans)
  exact ⟨hunit, hInv, hm, hcov⟩

private theorem bounds_from_limits {ι : Type*} [Fintype ι]
    (f : ℝ → (ι → ℝ)) (g : ℝ → Matrix ι ι ℝ)
    (f0 : ι → ℝ) (g0 : Matrix ι ι ℝ)
    (hf : Tendsto f (𝓝[>] (0 : ℝ)) (𝓝 f0))
    (hg : Tendsto g (𝓝[>] (0 : ℝ)) (𝓝 g0)) :
    ∃ B V δ : ℝ, 0 ≤ B ∧ 0 ≤ V ∧ 0 < δ ∧
      ∀ r : ℝ, 0 < r → r < δ →
        (∀ i, |f r i| ≤ B) ∧ (∀ i, |g r i i| ≤ V) := by
  have hfb : ∀ᶠ r in 𝓝[>] (0 : ℝ), ‖f r‖ < ‖f0‖ + 1 :=
    hf.norm.eventually (gt_mem_nhds (by linarith))
  have hgg : Tendsto (fun r => fun i j => g r i j) (𝓝[>] (0 : ℝ))
      (𝓝 (fun i j => g0 i j)) := hg
  have hgb : ∀ᶠ r in 𝓝[>] (0 : ℝ), ‖fun i j => g r i j‖ < ‖fun i j => g0 i j‖ + 1 :=
    hgg.norm.eventually (gt_mem_nhds (by linarith))
  obtain ⟨δ, hδ, hbounds⟩ := mem_nhdsGT_iff_exists_Ioo_subset.mp (hfb.and hgb)
  refine ⟨‖f0‖+1, ‖fun i j => g0 i j‖+1, δ, by positivity, by positivity, hδ, ?_⟩
  intro r hr hrδ
  have hb := hbounds ⟨hr, hrδ⟩
  constructor
  · intro i
    exact (show |f r i| ≤ ‖f r‖ by simpa only [Real.norm_eq_abs] using norm_le_pi_norm (f r) i).trans hb.1.le
  · intro i
    exact (show |g r i i| ≤ ‖fun i j => g r i j‖ by
      exact (show |g r i i| ≤ ‖g r i‖ by simpa only [Real.norm_eq_abs] using norm_le_pi_norm (g r i) i).trans
        (norm_le_pi_norm (fun i j => g r i j) i)).trans hb.2.le


/-- One small-radius interval supplies current invertibility and simultaneous
mean/diagonal-covariance bounds, derived solely from the block limits. -/
theorem regression_uniform_bounds
    {p t : Type*} [Fintype p] [DecidableEq p] [Fintype t]
    (G : ℝ → Matrix p p ℝ) (C : ℝ → Matrix t p ℝ)
    (B : ℝ → Matrix t t ℝ) (d : ℝ → p → ℝ)
    (G0 : Matrix p p ℝ) (C0 : Matrix t p ℝ) (B0 : Matrix t t ℝ) (d0 : p → ℝ)
    (hG : Tendsto G (𝓝[>] (0 : ℝ)) (𝓝 G0))
    (hC : Tendsto C (𝓝[>] (0 : ℝ)) (𝓝 C0))
    (hB : Tendsto B (𝓝[>] (0 : ℝ)) (𝓝 B0))
    (hd : Tendsto d (𝓝[>] (0 : ℝ)) (𝓝 d0)) (hG0 : IsUnit G0) :
    ∃ M V δ : ℝ, 0 ≤ M ∧ 0 ≤ V ∧ 0 < δ ∧
      ∀ r : ℝ, 0 < r → r < δ → IsUnit (G r) ∧
        (∀ i, |regressionMean (G r) (C r) (d r) i| ≤ M) ∧
        (∀ i, |regressionCovariance (G r) (C r) (B r) i i| ≤ V) := by
  obtain ⟨hunit, _hInv, hm, hcov⟩ := regression_parameter_limits
    (𝓝[>] (0 : ℝ)) G C B d G0 C0 B0 d0 hG hC hB hd hG0
  obtain ⟨M, V, δB, hM, hV, hδB, hb⟩ := bounds_from_limits
    (fun r => regressionMean (G r) (C r) (d r))
    (fun r => regressionCovariance (G r) (C r) (B r))
    (regressionMean G0 C0 d0) (regressionCovariance G0 C0 B0) hm hcov
  obtain ⟨δU, hδU, hu⟩ := mem_nhdsGT_iff_exists_Ioo_subset.mp hunit
  refine ⟨M, V, min δB δU, hM, hV, lt_min hδB hδU, ?_⟩
  intro r hr hrδ
  exact ⟨hu ⟨hr, lt_of_lt_of_le hrδ (min_le_right _ _)⟩,
    hb r hr (lt_of_lt_of_le hrδ (min_le_left _ _))⟩

end I4GaussianRegressionLimits
