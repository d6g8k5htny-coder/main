import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Topology.Algebra.InfiniteSum.Defs
import Mathlib.Topology.Algebra.InfiniteSum.Real

/-! NON-CERTIFYING scratch: kernel check that the Prop `RareClusterPoissonObstructionO`
as written in Specifications/RN.lean (copied byte-for-byte below with its two helpers)
holds for EVERY input, because its hypothesis block is unsatisfiable. Verifier scratch only. -/

namespace VerifyRN

noncomputable def pmfTV (μ ν : ℕ → ℝ) : ℝ :=
  (1 / 2) * ∑' n : ℕ, |μ n - ν n|

noncomputable def poissonPMF (lam : ℝ) (n : ℕ) : ℝ :=
  Real.exp (-lam) * lam ^ n / (Nat.factorial n : ℝ)

def RareClusterPoissonObstructionO (μ : ℝ → ℕ → ℝ) (a A : ℝ) : Prop :=
  0 < a →
  (∃ r0 : ℝ, 0 < r0 ∧ ∀ r : ℝ, 0 < r → r ≤ r0 →
    (∀ n : ℕ, 0 ≤ μ r n) ∧ Summable (μ r) ∧ ∑' n : ℕ, μ r n = 1 ∧
    Summable (fun n : ℕ => (n : ℝ) * μ r n) ∧ ∑' n : ℕ, (n : ℝ) * μ r n ≤ A * r ^ 3 ∧
    a * r ^ 3 ≤ 1 - μ r 0 - μ r 1) →
  ∃ rStar : ℝ, 0 < rStar ∧ ∀ r : ℝ, 0 < r → r ≤ rStar →
    (∀ lam : ℝ, 0 ≤ lam → (a / 2) * r ^ 3 ≤ pmfTV (μ r) (poissonPMF lam)) ∧
    (∃ lam : ℝ, 0 ≤ lam ∧ pmfTV (μ r) (poissonPMF lam) ≤ A * r ^ 3)

theorem rareCluster_vacuous (μ : ℝ → ℕ → ℝ) (a A : ℝ) :
    RareClusterPoissonObstructionO μ a A := by
  intro ha _ h
  exfalso
  have hy : 0 < a⁻¹ := inv_pos.mpr ha
  have hr : 0 < 1 + a⁻¹ := by linarith
  obtain ⟨hnn, -, -, -, -, hlow⟩ := h (1 + a⁻¹) hr
  have h0 := hnn 0
  have h1 := hnn 1
  have hcube : 3 * a⁻¹ ≤ (1 + a⁻¹) ^ 3 := by nlinarith [sq_nonneg a⁻¹, pow_pos hy 3]
  have hbig : 3 ≤ a * (1 + a⁻¹) ^ 3 := by
    calc (3 : ℝ) = a * (3 * a⁻¹) := by field_simp
      _ ≤ a * (1 + a⁻¹) ^ 3 := by exact mul_le_mul_of_nonneg_left hcube ha.le
  linarith

end VerifyRN
