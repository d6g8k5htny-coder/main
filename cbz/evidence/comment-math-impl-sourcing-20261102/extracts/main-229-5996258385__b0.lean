/- SPEC ONLY — ASSIGN-20261005-P02-LM-008-C
   UNCOMPILED. No toolchain invoked. Not production Lean. -/
import Mathlib.Analysis.Asymptotics.Asymptotics
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open Filter Asymptotics Real

def SqrtDomBound (p q : ℝ → ℝ) (C : ℝ) (r0 : ℝ) : Prop :=
  (0 ≤ C) ∧ (0 < r0) ∧
  (∀ r, 0 < r → r ≤ r0 → 0 ≤ p r) ∧
  (∀ r, 0 < r → r ≤ r0 → 0 ≤ q r) ∧
  (∀ r, 0 < r → r ≤ r0 → p r ≤ C * Real.sqrt (q r))

def SuperPolySmall (f : ℝ → ℝ) : Prop :=
  ∀ m : ℝ, 0 < m →
    (fun r => f r) =o[nhdsWithin (0 : ℝ) (Set.Ioi 0)] (fun r => r ^ m)

theorem superpoly_of_sqrt_bound
    (p q : ℝ → ℝ) (C r0 : ℝ)
    (hBound : SqrtDomBound p q C r0) (hq : SuperPolySmall q) :
    SuperPolySmall p := by
  -- C=0: p≡0; C>0: use hq at m=2k with η=(ε/C)^2
  sorry

theorem consequence_C_of_P02_LM_008
    (P Q : ℝ → ℝ) (C_W c_Z r0 : ℝ)
    (hC_W : 0 ≤ C_W) (hc_Z : 0 < c_Z) (hr0 : 0 < r0)
    (htransfer : ∀ r, 0 < r → r ≤ r0 →
      0 ≤ P r ∧ 0 ≤ Q r ∧
      P r ≤ (Real.sqrt C_W / c_Z) * Real.sqrt (Q r))
    (hQ : SuperPolySmall Q) :
    SuperPolySmall P := by
  -- reduce to superpoly_of_sqrt_bound with C := √C_W / c_Z
  sorry
