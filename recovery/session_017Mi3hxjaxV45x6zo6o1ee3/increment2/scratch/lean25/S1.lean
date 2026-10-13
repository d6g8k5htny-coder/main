import Mathlib

open Set Filter Topology

namespace CapFirstExit

section PartialBlock

variable {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]

theorem iteratedFDeriv_three_swap₁₂ {f : F → ℝ} {p : F} (hf : ContDiffAt ℝ 3 f p) (a b c : F) :
    iteratedFDeriv ℝ 3 f p ![a, b, c] = iteratedFDeriv ℝ 3 f p ![b, a, c] := by
  have h2 : ContDiffAt ℝ 2 (fderiv ℝ f) p := hf.fderiv_right (by norm_num)
  have hs : IsSymmSndFDerivAt ℝ (fderiv ℝ f) p := h2.isSymmSndFDerivAt (by simp)
  have hr : ∀ x y : F, iteratedFDeriv ℝ 3 f p ![x, y, c] =
      iteratedFDeriv ℝ 2 (fderiv ℝ f) p ![x, y] c := by
    intro x y
    rw [iteratedFDeriv_succ_apply_right]
    have e : Fin.init ![x, y, c] = ![x, y] := by ext i; fin_cases i <;> rfl
    rw [e]
    rfl
  rw [hr, hr, IsSymmSndFDerivAt.iteratedFDeriv_cons (hf := hs)]

theorem iteratedFDeriv_three_swap₂₃ {f : F → ℝ} {p : F} (hf : ContDiffAt ℝ 3 f p) (a b c : F) :
    iteratedFDeriv ℝ 3 f p ![a, b, c] = iteratedFDeriv ℝ 3 f p ![a, c, b] := by
  -- `D²f` is symmetric near `p`, so its derivative at `p` is symmetric in the last two slots.
  have hev : (fun q => iteratedFDeriv ℝ 2 f q ![b, c]) =ᶠ[𝓝 p]
      (fun q => iteratedFDeriv ℝ 2 f q ![c, b]) := by
    filter_upwards [hf.eventually (by simp)] with q hq
    have hq2 : ContDiffAt ℝ 2 f q := hq.of_le (by norm_num)
    exact IsSymmSndFDerivAt.iteratedFDeriv_cons (hf := hq2.isSymmSndFDerivAt (by simp))
  have hd : DifferentiableAt ℝ (iteratedFDeriv ℝ 2 f) p :=
    (hf.iteratedFDeriv_right (i := 2) (m := 1) (by norm_num)).differentiableAt (by norm_num)
  have key : ∀ v : Fin 2 → F, fderiv ℝ (fun q => iteratedFDeriv ℝ 2 f q v) p a =
      fderiv ℝ (iteratedFDeriv ℝ 2 f) p a v := by
    intro v
    have := ((ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => F) ℝ v).hasFDerivAt.comp p
      hd.hasFDerivAt).fderiv
    rw [show (fun q => iteratedFDeriv ℝ 2 f q v) =
      (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => F) ℝ v) ∘ iteratedFDeriv ℝ 2 f from rfl,
      this]
    rfl
  rw [iteratedFDeriv_succ_apply_left, iteratedFDeriv_succ_apply_left]
  have t1 : Fin.tail ![a, b, c] = ![b, c] := by ext i; fin_cases i <;> rfl
  have t2 : Fin.tail ![a, c, b] = ![c, b] := by ext i; fin_cases i <;> rfl
  simp only [Matrix.cons_val_zero, t1, t2]
  rw [← key, ← key, hev.fderiv_eq]

end PartialBlock

#print axioms block_of_partial
#print axioms iteratedFDeriv_three_swap₂₃
end CapFirstExit
