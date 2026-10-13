import CapFirstExit

open Set Filter Topology

namespace CapFirstExit

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

-- stand-ins for the Slice section (already compiled separately)
axiom slice_hasFDerivAt' {f : ℝ × E → ℝ} {x : ℝ} {y : E} {D : ℝ × E →L[ℝ] ℝ}
    (hf : HasFDerivAt f D (x, y)) :
    HasFDerivAt (fun y' => f (x, y')) (D.comp (ContinuousLinearMap.inr ℝ ℝ E)) y
axiom ridge_hasDerivAt' {f : ℝ × E → ℝ} {h : ℝ → E} {x : ℝ} {D : ℝ × E →L[ℝ] ℝ} {h' : E}
    (hf : HasFDerivAt f D (x, h x)) (hh : HasDerivAt h h' x)
    (hy : D.comp (ContinuousLinearMap.inr ℝ ℝ E) = 0) :
    HasDerivAt (fun t => f (t, h t)) (D (1, 0)) x

theorem joint_inputs {f : ℝ × E → ℝ} {h h' : ℝ → E} {D : ℝ → ℝ × E →L[ℝ] ℝ} {r : ℝ} (hr : 0 < r)
    (hD : ∀ x ∈ Icc (-2 * r) (2 * r), HasFDerivAt f (D x) (x, h x))
    (hh' : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt h (h' x) x)
    (hridge : ∀ x ∈ Icc (-2 * r) (2 * r), (D x).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0) :
    ContinuousOn h (Icc (-r / 2) (2 * r)) ∧
      (∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (D x (1, 0)) x) ∧
      ∀ x ∈ Icc (-2 * r) (r / 2), HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x) := by
  refine ⟨fun x hx => ?_, fun x hx => ridge_hasDerivAt' (hD x hx) (hh' x hx) (hridge x hx),
    fun x hx => ?_⟩
  · exact (hh' x ⟨by linarith [hx.1], hx.2⟩).continuousAt.continuousWithinAt
  · have hx' : x ∈ Icc (-2 * r) (2 * r) := ⟨hx.1, by linarith [hx.2]⟩
    simpa [hridge x hx'] using slice_hasFDerivAt' (hD x hx')

theorem ridgeToy_joint (x : ℝ) :
    HasFDerivAt ridgeToyF ((x ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ) (x, 0) ∧
      ((x ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ).comp (ContinuousLinearMap.inr ℝ ℝ ℝ) = 0 ∧
      ((x ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ) (1, 0) = x ^ 2 - 1 / 4 := by
  refine ⟨?_, by ext; simp, by simp⟩
  have hu : HasDerivAt (fun t : ℝ => t ^ 3 / 3 - t / 4) (x ^ 2 - 1 / 4) x := by
    have h1 := ((hasDerivAt_pow 3 x).div_const 3).sub ((hasDerivAt_id x).div_const 4)
    convert h1 using 1
    · funext t; simp
    · push_cast; ring
  have hv : HasDerivAt (fun t : ℝ => t ^ 2) 0 (0 : ℝ) := by
    simpa using hasDerivAt_pow 2 (0 : ℝ)
  have h1 := hu.comp_hasFDerivAt ((x, (0 : ℝ)) : ℝ × ℝ) (hasFDerivAt_fst (𝕜 := ℝ))
  have h2 := hv.comp_hasFDerivAt ((x, (0 : ℝ)) : ℝ × ℝ) (hasFDerivAt_snd (𝕜 := ℝ))
  convert h1.sub h2 using 1
  · funext p; simp [ridgeToyF]
  · simp

end CapFirstExit
