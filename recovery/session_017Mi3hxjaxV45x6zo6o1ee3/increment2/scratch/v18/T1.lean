import Mathlib
open Set

noncomputable def h1ToyF (p : ℝ × ℝ) : ℝ := p.1 ^ 3 / 3 - p.1 / 4 - 20 * p.2 ^ 2

example (p : ℝ × ℝ) : HasFDerivAt h1ToyF ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
  have hf : HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p := hasFDerivAt_fst
  have hs : HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p := hasFDerivAt_snd
  have h1 := HasFDerivAt.sub (HasFDerivAt.sub (HasFDerivAt.const_mul (HasFDerivAt.pow hf 3) (1 / 3))
    (HasFDerivAt.const_mul hf (1 / 4))) (HasFDerivAt.const_mul (HasFDerivAt.pow hs 2) 20)
  convert h1 using 1
  · funext q
    simp only [h1ToyF, Pi.sub_apply]
    ring
  · ext <;> simp <;> ring

example (p : ℝ × ℝ) : HasStrictFDerivAt (fun q : ℝ × ℝ => ((q.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ).comp (ContinuousLinearMap.inr ℝ ℝ ℝ))
      (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
  convert (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L
      ContinuousLinearMap.snd ℝ ℝ ℝ).hasStrictFDerivAt (x := p) using 1
  funext q
  ext
  simp

example (p : ℝ × ℝ) : HasFDerivAt (fun y => ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * y) • ContinuousLinearMap.snd ℝ ℝ ℝ).comp (ContinuousLinearMap.inr ℝ ℝ ℝ))
      ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) p.2 := by
  convert ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ).hasFDerivAt (x := p.2) using 1
  funext y
  ext
  simp
