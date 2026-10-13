import Mathlib

open Set Filter
open scoped Topology

namespace Probe

variable {α 𝕜 E F : Type*} [NontriviallyNormedField 𝕜] [IsRCLikeNormedField 𝕜]
  [NormedAddCommGroup E] [NormedSpace 𝕜 E] [NormedAddCommGroup F] [CompleteSpace F]
  [NormedSpace 𝕜 F]

/-- Local (open preconnected set) version of `iteratedFDeriv_tsum`: summable bounds on the
successive derivatives are only required on `U`. -/
theorem iteratedFDeriv_tsum_of_isOpen {f : α → E → F} {v : ℕ → α → ℝ} {N : ℕ∞} {U : Set E}
    (hU : IsOpen U) (hU' : IsPreconnected U)
    (hf : ∀ i, ContDiff 𝕜 N (f i))
    (hv : ∀ k : ℕ, (k : ℕ∞) ≤ N → Summable (v k))
    (h'f : ∀ (k : ℕ) (i : α) (x : E), (k : ℕ∞) ≤ N → x ∈ U → ‖iteratedFDeriv 𝕜 k (f i) x‖ ≤ v k i)
    {k : ℕ} (hk : (k : ℕ∞) ≤ N) {x : E} (hx : x ∈ U) :
    iteratedFDeriv 𝕜 k (fun y => ∑' n, f n y) x = ∑' n, iteratedFDeriv 𝕜 k (f n) x := by
  induction k generalizing x with
  | zero =>
    simp_rw [iteratedFDeriv_zero_eq_comp]
    exact (continuousMultilinearCurryFin0 𝕜 E F).symm.toContinuousLinearEquiv.map_tsum
  | succ k IH =>
    have h'k : (k : ℕ∞) < N := lt_of_lt_of_le (WithTop.coe_lt_coe.2 (Nat.lt_succ_self _)) hk
    have hEq : (iteratedFDeriv 𝕜 k fun y => ∑' n, f n y) =ᶠ[𝓝 x]
        fun y => ∑' n, iteratedFDeriv 𝕜 k (f n) y :=
      Filter.eventually_of_mem (hU.mem_nhds hx) fun y hy => IH h'k.le hy
    have hD : HasFDerivAt (fun y => ∑' n, iteratedFDeriv 𝕜 k (f n) y)
        (∑' n, fderiv 𝕜 (iteratedFDeriv 𝕜 k (f n)) x) x := by
      refine hasFDerivAt_tsum_of_isPreconnected (hv (k + 1) hk) hU hU' (fun n y _ => ?_)
        (fun n y hy => ?_) hx ?_ hx
      · exact ((hf n).differentiable_iteratedFDeriv (mod_cast h'k) y).hasFDerivAt
      · rw [norm_fderiv_iteratedFDeriv]
        exact h'f (k + 1) n y hk hy
      · exact .of_norm_bounded (hv k h'k.le) fun n => h'f k n x h'k.le hx
    rw [iteratedFDeriv_succ_eq_comp_left, Function.comp_apply, hEq.fderiv_eq, hD.fderiv]
    simp_rw [iteratedFDeriv_succ_eq_comp_left, Function.comp_apply]
    exact (continuousMultilinearCurryLeftEquiv 𝕜
      (fun _ : Fin (k + 1) => E) F).symm.toContinuousLinearEquiv.map_tsum

end Probe

#print axioms Probe.iteratedFDeriv_tsum_of_isOpen
