import RegularSuperlevelLiteral

/-!
Intentionally ill-typed interface control; not a theorem or a counterexample.

After the actual imported modules compile, the invocation below must be rejected:
it supplies all preceding arguments but omits the final regular-or-strict-interior
premise. The resulting function cannot inhabit the requested neighborhood-basis
proposition. Import or syntax failure does not satisfy this control.
This draft has not been executed.
-/

open Set Filter Topology
open UniversalLaw.Spec.Lifetime

example {d : ℕ} {g : E d → ℝ}
    (hsmooth : IsSmooth d g)
    (a : ℝ) (x : {y : E d // a ≤ g y}) :
    (𝓝 x).HasBasis
      (fun V : Set {y : E d // a ≤ g y} =>
        V ∈ 𝓝 x ∧ IsPathConnected V) id := by
  exact TorusRegularSuperlevel.closed_superlevel_path_connected_basis hsmooth a x
