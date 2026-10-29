import RegularSuperlevel

/-!
Intentionally ill-typed interface control; not a theorem or a counterexample.

After the actual imported module compiles, the invocation below must be rejected:
it supplies the chart and source-membership proof but omits the final coordinate
identity on the chart source. The resulting function cannot inhabit the requested
neighborhood-basis proposition. Import or syntax failure does not satisfy this
control. This draft has not been executed.
-/

open Set Filter Topology
open UniversalLaw.Spec.Lifetime

example {d : ℕ} {g : E d → ℝ} {F : Type*}
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    (a : ℝ) (x : {y : E d // a ≤ g y})
    (p : OpenPartialHomeomorph (E d) (ℝ × F))
    (hx : x.val ∈ p.source) :
    (𝓝 x).HasBasis
      (fun V : Set {y : E d // a ≤ g y} =>
        V ∈ 𝓝 x ∧ IsPathConnected V) id := by
  exact TorusRegularSuperlevel.closed_superlevel_basis_of_chart a x p hx
