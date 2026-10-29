import RegularSuperlevel
import RegularPointChart

/-!
Draft literal regular-point consumers of the separately owned scalar chart.

Released dependency: Research Task Watch's RegularPointChart.lean at Math-
commit 72edcf5b93be45c29f2d10a26f7d8e861a48f528, 2,767 bytes, SHA-256
5568c8d6e03f7064cd5924ebe2814ae746100505ae31789d5c975902528f9fa2.
That source was frozen and released by its owner. This C297 consumer has not
yet been compiled or kernel-checked.
It neither reimplements the chart nor modifies the topology candidate under replay.

The public conclusions are actual path-connected neighborhood bases in the
closed-superlevel subtype. The critical-boundary case and quotient transport
remain outside these statements.
-/

open Set Filter Topology
open UniversalLaw.Spec.Lifetime

namespace TorusRegularSuperlevel

/-- At a literal noncritical point, the closed superlevel has a neighborhood
basis of path-connected sets. The supplied smoothness and noncriticality
construct the chart through the separately owned regular-point module. -/
theorem closed_superlevel_regular_basis
    {d : ℕ} {g : E d → ℝ}
    (hsmooth : IsSmooth d g)
    (a : ℝ) (x : {y : E d // a ≤ g y})
    (hregular : ¬ IsCriticalPoint d g x.val) :
    (𝓝 x).HasBasis
      (fun V : Set {y : E d // a ≤ g y} =>
        V ∈ 𝓝 x ∧ IsPathConnected V) id := by
  exact closed_superlevel_basis_of_chart a x
    (regularPointChart hsmooth hregular)
    (mem_regularPointChart_source hsmooth hregular)
    (fun y _ => regularPointChart_fst hsmooth hregular y)

/-- The pointwise basis required for eventual global LPC holds at every
strict-interior or literal regular point of the closed superlevel.
No chart or LPC hypothesis appears in this public interface. -/
theorem closed_superlevel_path_connected_basis
    {d : ℕ} {g : E d → ℝ}
    (hsmooth : IsSmooth d g)
    (a : ℝ) (x : {y : E d // a ≤ g y})
    (hx : a < g x.val ∨ ¬ IsCriticalPoint d g x.val) :
    (𝓝 x).HasBasis
      (fun V : Set {y : E d // a ≤ g y} =>
        V ∈ 𝓝 x ∧ IsPathConnected V) id := by
  rcases hx with hinterior | hregular
  · exact closed_superlevel_interior_basis (hsmooth 0).continuous a x hinterior
  · exact closed_superlevel_regular_basis hsmooth a x hregular

end TorusRegularSuperlevel
