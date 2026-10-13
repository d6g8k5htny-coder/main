import TorusFiltration
open TorusFiltration

-- Lattice membership of the literal Field lattice vector (torus_lattice_mem_iff).
example (d : ℕ) (L : ℝ) (hL : 0 < L) (n : Fin d → ℤ) :
    torus_latticeVec d L n ∈ torus_periodLattice d L hL :=
  (torus_lattice_mem_iff d L hL _).2 ⟨n, rfl⟩

-- A concrete lattice translate of the origin has the same quotient image (torus_control_lattice_translate).
example : torus_proj 2 1 one_pos (0 + torus_latticeVec 2 1 ![3, -5]) = torus_proj 2 1 one_pos 0 :=
  torus_control_lattice_translate 2 1 one_pos 0 ![3, -5]

-- Constants are periodic and descend to themselves (torus_descend_proj).
example (d : ℕ) (L : ℝ) (hL : 0 < L) (c : ℝ) (x : torus_E d) :
    torus_descend d L hL (fun _ => c) (fun _ _ => rfl) (torus_proj d L hL x) = c :=
  torus_descend_proj d L hL (fun _ => c) (fun _ _ => rfl) x

-- The path-component map on a representative (superlevel_componentMap_mk).
example {X Y : Type} [TopologicalSpace X] [TopologicalSpace Y] (f : C(X, Y)) (x : X) :
    superlevel_componentMap f (ZerothHomotopy.mk x) = ZerothHomotopy.mk (f x) :=
  superlevel_componentMap_mk f x

-- Closed-superlevel monotonicity at concrete parameters (superlevel_mono).
example : superlevel_set (fun x : ℝ => x) (-2) ⊆ superlevel_set (fun x : ℝ => x) 1 :=
  superlevel_mono (fun x : ℝ => x) (by norm_num)

-- Closed endpoint membership for the identity height: 2 ≤ 3, so 3 ∈ F(-2).
example : (3 : ℝ) ∈ superlevel_set (fun x : ℝ => x) (-2) := by
  norm_num [superlevel_set]
