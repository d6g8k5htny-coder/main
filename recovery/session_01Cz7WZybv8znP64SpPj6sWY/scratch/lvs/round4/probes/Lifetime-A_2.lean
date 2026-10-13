import Specifications.Lifetime

/-! Probe (Lifetime-A_2): kernel-checked degeneracy / falsity witnesses for the universal
closures of the Lifetime-A predicates. Probe file only. No `sorry`. -/

open UniversalLaw.Spec.Lifetime

-- IsLPeriodic with L = 0 holds for every g (the lattice vector is 0).
example (d : ℕ) (g : E d → ℝ) : IsLPeriodic d 0 g := by
  intro x n
  have h : ∀ i : Fin d, EuclideanSpace.single i ((0:ℝ) * (n i : ℝ)) = 0 := fun i => by
    simp
  simp [latticeVec, h]

-- The universal closure of IsUnrestrictedLeadingCoefficient is false whenever the interfaces are inhabited: c = 0.
example (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L) (I : SuperlevelBarInterface d L) :
    ¬ IsUnrestrictedLeadingCoefficient d L G I 0 := fun h => lt_irrefl 0 h.1

-- The universal closure of IsLifetimeDensityVersion is false whenever G is inhabited: ν ≡ -1.
example (d : ℕ) (L : ℝ) (G : GaussianFieldInterface d L) (Pair : (E d → ℝ) → E d → E d → Prop) :
    ¬ IsLifetimeDensityVersion d L G Pair (fun _ => -1) := fun h => by
  have := h.1 0
  norm_num at this

-- HessianHasInertia d g x j is false for j > d (a subspace of E d has finrank ≤ d).
example (d : ℕ) (g : E d → ℝ) (x : E d) : ¬ HessianHasInertia d g x (d + 1) := by
  rintro ⟨V, W, -, hV, -, -, -⟩
  have h1 : Module.finrank ℝ V ≤ Module.finrank ℝ (E d) := Submodule.finrank_le V
  rw [hV, finrank_euclideanSpace_fin] at h1
  omega

-- HessianHasInertia 0 g x 0 holds for every g, x (d = 0 degeneracy): V = W = ⊥ = ⊤.
example (g : E 0 → ℝ) (x : E 0) : HessianHasInertia 0 g x 0 := by
  refine ⟨⊥, ⊤, isCompl_bot_top, by simp, by simp, ?_, ?_⟩
  · intro v hv hne; exact absurd ((Submodule.mem_bot ℝ).1 hv) hne
  · intro w _ hne; exact absurd (Subsingleton.elim w 0) hne
