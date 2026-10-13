import Specifications.Field
/-! Probe (Field group, round 4): the ten Field Props are parametrised predicates. This file
checks that a theorem of the type of three of them (universal closure over their own binders)
is refutable, i.e. that they are vocabulary, not targets. Probe only; `example`/`sorry` allowed here. -/
open UniversalLaw.Spec.Field

#check @HessianHasInertia
#check @IsCriticalPoint
#check @IsLPeriodic
#check @IsSmooth
#check @IsLifetimeDensityVersion
#check @finrank_euclideanSpace
#check @Submodule.finrank_le
#check @ContinuousLinearMap.fderiv
#check @Set.Infinite.ncard

-- HessianHasInertia: with d = 0 and j = 1 no subspace of E 0 has finrank 1.
example : ¬ ∀ (d : ℕ) (g : E d → ℝ) (x : E d) (j : ℕ), HessianHasInertia d g x j := by
  intro h
  obtain ⟨V, W, -, hV, -, -, -⟩ := h 0 0 0 1
  have hle := Submodule.finrank_le V
  rw [hV, finrank_euclideanSpace] at hle
  simp at hle

-- IsCriticalPoint: the coordinate functional on E 1 has nonzero derivative.
example : ¬ ∀ (d : ℕ) (g : E d → ℝ) (x : E d), IsCriticalPoint d g x := by
  intro h
  have h1 := h 1 (EuclideanSpace.proj (0 : Fin 1)) 0
  unfold IsCriticalPoint at h1
  rw [ContinuousLinearMap.fderiv] at h1
  have h2 := congrArg (fun f : E 1 →L[ℝ] ℝ => f (EuclideanSpace.single 0 1)) h1
  simp at h2

-- IsLPeriodic: the coordinate functional is not 1-periodic on E 1.
example : ¬ ∀ (d : ℕ) (L : ℝ) (g : E d → ℝ), IsLPeriodic d L g := by
  intro h
  have h1 := h 1 1 (fun x => x 0) 0 (fun _ => 1)
  simp [latticeVec] at h1
