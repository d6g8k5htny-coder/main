import Specifications.Lifetime

/-! Probe (Lifetime-A_1): confirm binder structure of the 11 Props and the degeneracies cited
in the findings. Probe file only; `example … := by sorry` would be allowed here but none is
needed: everything below is `#check` / `#print` / fully proved. -/

open UniversalLaw.Spec.Lifetime

#check @HessianHasInertia
#check @IsCandidatePair
#check @IsCriticalPoint
#check @IsFiniteBar
#check @IsIndexSaddle
#check @IsLPeriodic
#check @IsLifetimeDensityVersion
#check @IsMorseDistinct
#check @IsNondegMax
#check @IsSmooth
#check @IsUnrestrictedLeadingCoefficient

-- Universal closures are false / degenerate: a few cheap witnesses.

-- IsCriticalPoint holds for every function at every point where it is not differentiable.
example (d : ℕ) (g : E d → ℝ) (x : E d) (h : ¬ DifferentiableAt ℝ g x) : IsCriticalPoint d g x :=
  fderiv_zero_of_not_differentiableAt h

-- IsCriticalPoint and IsSmooth hold for every constant function (so a critical point need not be isolated).
example (d : ℕ) (c : ℝ) (x : E d) : IsCriticalPoint d (fun _ => c) x := by
  unfold IsCriticalPoint; simp

example (d : ℕ) (c : ℝ) : IsSmooth d (fun _ => c) := fun _ => contDiff_const

-- IsLPeriodic with L = 0 is trivially true for every g (the lattice vector is 0).
example (d : ℕ) (g : E d → ℝ) : IsLPeriodic d 0 g := by
  intro x n
  simp [latticeVec]

-- IsSmooth fails for a non-continuous function in d = 1 (so the universal closure is false).
example : ¬ IsSmooth 1 (fun x : E 1 => if x 0 ≤ 0 then (0:ℝ) else 1) := by
  intro h
  have hc : Continuous (fun x : E 1 => if x 0 ≤ 0 then (0:ℝ) else 1) := (h 0).continuous
  sorry
