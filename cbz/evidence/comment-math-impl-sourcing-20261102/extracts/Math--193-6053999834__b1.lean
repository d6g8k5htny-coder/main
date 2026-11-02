import ContactAlgebra

set_option autoImplicit false
open SourceContactRank

-- A sign error in phase elimination fails this exact quarter-turn application.
example (R S : ℝ) (hcos : -S = 0) (hsin : R = 0) : R = 0 ∧ S = 0 := by
  apply contact_phase_elimination 1 0 R S (by norm_num)
  · simpa using hcos
  · simpa using hsin

-- Omitting circle normalization admits this nonzero null pair.
example : (1 : ℝ) * 0 - 0 * 0 = 0 ∧ (1 : ℝ) * 0 + 0 * 0 = 0 ∧ (1 : ℝ) ≠ 0 := by
  norm_num

-- Coordinate modes alone miss the odd cubic/linear contrast u=V=e0,C=c=1.
example (i : Fin 2) :
    (if i = 0 then (1 : ℝ) else 0) - (if i = 0 then (1 : ℝ) else 0) ^ 3 = 0 := by
  fin_cases i <;> norm_num

-- The doubled e0 mode exposes that same missing contrast with literal residual -6.
example : (2 : ℝ) * 1 - 1 * ((2 : ℝ) * 1) ^ 3 = -6 := by norm_num

-- Diagonal modes miss the mixed quadratic block u=e0,W=e1.
example (i : Fin 2) :
    -(if i = 0 then (1 : ℝ) else 0) * (if i = 1 then (1 : ℝ) else 0) = 0 := by
  fin_cases i <;> norm_num

-- The actual pair sum e0+e1 exposes it with literal residual -1.
example : -((1 : ℝ) + 0) * ((0 : ℝ) + 1) = -1 := by norm_num

-- Dropping u≠0 leaves a nonzero cubic weight invisible on all frequencies.
example (k : ℝ) : (0 : ℝ) - 1 * (k * 0) ^ 3 = 0 ∧ (1 : ℝ) ≠ 0 := by norm_num

-- Omitting the constant-mode A=0 condition misses a value/second contrast in D=1.
example : (1 : ℝ) - 1 * 1 = 0 ∧ (1 : ℝ) ≠ 0 := by norm_num

