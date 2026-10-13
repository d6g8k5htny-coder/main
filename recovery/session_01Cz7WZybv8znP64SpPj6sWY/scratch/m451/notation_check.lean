import Mathlib.Analysis.Calculus.ContDiff.Defs
set_option pp.all true
open scoped ContDiff in
#check (∞ : WithTop ℕ∞)
#check (((⊤ : ℕ∞)) : WithTop ℕ∞)
open scoped ContDiff in
example : (∞ : WithTop ℕ∞) = ((⊤ : ℕ∞) : WithTop ℕ∞) := rfl
