import Init
/- Genuine bounded strategy attempt: a decision procedure can evaluate closed
   numerals, but does not eliminate the arbitrary natural parameters here. -/
set_option autoImplicit false
namespace Frontier

theorem second_moment_bound (n d : Nat) (h : n ≤ d) :
    d*d + 3*(n*n) ≤ 4*(d*d) := by
  decide

theorem fourth_moment_bound (n d : Nat) (h : n ≤ d) :
    9*((d*d)*(d*d)) + 18*((n*n)*(d*d)) + 105*((n*n)*(n*n)) ≤ 132*((d*d)*(d*d)) := by
  decide

end Frontier
