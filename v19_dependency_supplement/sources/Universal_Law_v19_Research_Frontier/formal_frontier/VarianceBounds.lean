import Init
/- Universal natural-number numerator bounds for the rational parameter s=n/d.
   These prove scalar inequalities, not a Gaussian probability theorem. -/
set_option autoImplicit false
namespace Frontier

private theorem add_mul_explicit (a b x : Nat) : (a+b)*x = a*x + b*x :=
  (Nat.mul_comm (a+b) x).trans ((Nat.mul_add x a b).trans
    ((congrArg (fun t => t + x*b) (Nat.mul_comm x a)).trans
      (congrArg (fun t => a*x + t) (Nat.mul_comm x b))))

theorem second_moment_bound (n d : Nat) (h : n ≤ d) :
    d*d + 3*(n*n) ≤ 4*(d*d) := by
  have hs : n*n ≤ d*d := Nat.mul_le_mul h h
  have hb := Nat.add_le_add_left (Nat.mul_le_mul_left 3 hs) (d*d)
  exact Nat.le_trans hb (Nat.le_of_eq
    ((Nat.add_comm (d*d) (3*(d*d))).trans (Nat.succ_mul 3 (d*d)).symm))

theorem fourth_moment_bound (n d : Nat) (h : n ≤ d) :
    9*((d*d)*(d*d)) + 18*((n*n)*(d*d)) + 105*((n*n)*(n*n)) ≤ 132*((d*d)*(d*d)) := by
  have hs : n*n ≤ d*d := Nat.mul_le_mul h h
  have hm : (n*n)*(d*d) ≤ (d*d)*(d*d) := Nat.mul_le_mul hs (Nat.le_refl (d*d))
  have hq : (n*n)*(n*n) ≤ (d*d)*(d*d) := Nat.mul_le_mul hs hs
  have hb := Nat.add_le_add
    (Nat.add_le_add_left (Nat.mul_le_mul_left 18 hm) (9*((d*d)*(d*d))))
    (Nat.mul_le_mul_left 105 hq)
  exact Nat.le_trans hb (Nat.le_of_eq
    ((congrArg (fun t => t + 105*((d*d)*(d*d)))
      (add_mul_explicit 9 18 ((d*d)*(d*d))).symm).trans
      (add_mul_explicit (9+18) 105 ((d*d)*(d*d))).symm))

end Frontier
