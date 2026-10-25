import Init
/- n=d=1 saturates the fourth bound, so replacing 132 by 131 is false. -/
example : 9*((1*1)*(1*1)) + 18*((1*1)*(1*1)) + 105*((1*1)*(1*1)) ≤ (131 : Nat)*((1*1)*(1*1)) := by
  decide
