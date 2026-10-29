import D2SquareSchur

open ResearchFormalCoreR1

example (u v a b c : ℝ) (huv : u + v ≠ 0) :
    u * (a-c)^2 + v * (b-c)^2 =
      u*v/(u+v)*(a-b)^2 + (u+v)*(c-(u*a+v*b)/(u+v))^2 :=
  D2SquareSchur.weighted_square_identity u v a b c huv

example (u v a b c : ℝ) (hu : 0 < u) (hv : 0 < v) :
    u*v/(u+v)*(a-b)^2 ≤ u*(a-c)^2 + v*(b-c)^2 :=
  D2SquareSchur.weighted_square_lower u v a b c hu hv

example (m2 m4 q : ℝ) (hm2 : 0 < m2) (hm4 : m2^2 < m4)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1/4) :
    min (m4-m2^2) (2*m2^2) ≤ d2Schur m2 m4 q :=
  D2SquareSchur.schur_no_upper m2 m4 q hm2 hm4 hq0 hq1

example (m2 m4 q s0 g0 : ℝ) (hs0 : 0 < s0) (hg0 : 0 < g0)
    (hm2 : s0 ≤ m2) (hgap : g0 ≤ m4-m2^2) (hq0 : 0 ≤ q) (hq1 : q ≤ 1/4) :
    min g0 (2*s0^2) ≤ d2Schur m2 m4 q :=
  D2SquareSchur.schur_from_lower_inputs m2 m4 q s0 g0 hs0 hg0 hm2 hgap hq0 hq1
