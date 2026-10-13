"""The compact-window coefficient c_{B,K} of [LP] Theorem B in d = 6: reference value and transfer to the torus for every
L >= 10 and every frame.

Standard library only. Run from anywhere:  python -B -S window_coefficient_d6.py [--mutant NAME]   (about two minutes)
Every printed interval [lo, hi] is a rigorous enclosure: interval arithmetic over `decimal` (60 digits, directed rounding; exp
and sqrt widened by two units in the last place and checked against exact rational brackets by rule LIBRARY_EXACT), pi by
Machin, fractional powers by exact integer roots, erf and the lower incomplete gamma function by positive series with geometric
tails (and, for |x| >= 12, the enclosure 0 < erfc(x) < e^{-x^2}/(x sqrt(pi))), arctan and Owen's T by alternating series bracketed
by consecutive partial sums (the Math-#197 toolkit, unchanged), and one one-dimensional integral by an order-48 Taylor expansion
at each cell centre with a Cauchy remainder (Math-#199 section 3), carrying the Owen-type kinds as a certified cumulative layer
(Math-#204 section 3).

Object ([LP] = imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md, (11.3) with the section 15 factorization,
both stated for every d, for compact windows B = [b_-, b_+] and K = [k_-, k_+] of positive length, b_- < b_+ and
0 < k_- < k_+ < infinity; frame (u, w1, ..., w5), G = grad f, V = (f_uu, f_uw1, ..., f_uw5), A the 5 x 5 transverse Hessian,
t = f_uuu):

  c^(6)_{B,K} = 144 int_{S^5} p_G(0) p_V(0) I_B(u) J_K(u) dsigma(u),   I_B(u) = E[1{f in B} det(A)^2 1{A < 0} | V = 0],
  J_K(u) = int_K k^(4/3) phi_tau(12 k) dk,   tau^2 = Var(t | G = 0).

Reference kernel.  (f, A) | V = 0 is f ~ N(0, 2/3) and A = Q - f I with Q the 5 x 5 GOE (density ~ exp(-tr Q^2/4)) independent
of f.  On the ordered eigenvalue sector of A, with total trace -T and nested coordinates (w', w, s, q) (Jacobian 1/120), the
exponent is 3T^2/160 + w'^2/80 + w^2/48 + s^2/24 + q^2/8, the birth window enters only through
W_B(T) = Phi(2 (b_+ - T/8)) - Phi(2 (b_- - T/8)), and only odd powers of q occur, so the q-, s- and w-integrals are exact (in the
w-integral the Owen-type terms cancel identically, as in d = 5).  The w'-integral leaves three Owen-type kinds
K_{al,g}(T) = int_0^T e^{-al w^2} E_g(w) dw, (al, g) = (1/80, 1/16), (1/30, 1/24), (1/30, 1/6), with E_g(w) = int_0^w e^{-g s^2} ds:

  I^ref_B = (sqrt(3/8)/(46080 pi^(3/2))) int_0^inf exp(-3T^2/160) F_5(T) W_B(T) dT,
  F_5 = polynomials times exp(-T^2/5), exp(-3T^2/40), exp(-al T^2) E_g(T) and K_{al,g}(T), all three K with one polynomial R_10.

The K_{al,g} are Owen's T functions, K = (pi/sqrt(al g)) [arctan(a)/(2 pi) - T(sqrt(2 al) T, a)], a = sqrt(g/al); the quadrature
carries them as a certified cumulative layer, and rule OWEN_EXACT checks that layer against Owen's series.  B = R returns
Math-#204's D_5.  The same code at m = 2 returns 29/6 - sqrt6 and Math-#197's d = 3 window values, at m = 3 Math-#201's D_3 and
Math-#209's d = 4 band factor, and at m = 4 Math-#202's D_4, Math-#213's stated F_4 and its d = 5 band factor.

Torus.  If (1-eps) C'_ref <= C' <= (1+eps) C'_ref for the covariance of the 16-vector (f, A) | V = 0, the density comparison and
the degree-10 homogeneity of det(A)^2 give (Lemma S of Math-#205 with n = 16, degree 10)

  (1-eps)^13/(1+eps)^8 I^ref_{B/sqrt(1-eps)}  <=  I'_B  <=  (1+eps)^13/(1-eps)^8 I^ref_{B/sqrt(1+eps)}.

lambda_min(C'_ref) = (6 - sqrt(92/3))/2 = 0.2311 (below the d = 5 floor 1/4): eps = ||C' - C'_ref||_F / (23/100) over the entry
box of the d = 6 image bound E^(6)_L = 314496 (76 L^6 + 15) exp(-L^2/2); p_G(0), p_V(0) and tau^2 over the same box.  One
enclosure covers every frame and every L >= 10.  Windows touching k = 0 or infinity, unbounded B, and singleton windows
(b_- = b_+ or k_- = k_+, where the integral is zero) are evaluated as coefficient integrals only.
Scientific effect NONE.
"""
