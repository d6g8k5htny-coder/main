"""The compact-window coefficient c_{B,K} of [LP] Theorem B in d = 5: reference value and transfer to the torus for every
L >= 10 and every frame.

Standard library only. Run from anywhere:  python -B -S window_coefficient_d5.py [--mutant NAME]   (about eighty seconds)
Every printed interval [lo, hi] is a rigorous enclosure: interval arithmetic over `decimal` (60 digits, directed rounding; exp
and sqrt widened by two units in the last place and checked against exact rational brackets by rule LIBRARY_EXACT), pi by
Machin, fractional powers by exact integer roots, erf and the lower incomplete gamma function by positive series with geometric
tails (and, for |x| >= 12, the enclosure 0 < erfc(x) < e^{-x^2}/(x sqrt(pi))), arctan and Owen's T by alternating series bracketed
by consecutive partial sums (the Math-#197 toolkit, unchanged), and one one-dimensional integral by an order-48 Taylor expansion
at each cell centre with a Cauchy remainder (Math-#199 section 3, as in Math-#209).

Object ([LP] = imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md, (11.3) with the section 15 factorization,
both stated for every d, for compact windows B = [b_-, b_+] and K = [k_-, k_+] of positive length, b_- < b_+ and
0 < k_- < k_+ < infinity; frame (u, w1, ..., w4), G = grad f, V = (f_uu, f_uw1, ..., f_uw4), A the 4 x 4 transverse Hessian,
t = f_uuu):

  c^(5)_{B,K} = 144 int_{S^4} p_G(0) p_V(0) I_B(u) J_K(u) dsigma(u),   I_B(u) = E[1{f in B} det(A)^2 1{A < 0} | V = 0],
  J_K(u) = int_K k^(4/3) phi_tau(12 k) dk,   tau^2 = Var(t | G = 0).

Reference kernel.  (f, A) | V = 0 is f ~ N(0, 2/3) and A = Q - f I with Q the 4 x 4 GOE (density ~ exp(-tr Q^2/4)) independent
of f.  On the ordered eigenvalue sector of A, with total trace -T and nested differences (w, s, q) (Jacobian 1/24), the exponent
is 3T^2/112 + w^2/48 + s^2/24 + q^2/8, the birth window enters only through
W_B(T) = Phi(sqrt(7/2) (b_+ - T/7)) - Phi(sqrt(7/2) (b_- - T/7)), and only odd powers of q occur, so the q- and s-integrals are
exact; in the w-integral the Owen-type terms int_0^T e^{-w^2/48} erf(sqrt(g) w) dw (g = 1/24, 1/6) cancel identically, so

  I^ref_B = (sqrt(3/7)/(1536 pi)) int_0^inf exp(-3T^2/112) F_4(T) W_B(T) dT,   F_4 elementary (polynomials, exp(-T^2/16),
                                                                                erf(T/4), exp(-T^2/48) erf(T/sqrt24), exp(-T^2/48) erf(T/sqrt6), exp(-3T^2/16)),

and B = R returns Math-#202's D_4.  The same code at m = 2 returns 29/6 - sqrt6 and Math-#197's d = 3 window values, and at m = 3
Math-#201's D_3 and Math-#209's d = 4 band factor.

Torus.  If (1-eps) C'_ref <= C' <= (1+eps) C'_ref for the covariance of the 11-vector (f, A) | V = 0, the density comparison and
the degree-8 homogeneity of det(A)^2 give (Lemma S of Math-#205 with n = 11, degree 8)

  (1-eps)^(19/2)/(1+eps)^(11/2) I^ref_{B/sqrt(1-eps)}  <=  I'_B  <=  (1+eps)^(19/2)/(1-eps)^(11/2) I^ref_{B/sqrt(1+eps)}.

lambda_min(C'_ref) = (8 - 2 sqrt13)/3 = 0.2630 (below the d = 4 floor 3/10): eps = ||C' - C'_ref||_F / (1/4) over the entry box of
the d = 5 image bound E^(5)_L = 60500 (76 L^6 + 15) exp(-L^2/2); p_G(0), p_V(0) and tau^2 over the same box.  One enclosure covers
every frame and every L >= 10.  Windows touching k = 0 or infinity, unbounded B, and singleton windows (b_- = b_+ or k_- = k_+,
where the integral is zero) are evaluated as coefficient integrals only.
Scientific effect NONE.
"""
