"""The compact-window coefficient c_{B,K} of [LP] Theorem B in d = 4: reference value and transfer to the torus for every
L >= 10 and every frame.

Standard library only. Run from anywhere:  python -B -S window_coefficient_d4.py [--mutant NAME]   (about seventy seconds)
Every printed interval [lo, hi] is a rigorous enclosure: interval arithmetic over `decimal` (60 digits, directed rounding; exp
and sqrt widened by two units in the last place and checked against exact rational brackets by rule LIBRARY_EXACT), pi by
Machin, fractional powers by exact integer roots, erf and the lower incomplete gamma function by positive series with geometric
tails, arctan and Owen's T by alternating series bracketed by consecutive partial sums (the Math-#197 toolkit, unchanged), and one
one-dimensional integral by an order-40 Taylor expansion at each cell centre with a Cauchy remainder (Math-#199 section 3).

Object ([LP] = imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md, (11.3) with the section 15 factorization,
both stated for every d; frame (u, w1, w2, w3), G = grad f, V = (f_uu, f_uw1, f_uw2, f_uw3), A the 3 x 3 transverse Hessian,
t = f_uuu):

  c^(4)_{B,K} = 144 int_{S^3} p_G(0) p_V(0) I_B(u) J_K(u) dsigma(u),   I_B(u) = E[1{f in B} det(A)^2 1{A < 0} | V = 0],
  J_K(u) = int_K k^(4/3) phi_tau(12 k) dk,   tau^2 = Var(t | G = 0).

Reference kernel.  (f, A) | V = 0 is f ~ N(0, 2/3) and A = Q - f I with Q the 3 x 3 GOE (density ~ exp(-tr Q^2/4)) independent
of f.  On the ordered eigenvalue sector of A, with total trace -T and differences (s, q), the birth window enters only through
W_B(T) = Phi(sqrt3 (b_+ - T/6)) - Phi(sqrt3 (b_- - T/6)), and only odd powers of q occur, so the q- and s-integrals are exact:

  I^ref_B = (1/(96 pi)) int_0^inf exp(-T^2/24) F_3(T) W_B(T) dT,    F_3 elementary (polynomials, exp(-T^2/24), exp(-T^2/6),
                                                                      erf(T/sqrt24), erf(T/sqrt6)),

and B = R returns Math-#201's D_3 = (50 pi + 200 arctan 2 - 228)/(9 pi).  The same code at m = 2 returns 29/6 - sqrt6 and
Math-#197's d = 3 window values (closed form through Owen's T).

Torus.  If (1-eps) C'_ref <= C' <= (1+eps) C'_ref for the covariance of the 7-vector (f, A) | V = 0, the density comparison and
the degree-6 homogeneity of det(A)^2 give (Lemma S of Math-#205 with n = 7, degree 6)

  (1-eps)^(13/2)/(1+eps)^(7/2) I^ref_{B/sqrt(1-eps)}  <=  I'_B  <=  (1+eps)^(13/2)/(1-eps)^(7/2) I^ref_{B/sqrt(1+eps)}.

lambda_min(C'_ref) = (7 - sqrt37)/3 = 0.3057 (below SIDE24's 1/3): eps = ||C' - C'_ref||_F / (3/10) over the entry box of the
d = 4 image bound E^(4)_L = 10240 (76 L^6 + 15) exp(-L^2/2); p_G(0), p_V(0) and tau^2 over the same box.  One enclosure covers
every frame and every L >= 10.  Scientific effect NONE.
"""
