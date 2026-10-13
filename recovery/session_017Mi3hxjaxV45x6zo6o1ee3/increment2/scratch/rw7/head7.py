"""The compact-window coefficient c_{B,K} of [LP] Theorem B in d = 7: the cone moment D_6, c_{7,ref}, the reference window
coefficient, and the transfer to the torus for every L >= 10 and every frame.

Standard library only. Run from anywhere:  python -B -S window_coefficient_d7.py [--mutant NAME]   (about four minutes)
Every printed interval [lo, hi] is a rigorous enclosure: interval arithmetic over `decimal` (60 digits, directed rounding; exp
and sqrt widened by two units in the last place and checked against exact rational brackets by rule LIBRARY_EXACT), pi by
Machin, fractional powers by exact integer roots, erf and the lower incomplete gamma function by positive series with geometric
tails (and, for |x| >= 12, the enclosure 0 < erfc(x) < e^{-x^2}/(x sqrt(pi))), arctan and Owen's T by alternating series bracketed
by consecutive partial sums (the Math-#197 toolkit, unchanged), and one one-dimensional integral by an order-48 Taylor expansion
at each cell centre with a Cauchy remainder (Math-#199 section 3), carrying the Owen-type kinds as a certified cumulative layer
(Math-#204 section 3).

Object ([LP] = imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md, (11.3) with the section 15 factorization,
both stated for every d, for compact windows B = [b_-, b_+] and K = [k_-, k_+] of positive length, b_- < b_+ and
0 < k_- < k_+ < infinity; frame (u, w1, ..., w6), G = grad f, V = (f_uu, f_uw1, ..., f_uw6), A the 6 x 6 transverse Hessian,
t = f_uuu):

  c^(7)_{B,K} = 144 int_{S^6} p_G(0) p_V(0) I_B(u) J_K(u) dsigma(u),   I_B(u) = E[1{f in B} det(A)^2 1{A < 0} | V = 0],
  J_K(u) = int_K k^(4/3) phi_tau(12 k) dk,   tau^2 = Var(t | G = 0).

Reference kernel.  (f, A) | V = 0 is f ~ N(0, 2/3) and A = Q - f I with Q the 6 x 6 GOE (density ~ exp(-tr Q^2/4)) independent
of f.  On the ordered eigenvalue sector of A, with total trace -T and nested coordinates (w'', w', w, s, q) (Jacobian 1/720), the
exponent is T^2/72 + w''^2/120 + w'^2/80 + w^2/48 + s^2/24 + q^2/8, the birth window enters only through
W_B(T) = Phi((3/sqrt2) (b_+ - T/9)) - Phi((3/sqrt2) (b_- - T/9)), and only odd powers of q occur.  One exact layer recursion
integrates q, s, w, w' and w'' in closed form.  The w-layer cancellation of Math-#213 recurs; the w'-layer leaves the three
Owen-type kinds of Math-#222, K_{al,g}(y) = int_0^y e^{-al x^2} E_g(x) dx, E_g(x) = int_0^x e^{-g s^2} ds; the w''-layer
integrates them by parts, and the nested kinds int_0^T e^{-y^2/120} K_{al,g}(y) dy cancel identically, so that

  I^ref_B = (1/(sqrt3 Z_6)) int_0^inf exp(-T^2/72) F_6(T) W_B(T) dT,   Z_6 = 4423680 sqrt2 pi^(3/2) (Mehta),
  F_6 = polynomials times exp(-T^2/12), exp(-5T^2/24), exp(-al T^2) E_g(T), K_{1/48,1/16}(T) and exp(-T^2/120) K_{al,g}(T).

The K_{al,g} are Owen's T functions, K = (pi/sqrt(al g)) [arctan(a)/(2 pi) - T(sqrt(2 al) T, a)], a = sqrt(g/al); the quadrature
carries them as a certified cumulative layer, and rule OWEN_EXACT checks that layer against Owen's series.  The same layers with
the Vandermonde alone return Mehta's Z_6 through arctan(sqrt3) = pi/3 (rule MEHTA_Z6).  B = R gives the cone moment D_6 and
c_{7,ref} = Gamma(7/6) (3/2)^(1/3) D_6/(120 sqrt3 pi^(9/2)).  The same code at m = 2, ..., 5 returns 29/6 - sqrt6, Math-#197's
d = 3 window values, Math-#201's D_3, Math-#202's D_4, Math-#204's and Math-#222's D_5, and the band factors of Math-#209, #213
and #222.

Torus.  If (1-eps) C'_ref <= C' <= (1+eps) C'_ref for the covariance of the 22-vector (f, A) | V = 0, the density comparison and
the degree-12 homogeneity of det(A)^2 give (Lemma S of Math-#205 with n = 22, degree 12)

  (1-eps)^17/(1+eps)^11 I^ref_{B/sqrt(1-eps)}  <=  I'_B  <=  (1+eps)^17/(1-eps)^11 I^ref_{B/sqrt(1+eps)}.

lambda_min(C'_ref) = (10 - 2 sqrt22)/3 = 0.2064 (below the d = 6 floor 23/100): eps = ||C' - C'_ref||_F / (1/5) over the entry
box of the d = 7 image bound E^(7)_L = 1499596 (76 L^6 + 15) exp(-L^2/2); p_G(0), p_V(0) and tau^2 over the same box.  One
enclosure covers every frame and every L >= 10.  Windows touching k = 0 or infinity, unbounded B, and singleton windows
(b_- = b_+ or k_- = k_+, where the integral is zero) are evaluated as coefficient integrals only.
Scientific effect NONE.
"""
