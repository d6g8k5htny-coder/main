# LPW-R05-RAYLEIGH — proposed repair preserving the useful numerical scale

**Status:** NEW AUTHOR-SIDE ANALYTIC REPAIR + DIRECTED-INTERVAL COMPUTATION; targeted external review required. The previous Kimi endorsement does not automatically attach to this file. The exact qualitative LPW candidate and the exact endpoint floor 31/250 are inherited premises at their stated scopes. No full formal proof assistant was run.

## 1. Exact real Fourier representation and amplitude lemma

Let p_k>0 be the normalized exact side-24 two-dimensional spectral masses, sum_k p_k=1, and let (xi_k,eta_k) be independent pairs of independent N(0,1) variables for all full-lattice modes. The real field

    f(z)=sum_k sqrt(p_k)[xi_k cos(k.z)+eta_k sin(k.z)]

has covariance sum_k p_k cos(k.(z-w)), the required normalized covariance. Redundancy between opposite lattice modes is harmless: this is a valid Gaussian representation with that exact covariance, not a half-lattice convention with a missing sqrt(2).

For every multi-index alpha, differentiation rotates the sine/cosine phase. Consequently, pointwise for every z,

    |d^alpha{xi cos(k.z)+eta sin(k.z)}|
        <= |k_1|^alpha_1 |k_2|^alpha_2 sqrt(xi^2+eta^2).

Put rho=sqrt(xi^2+eta^2). For the max-convention C^p norm,

    ||f||_{C^p} <= sum_k sqrt(p_k)(1+|k|)^p rho_k.

The absolute Fourier majorant is integrable because the spectral weights have Gaussian decay. Tonelli and Minkowski therefore apply to finite partial sums and their limit. Independence between different amplitudes is not needed by Minkowski.

Exact moments:

    E rho = sqrt(pi/2),
    E rho^4 = E(xi^4+2xi^2 eta^2+eta^4)=3+2+3=8.

Since pi<4,

    sqrt(pi/2) <= 2 sqrt(2/pi),
    8 < 12+16/pi.

Thus the incoming NUMERICAL BUDGETS mu1=2sqrt(2/pi), mu4=12+16/pi can be retained as UPPER bounds for rho, even though mu4 was not the fourth moment of the originally used |xi|+|eta|. The repair changes the majorant and the justification, not the field law, the pins, the event, the compact box, or the determinant weight.

For comparison, expanding the old amplitude gives

    E(|xi|+|eta|)^4
      = 3+3+6 +4 E|xi|^3 E|eta|+4 E|xi| E|eta|^3
      =12+32/pi.

This identity must not remain printed as 12+16/pi in any repaired source.

## 2. Full infinite-sum bounds

Write h=pi/12, a=h^2/2, and use N=70 solely as a computational truncation. The normalization is the full product Z_1^2, where Z_1=sum_n exp(-a n^2).

For power p=0,...,12, n0=N+1, the two-sided one-dimensional tail is bounded by

    T_p <= 2 exp(-a n0^2)(h n0)^p/(1-rho_p),
    rho_p=exp(-a(2n0+1))(1+1/n0)^p <1/2.

The consecutive-term ratios decrease with n, so this bounds the ENTIRE tail. The program encloses numerator and denominator independently; it does not replace finite-torus moments by 1,3,15. Odd absolute moments needed in derivative estimates use Cauchy–Schwarz between enclosing even moments.

For the full two-dimensional sum S_p=sum_k sqrt(p_k)(1+|k|)^p, the square truncation misses shells max(|n1|,|n2|)=s. A shell has 8s points, n1^2+n2^2>=s^2, and |k|<=h sqrt(2)s. Hence its unnormalized contribution is bounded by

    T_shell(s)=8s exp(-a s^2/2)(1+h sqrt(2)s)^p.

The ratio of consecutive shell majorants is a product of decreasing positive factors and is <1 at s=71. The whole missing sum is at most T_shell(71)/(1-rho_shell). Division by the lower normalization bound is conservative. The program evaluates all finite expressions in outward interval arithmetic.

## 3. Hermite-profile covariance modulus (same analytic design, explicit bounds)

Use the exact six-pin Hermite coordinates and J of the qualitative candidate. On each Fourier mode, their symbols have the derivative powers listed in the program and real profiles of theta=k1*r/2:

    g0=cos(theta)+(theta/2)sin(theta),
    g1=(3/2)sinc(theta)-(1/2)cos(theta),
    g2=cos(theta), g3=-(1/2)sinc(theta), g4=-sinc(theta),
    g5=(theta cos(theta)-sin(theta))/(2 theta^3),
    g6=-1, g7=-1/2, g8=-1/2, g9=-1/6.

The apparent origin singularities are removable. For example,

    sinc(theta)=int_0^1 cos(t theta)dt,
    g5=-(1/2)int_0^1 t^2 sinc(t theta)dt.

Therefore |sinc|<=1, |sinc'|<=1/2, |g5|<=1/6<1/4 and |g5'|<=1/16<5/4 globally. All other declared bounds follow by the displayed elementary formulas. In particular |g0|<=1+|k1|R/4 and |g0'|<=1/2+|k1|R/4. The code uses deliberately looser valid constants for some profiles.

Differentiating a covariance product contributes (k1/2)(g_i' g_j+g_i g_j'). Combining the global profile bounds with complete spectral moments yields entry bounds C1_ij on the whole [0,R]. Hence

    ||Gamma_r-Gamma_0||op <= ||Gamma_r-Gamma_0||F
                           <= r sqrt(sum_ij C1_ij^2)=r L1.

For R=10^-5, the interval program gives L1 about 19.07115608049068. With the previously proved exact endpoint floor 31/250,

    lambda_min(Gamma_r) >= 31/250-L1 R >0.123809288439.

The six-pin principal block inherits this lower bound. So does its Schur complement: y^T S y=min_x (x,y)^T Gamma (x,y)>=lambda||y||^2. This is a uniform analytic modulus, not an interpolation of sampled rungs.

## 4. Conditional density on the explicitly tightened box

Retain the incoming explicitly changed compact set

    JBOX(R)=[-(10+2delta)R,0] x [2-delta,2+delta] x [-delta,delta]^2,
    delta=1/1024, R=10^-5.

Every thin box E_r lies in it for 0<r<=R; in particular the q-width remains 4delta*r and is not erased by rescaling.

The exact unconditional J covariance includes

    Cov(J_A,J_D3)=a2*a4/12.

The Gershgorin upper bound in the program includes this entry. The conditional Schur complement is Loewner-below this full J covariance. Let Lambda be that spectral upper bound and lambda the uniform lower bound above.

Let B_r=Cov(J,U_r), A_r=Cov(U_r), u_r the exact observed pin vector. The identity

    B_r A_r^-1-B_0 A_0^-1
      =(B_r-B_0)A_r^-1+B_0(A_r^-1-A_0^-1)

and the entrywise modulus imply the uniform mean bound M in the code, after adding the bounded difference u_r-u_0. This gives M about 1.22257583447 and R_J about 2.00097704161. The resulting four-dimensional density bound is

    phi_r(j)>=(2pi)^-2 Lambda^-2 exp(-(R_J+M)^2/(2lambda))
             >1.6759e-21 >10^-21

throughout JBOX(R) x [0,R]. The actual recorded interval endpoints, rather than these short displays, bind the numerical claim.

## 5. Conditional field norm bounds

With L_rv=C_r Gamma_r^-1 v, write the continuous Gaussian regression field as

    f|V_r=v = g_r+L_rv,  g_r=f-L_rV_r.

Use the triangle inequality through the representers; the residual is not assumed stationary. Covariance Cauchy–Schwarz and the uniform inverse bound give

    ||L_r||_{Euclidean -> C^k} <= sqrt(M_{2k} tr_sup)/lambda.

Here M_{2k} is the maximal exact derivative variance through order k, bounded using the exact moment intervals. For the six-pin field use A3; for the ten-coordinate conditioning use A4. The Gaussian vector fourth moment is exactly

    E||U_r||^4=(tr A_r)^2+2||A_r||F^2.

The Rayleigh majorant from Section 1 and the full Fourier sum bounds from Section 2 give

    E||f||C3^4 <= mu4 S3^4,
    E||f||C4 <= mu1 S4.

Substitute these into the original regression triangle bounds for M3=max(1,||f||C3) and the exact-order M4 seminorm. The resulting directed enclosures yield

    B3 <3790446482793,
    B4 <4715.75,
    K=max(1,2B4) <=9432.

Every bound on B4 is conditional on both the six pins and every j in JBOX(R). No fixed unconditional tail probability is subtracted from an O(r) jet-box mass.

## 6. Correct repaired bound and radius

Take m=10^-21, B3_ceiling=3790446482793, K_ceiling=9432. Then

    r0=1/(256*9432)=1/2414592 <R,
    16delta+8K_ceiling*r0=3/64<1/16.

The exact LPW topology/Taylor/weight proof, with its unchanged event, supplies

    1-q(r,6/5) >= c_exact r^3,  0<r<=r0,
    c_exact=260/(3790446482793*2^40*10^21)
           =6.2385427029355877929430410845189...e-44.

Thus **6.238e-44** is a downward-safe decimal floor. A constant can be chosen as this exact rational or any smaller positive number. This does not prove that the ratio converges, does not supply a matching 2D upper bound, and does not alter P0.1 or any 3D result.

## 7. Execution, review, and failure conditions

The independent finite-arithmetic implementation is `checks/interval_repair.py` (independent CODE, but not an independent mathematical review lineage or blind derivation). It uses mpmath.iv, stores interval endpoints as exact rational encodings of their binary endpoints, and imports the already proved endpoint floor rather than estimating it from one eigenpair residual. Normal and optimized outputs agree byte-for-byte; 39 checks pass. The oversized headline 6.239e-44 is rejected in both modes. The program does not use the old blanket ROUND=10^-45 convention.

Required Kimi review: (R05-Q1) exact full-lattice Gaussian representation and Rayleigh moment proof; (R05-Q2) full-tail and profile-modulus bounds; (R05-Q3) shared compact set, conditional mean/density/norm estimates; (R05-Q4) exact final fraction and radius; (R05-Q5) code-to-statement link and mutation of the actual exported headline. Record an exact hash-bound acknowledgment for the new amendment, not a retrospective claim that the old amplitude identity was true.

This is not a Lean proof. Its infinite-dimensional analytic claims are the written argument above and the inherited LPW proof; finite tests alone do not prove them. Any failed analytic inequality, law/normalization mismatch, invalid infinite tail, or wrong interval direction reopens this quantitative repair, not automatically the distinct qualitative/fallback argument.
