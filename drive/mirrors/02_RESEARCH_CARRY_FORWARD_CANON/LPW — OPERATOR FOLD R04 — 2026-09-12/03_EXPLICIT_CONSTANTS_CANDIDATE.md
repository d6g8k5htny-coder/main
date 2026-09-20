# QC-RETURN03 — A deliberately conservative explicit LPW bound pair

**Date:** 2026-09-12. **Class:** NEW AUTHOR-SIDE ANALYTICAL CANDIDATE, independent review required. This is not part of Kimi's already-endorsed candidate hash and does not inherit that endorsement. It gives a fully specified, extremely weak quantitative fallback, not a useful numerical prediction or an optimal coefficient. Mathematical inequalities below are proved in the text; the executable companion checks their symbolic and rational arithmetic components. It is not a proof-assistant certificate.

## 0. Proposed conclusion and exact dependencies

Under exactly the original LPW 2D side-24 covariance, six-pin continuous Gaussian regression law, typed pair determinant weight, and elder-rule estimand, the following explicit pair is proposed:

\[
\boxed{r_*=10^{-28},\qquad c_*=10^{-1235}.}
\]

The proposed conclusion is

\[
1-q(r,6/5)\ge c_*r^3\qquad(0<r\le r_*).
\]

Both numbers are exact positive rational powers of ten. They must never be materialized as ordinary floating-point numbers: c_* underflows and an output of zero would be false. Store it as {base:10, exponent:-1235, exact:true}.

The argument retains the reviewed candidate's polynomial, path, delta=1/1024, original K_J box, weight bound 65r^4, and determinant-normalizer construction. No tighter compact box is silently substituted. The new work makes quantitative the finite-r covariance neighborhood and the norm/density bounds. All required original proof material, the R3 clarification, and the previous endpoint proof are included in prior_reconciliation/.

New finite-r claim:

\[
\boxed{\tfrac1{10}I\preceq\Gamma_r\preceq81I
\quad (0\le r\le R),\qquad R=1/1600,}
\]

where Gamma_r is the covariance of the candidate's normalized ten-vector V_r=(U_r,J). This interval claim is a new supplement, not a restatement of an endpoint eigenvalue diagnostic.

## 1. Exact-periodic derivative moments through order four

Write the one-dimensional spectral moments as a_{2j}=(-1)^j k_{24}^{(2j)}(0). The probabilists' Hermite polynomial identity gives

    a_{2j} = [H_{2j}(0)*(-1)^j
               + 2 sum_{n>=1}(-1)^j H_{2j}(24n)t^(n^2)]/Z,
    t=exp(-288), Z=1+2 sum_{n>=1}t^(n^2).

For j=1,2,3,4, the coefficient absolute sum of

    (-1)^j H_{2j}(24n) - (2j-1)!!

is less than 10^13; its degree is at most eight, and n>=1. Since t<10^-120, n^2>=n, and

    n^8 <= 8!*binom(n+7,8),
    sum_{n>=1} binom(n+7,8)t^n = t/(1-t)^9,

we obtain

    |a_{2j}-(2j-1)!!|
      <= 2*10^13*8!*t/(1-t)^9
      < 10^22*t < 10^-98.

The normalization is already accounted for by subtracting the planar value inside the numerator; Z>=1. This is an infinite sum estimate, not a finite spectral truncation.

In particular

    0<a2<2, 0<a4<4, 0<a6<16, 0<a8<106.

Positivity follows directly from the exact positive spectral weights. For the product covariance in two dimensions,

    E|partial_x^i partial_y^j f(z)|^2 = a_(2i)*a_(2j).

For i+j<=4, the possible upper products are bounded by 106 (the other order-four cases are <32 and <16). Consequently

\[
\sup_{z,\ |\alpha|\le4}\|\partial^\alpha f(z)\|_{L^2}\le11.
\]

The coordinate units and torus side are precisely those of the original candidate.

## 2. L2 convergence with an explicit rate for the Hermite pin frame

Use the candidate's exact transformation T_r and observed vector

    u_r=(b-r^3/12,-r^2/4,0,0,0,1/3), b=6/5.

Let

    U0=(f,fx,fy,fxx/2,fxy,fxxx/6)(0),
    J=(fyy,fxxy/2,fxyy/2,fyyy/6)(0),
    V0=(U0,J), Vr=(Ur,J).

Taylor expansions below are in Gaussian L2. A remainder of order p is bounded by its integral kernel times the uniform L2 derivative bound. No sample-path supremum or conditioning is used at this step.

The six coordinate errors have the following bounds:

| coordinate of U_r-U_0 | L2 bound |
|---|---|
| value Hermite coordinate | 11 r^4/128 |
| x-gradient Hermite coordinate | 77 r^3/384 |
| average y-gradient | 11 r^2/8 |
| divided x-gradient / 2 | 11 r^2/48 |
| divided y-gradient | 11 r^2/24 |
| cubic Hermite coordinate | 55 r/96 |

Derivation: expand f(+/-r/2,0) through degree three with L2 remainders <=11r^4/384, and f_x(+/-r/2,0), f_y(+/-r/2,0) through degree two with remainders <=11r^3/48. Insert these into the exact T_r rows. The r^2 terms cancel in row 1 and the cubic terms cancel in row 2. Row 2's two remainder coefficients are 3/384 and 4/384. For row 3, the linear Taylor expansion of f_y with its second-derivative L2 bound gives 11r^2/8. The remaining rows follow from the displayed sums/differences. The row-6 error coefficient is 1/96+1/24=5/96. All three normalized derivative factors 1/2, 1/6 are retained.

Since r<=1, the squared Euclidean L2 error is bounded by

    E||Vr-V0||_2^2
      <=121r^2[(1/128)^2+(7/384)^2+(1/8)^2
                  +(1/48)^2+(1/24)^2+(5/96)^2]
      =(186461/73728)r^2 <4r^2.

Thus ||Vr-V0||_{L2(Euclidean)} <=2r. Each coordinate of V0 has variance <4, so ||V0||_{L2(Euclidean)}<sqrt(40)<7.

## 3. Explicit finite-r covariance interval

All these vectors are centered. Writing Dr=Vr-V0 and expanding covariance gives

    ||Gamma_r-Gamma_0||_op
       <= 2||V0||_L2||Dr||_L2+||Dr||_L2^2
       <=28r+4r^2 <=32r          (r<=1).

For example, each cross-covariance operator is bounded by the product of these L2 norms by applying scalar Cauchy-Schwarz to its bilinear form. This estimate does not require the errors to be independent of V0.

The preceding exact-periodic endpoint supplement proves

    lambda_min(Gamma_0)>31/250.

For R=1/1600, the Rayleigh quotient bound therefore gives

    lambda_min(Gamma_r)>31/250-32/1600=13/125>1/10.

Also

    tr Gamma_r=E||Vr||^2 <=(7+2r)^2<=81,
    ||Gamma_r||_op<=81.

The six-pin covariance is a principal submatrix, so the same lower bound 1/10 holds. The four-dimensional conditional covariance Sigma_r of J given U_r has the bounds

\[
\tfrac1{10}I\preceq\Sigma_r\preceq81I.
\]

For the lower bound, y^T Sigma_r y is the minimum of (x,y)^T Gamma_r(x,y) over x. The full covariance lower bound gives at least ||y||^2/10. For the upper bound use Sigma_r<=Gamma_JJ<=81I.

This proves a positive neighborhood including all 0<r<=R. It is not inferred from finite rungs.

## 4. An explicit global Gaussian C4 fourth-moment bound

Let alpha=pi/12. The exact real Fourier representation can be written as the zero mode plus independent sine/cosine modes from one representative of each pair {n,-n}. With p_n the full-lattice spectral weights, p_n=exp(-alpha^2|n|^2/2)/Z_spec and Z_spec>=1.

A pointwise bound for the full C4 norm is obtained by summing the absolute mode coefficients times (1+|alpha n|_1)^4. Taking L4 norms by Minkowski, a standard real Gaussian coefficient has L4 norm 3^(1/4)<2. The two coefficients per nonzero cosine/sine pair and their sqrt(2p_n) normalization are accounted for by

    (E||f||_C4^4)^(1/4)
       <=sqrt(2)*3^(1/4) sum_{n in Z2} sqrt(p_n)(1+|alpha n|_1)^4
       <4 sum_{n in Z2} exp(-alpha^2|n|^2/4)(1+|alpha n|_1)^4.

Since 1/4<alpha<1/3,

    exp(-alpha^2 n^2/4)<=exp(-n^2/64),
    1+alpha(|n1|+|n2|) <= (1+|n1|)(1+|n2|).

For q=64/65, exp(-n^2/64)<=q^|n|. This follows from n^2>=|n| and exp(1/64)>65/64. Therefore, for S=sum_{n in Z}exp(-n^2/64)(1+|n|)^4,

    S <=2 sum_{n>=0}(n+1)^4 q^n
      <=48 sum_{n>=0}binom(n+4,4)q^n
      =48*65^5.

The full two-dimensional bound is consequently

    (E||f||_C4^4)^(1/4) <4(48*65^5)^2
      =12407264266410000000000 <10^23.

Absolute summability also justifies almost-sure C4 convergence of the Fourier series and passage to these norm bounds. No conditional-residual stationarity is assumed. Lower Ck norms and first moments are bounded by the same number.

## 5. Explicit regression-operator and conditional-moment bounds

For either V=Vr (ten coordinates) or V=Ur (six), set C_r(z)=Cov(f(z),V). Componentwise covariance Cauchy-Schwarz and tr Cov(V)<=81 give

    ||partial^alpha C_r(z)||_2^2
      <= E|partial^alpha f(z)|^2 * tr Cov(V)
      <=121*81,

uniformly over |alpha|<=4 and z on the torus. Thus this vector norm is <=99. Since ||Cov(V)^(-1)||_op<=10, the regression map

    L_r v = C_r Cov(V)^(-1)v

has operator norm from Euclidean space to C4 at most 990<1000.

Furthermore, ||V||_L1<=9 and ||V||_L4<=18: for a centered Gaussian vector,

    E||V||^4=(tr Gamma)^2+2 tr(Gamma^2)<=3(tr Gamma)^2,

and 3^(1/4)*9<18.

For the pinned vector, ||u_r||<2. In the original box

    K_J=[-11,1] x [2-delta,2+delta] x [-delta,delta]^2,

we have ||j||<12 and ||(u_r,j)||<14. These inequalities use the original box, without changing it.

The regression representation, including its nonstationary residual, obeys the triangle inequality

    ||f | V=v||_{Lp(C4)}
      <= ||f||_{Lp(C4)}+1000(||V||_Lp+||v||).

For the ten-coordinate conditioning and every j in K_J,

    E[M4 | Ur=u_r,J=j] <=10^23+1000(9+14)<10^24.

Choose the explicit upper bound

\[
\boxed{B_4=10^{24},\qquad K=2\times10^{24}.}
\]

For the six-coordinate conditioning, use p=4 and max(1,||f||_C3)<=1+||f||_C3:

    (E_Qr[max(1,||f||_C3)^4])^(1/4)
      <=1+10^23+1000(18+2)<2*10^23.

Its fourth power is <16*10^92<10^96. Choose

\[
\boxed{B_3=10^{96},\qquad C_Z=4\times10^{96}.}
\]

These are chosen dominating bounds, not claimed exact moments. The conditional kernel is the continuous regression version for every j, as in the accepted R3 clarification. Markov is applied at each j before integration over the width-r event.

## 6. Uniform density floor on the same interval and compact box

First improve the conditional mean bound using the exact endpoint projection, rather than the extremely loose product of three matrix norms. Write Gamma_r in blocks A_r=Cov(U_r), B_r=Cov(J,U_r), and D=Cov(J). The matrix D is fixed because J is fixed. Put beta_0=B_0 A_0^(-1). The exact endpoint formulas in the preceding supplement give

    beta_0 = [ -a2, 0, 0, 0, 0, 0;
                 0, 0, -a2/2, 0, 0, 0;
                 0, -a2/2, 0, 0, 0, 0;
                 0, 0, -a4/(6a2), 0, 0, 0 ].

The moment estimates imply 99/100<a2<101/100 and a4<301/100. Consequently

    ||beta_0||_op^2 <= ||beta_0||_F^2
       =3a2^2/2+a4^2/(36a2^2)<4.

For the exact observed u_r and r<=1,

    ||beta_0 u_r||
      <=a2*(b+r^2/8)
      <=(101/100)*(6/5+1/8)=5353/4000<7/5.

The identity B_0=beta_0 A_0 gives

    mu_r-beta_0 u_r
      =[(B_r-B_0)-beta_0(A_r-A_0)] A_r^(-1)u_r.

All subblock errors are at most ||Gamma_r-Gamma_0||<=32r. Hence for r<=1/1600,

    ||mu_r-beta_0 u_r|| <=(1+2)*32r*10*2<=6/5,
    ||mu_r||<13/5<3.

This is a uniform bound, not a center evaluation.

Also Sigma_r<=D and

    tr D =a4+a2*a4/2+a6/36
          <4+4+16/36=76/9<9.

Thus we may use

    ||mu_r||<=3, ||j||<=12,
    (1/10)I <= Sigma_r <=9I

on the entire same interval and original K_J. The exact four-dimensional Gaussian density is bounded below by

    (2*pi)^(-2)*9^(-2)*exp(-(12+3)^2/(2/10)).

Here N=5*15^2=1125. Since pi<4, the prefactor exceeds 1/5184>10^-4. Since exp(1)<10, exp(-N)>10^-N. Therefore the exact rational choice

\[
\boxed{m_*=10^{-1129}}
\]

is a valid uniform lower density bound throughout [0,R] x K_J, within this derivation. No rounded pi or exp value, finite-r mesh, or planar-moment substitution enters the bound.

## 7. Geometric radius and final lower coefficient

Take r_*=10^-28. It is below R=1/1600 and

    256*K*r_*=512*10^24*10^-28=0.0512<1.

The original geometric condition holds:

    16delta+8Kr_* =1/64+1/625 <3/64<1/16.

The candidate's construction therefore gives its same field event, same strict topology, same weight >=65r^4, and

    Qr(G_r)>=16m_*delta^4 r,
    Zr<=4B3 r^2.

Thus

    1-q >=(260m_*delta^4/B3)r^3.

Finally delta^4=2^-40 and 260/2^40>10^-10, because 260*10^10>2^40. It follows that

    260m_*delta^4/B3
      >10^(-1129)*10^-10*10^-96
      =10^(-1235)=c_*.

This proves the proposed explicit inequality within the displayed author-side extension of LPW. It remains a candidate for independent mathematical review. Both constants are fixed before r varies, and all bounds use the same original probability law and original compact box.

## 8. What this is useful for

The pair is intentionally impractical. Its purpose is to give a fully specified analytical fallback and to expose exactly which estimates dominate the loss. A sharper constants agent should improve covariance-block bounds, the mean bound, the compact box, and Fourier/regression moment estimates under an explicitly versioned successor.

This result is not a new limiting coefficient, a 0.9144-class constant, an empirical prediction, a two-sided law, a three-dimensional result, a WP closure, a W8 result, or a P0.1 closure. The numerical radius 10^-28 is a rigor target, not a recommendation to sample tiny separations in ordinary floating-point arithmetic.

## 9. Independent review targets Q1–Q7

Q1: exact periodic eighth-moment tail and L2 derivative bound 11.
Q2: all six L2 Hermite errors, especially row 2 and row 6 cancellations.
Q3: covariance perturbation operator norm and explicit R=1/1600; Schur lower bound.
Q4: full real Fourier normalization, sine/cosine multiplicities, C4 L4 majorant.
Q5: representer vector bound, nonstationary-residual triangle inequality, B3/B4 distinction.
Q6: uniform Gaussian density lower bound, correct inequality directions, exact tiny-number representation.
Q7: original topology/weight linkage and final c,r0 powers; no new law or compact-set substitution.

A failure in Q1–Q7 affects this quantitative successor; it does not automatically retract the independently reported qualitative LPW endorsement. No existing source or state record is changed.
