# Exact periodic endpoint companion and eigenfloor repair

**Status:** New author-side analytic companion with finite exact symbolic/rational checks. Not a replay of Kimi's missing raw numerical certificate and not a formal proof-assistant certificate. No change to LPW v1.0 is made. Independent review remains appropriate for this appendix.

## 1. Exact object

Use the exact LPW field kernel at L=24. It factors into identical normalized one-dimensional kernels

    k_L(t)=sum_{n in Z} exp(-(t+Ln)^2/2) / D_L,
    D_L=sum_{n in Z} exp(-(Ln)^2/2),
    K_L(x,y)=k_L(x) k_L(y).

Define the exact one-dimensional spectral moments

    mu_{2j}=(-1)^j k_L^{(2j)}(0).

They equal the normalized lattice moments with physical frequencies pi*n/12. They are NOT replaced by 1,3,15.

Order the ten coordinates as

    U0=(f,fx,fy,fxx/2,fxy,fxxx/6),
    J=(fyy,fxxy/2,fxyy/2,fyyy/6).

For multiindices alpha,beta their covariance is

    Cov(partial^alpha f,partial^beta f)
    =(-1)^{|beta|} partial^{alpha+beta} K_L(0).

If either component of alpha+beta is odd, this is zero. Otherwise it is the product of the corresponding one-dimensional moments with sign (-1)^{|beta|+|alpha+beta|/2}, followed by the coordinate scale factors.

## 2. Exact Schur complement and conditional mean

Direct block elimination, verified symbolically in the included checker, gives

    Cov(J | U0)=diag(
      mu4-mu2^2,
      mu2*(mu4-mu2^2)/4,
      mu2*(mu4-mu2^2)/4,
      (mu6-mu4^2/mu2)/36
    ).

At u0=(b,0,0,0,0,1/3), the exact conditional mean is

    E[J | U0=u0]=(-b*mu2,0,0,0).

Specializing (mu2,mu4,mu6)=(1,3,15) yields diag(2,1/2,1/2,1/6) and (-b,0,0,0). Those are the planar-moment values. Their numerical agreement with the periodic values is expected at moderate precision, but numerical agreement is not exact equality.

Indeed,

    mu2=1-[2 sum_{n>=1}(24n)^2 exp(-288n^2)]/D_24 < 1.

Thus the exact first conditional mean differs from -b, however minutely. The original LPW proof only needs positive-definiteness/continuity and does not need these quantities to be exact rationals.

## 3. Elementary rigorous moment perturbation bound

The relevant Hermite polynomials give, after subtracting the n=0 planar value and accounting for D_24,

    mu2-1 = D_24^{-1} sum_{n!=0} -(24n)^2 exp(-288n^2),
    mu4-3 = D_24^{-1} sum_{n!=0} [(24n)^4-6(24n)^2] exp(-288n^2),
    mu6-15= D_24^{-1} sum_{n!=0} [-(24n)^6+15(24n)^4-45(24n)^2] exp(-288n^2).

For |x|>=24 each displayed absolute polynomial is at most 2|x|^6. Also D_24>=1. Consequently every moment difference is bounded by

    4*24^6 * sum_{n>=1} n^6 exp(-288n^2).

For n>=1, log n<=n-1 and n^2>=1+3(n-1). Therefore

    n^6 exp(-288n^2) <= exp(-288) exp(-858(n-1)).

The sum is at most exp(-288)/(1-exp(-858))<2 exp(-288).

A finite rational check proves sum_{k=0}^5 (12/5)^k/k! > 10, hence exp(12/5)>10. Raising to the 120th power gives exp(288)>10^120. It follows that

    max(|mu2-1|,|mu4-3|,|mu6-15|)
    < 8*24^6*10^(-120) < 10^(-110) = epsilon.

All inequalities here are analytic/rational; no claimed 60-digit numerical lattice evaluation is being promoted into a tail theorem.

## 4. Full ten-coordinate covariance perturbation

Let Gamma_per be the exact periodic covariance and Gamma_pl its planar specialization. Since the total derivative order in every entry is at most six, the possible nonconstant moment monomials are mu2,mu4,mu6,mu2^2,mu2*mu4. Coordinate scale factors have magnitude at most one.

With epsilon<1, mu2<=2 and mu4<=4. The product differences are bounded by a small multiple of epsilon (at most 5 epsilon suffices); we use the deliberately loose common bound 40 epsilon. Thus

    max_ij |(Gamma_per-Gamma_pl)_ij| <= 40 epsilon,
    ||Gamma_per-Gamma_pl||_2 <= ||Gamma_per-Gamma_pl||_F <= 400 epsilon.

For a unit vector x, |x^T(Gamma_per-Gamma_pl)x|<=400 epsilon. Minimizing Rayleigh quotients bounds the change in the smallest eigenvalue by 400 epsilon. This supplies a full-matrix perturbation argument, not merely a residual for one approximate eigenpair.

## 5. Exact rational bracket for the planar smallest eigenvalue

The exact characteristic polynomial is

    det(x I-Gamma_pl)
    =(x-1)(4x^3-19x^2+18x-4)(12x^3-26x^2+11x-1)^2/576.

Let

    a=126553449667/10^12,
    b=126553449668/10^12,
    p(x)=12x^3-26x^2+11x-1.

The checker verifies all ten leading principal minors of Gamma_pl-a I are strictly positive. Sylvester's criterion therefore gives lambda_min(Gamma_pl)>a. It also verifies p(a)<0<p(b). By continuity a root of p lies in (a,b); that root is an eigenvalue of Gamma_pl by the characteristic-polynomial identity. Hence

    a < lambda_min(Gamma_pl) < b.

Combining with the periodic perturbation gives

    a-400*10^(-110) < lambda_min(Gamma_per)
                        < b+400*10^(-110).

In particular the exact periodic endpoint covariance satisfies

    lambda_min(Gamma_per) > 0.1265534496,
    lambda_min(Gamma_per) < 0.1265534497.

Therefore the user-relayed claim 'certified >=0.1265534497' is not a valid lower eigenfloor for this precise covariance. A rounded decimal approximation must not be rounded upward and called a lower bound. The PDF verdict itself prints the more precise numerical value, without that extra upward-rounded inequality; this correction distinguishes the relay from the PDF.

For orientation only, a 70-digit unvalidated floating calculation gives the planar value

    0.12655344966728947964415627118194802489390647012456...

The proof of the safe bracket is rational plus analytic perturbation, not that floating output. No value of r_G is supplied by this endpoint calculation. Obtaining an explicit all-r eigenfloor still requires controlling Gamma_r-Gamma_0 uniformly on [0,r_G].

## 6. Density floor caution

The candidate's m is a lower bound over a continuum of parameter values. In its written proof this is [0,r_G] x K_J with

    K_J=[-11,1] x [2-delta,2+delta] x [-delta,delta]^2,
    delta=1/1024.

The density near a thin-box center at a single r cannot substitute for that m. For illustration only, under the planar endpoint law with covariance diag(2,1/2,1/2,1/6) and mean (-6/5,0,0,0), the density at q=-1/4,A=2,B=D3=0 is about 0.0012825233. At the admissible fixed-box corner q=-11,A=2+delta,B=D3=delta it is about 5.9833432e-14. These are labeled diagnostics, not certified bounds for the finite-r periodic law.

A more efficient quantitative successor may explicitly use the smaller domain of the actual moving boxes:

    r in [0,r0], t in [-delta,delta],
    q=2r(-5+t),
    A in [2-delta,2+delta], B,D3 in [-delta,delta].

This does not change the r-Jacobian: the q interval still has width 4 delta r. If the density and conditional-moment proof are re-scoped to this domain, the certificate must say so explicitly and receive the appropriate review. One cannot keep the old fixed-box definition of m while inserting a new center value.

## 7. Reproduction and limitations

Run `python audit_review_claims.py` and `python -O audit_review_claims.py`. Both produce the same 27-check report. Symbolic equality is checked by simplifying the difference to zero, not by literal expression-tree identity.

The checker covers finite identities and rational inequalities. The Gaussian field factorization, tail argument, and Rayleigh-quotient reasoning are written above for mathematical review. No new independent review, numerical c/r0, all-r covariance certificate, or theorem-status transition is asserted.
