# SIDE24 coefficient in dimensions two and three

Object: SIDE24-COEFFICIENT-D23-20260924-v1. Author: OpenAI / ChatGPT.
Disposition: author-side coefficient-evaluation candidate; nonauthor review open.

## Scope and exact parent

This note evaluates the coefficient defined in equation (15.2) of
UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1, SHA256
`9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`,
Drive ID `1foDgiDi4XIKOfbrZEE8dWKV_BkZU8LIb`, main issue63.
It does not independently accept that parent's global elder-selection or Kac-Rice
proof. Interpreting this coefficient as the finite-bar asymptotic constant remains
conditional on that proof. The parent source is not modified.

Use its centered variance-one field with the exact normalized covariance

    K_24(z) = sum_{n in Z^d} exp(-|z+24n|^2/2)
              / sum_{n in Z^d} exp(-|24n|^2/2),  d in {2,3}.

Let c_d,24 be precisely its equation (15.2), using ordinary sphere area, ordered
maximum/saddle roles and the full height/gradient pin Jacobian. Then the bounds
proved below, with endpoints rational decimal numbers, are

    0.07340691930603427103 < c_2,24 < 0.07340691930603427104,
    0.04177593184059834334 < c_3,24 < 0.04177593184059834335.

The uncertainty in these intervals is numerical/arithmetic uncertainty in this
specified expression, not a probability of correctness of the parent theorem.
No finite-radius error estimate, unrestricted remainder or RN/24-jet claim follows.

## 1. A nonperiodic reference coefficient with exact cone moments

Use K_infty(z)=exp(-|z|^2/2) only to define a reference CONTACT covariance and its
coefficient expression. No infinite-volume persistence theorem is asserted.
In an orthonormal frame with u=e1, covariance differentiation gives

    Cov(G)=I_d,
    Cov(H_ij,H_kl)=delta_ij delta_kl+delta_ik delta_jl+delta_il delta_jk,
    Var(t)=15,  Cov(t,G)=(-3,0,...,0),  t=partial_1^3 f.

Thus V=Hu has covariance diag(3,1,...,1), tau^2=Var(t|G=0)=6, and

    p_G(0) p_V(0) = (2pi)^(-d) / sqrt(3).

Writing m=d-1 and A for the transverse Hessian, Gaussian conditioning on V=0 gives

    Cov(A_ij,A_kl | V=0)
      = (2/3)delta_ij delta_kl + delta_ik delta_jl + delta_il delta_jk.

Equivalently A=Q+sqrt(2/3) Z I_m, with Q's diagonal variances2, off-diagonal
variances1, and independent upper-triangular entries, and Z an independent standard
normal. This explicitly specified reference law is not substituted for the actual
periodic law without an error bound.

For m=1 the variance of A is8/3. Centered Gaussian symmetry gives

    D_1=E[A^2 1{A<0}]=4/3.

For m=2 write A=[[s+x,y],[y,s-x]]. Then s,x,y are independent,
Var(s)=5/3 and Var(x)=Var(y)=1. The negative-definite cone is s<-R,
R=sqrt(x^2+y^2), and det A=s^2-R^2. The variable R^2 has density e^(-z/2)/2.
For a>=0, elementary integration gives

    integral_0^a (a-z)^2 e^(-z/2) dz/2 = a^2-4a+8-8e^(-a/2).

Using symmetry in s, E s^2=5/3, E s^4=25/3 and
E exp(-s^2/2)=sqrt(3/8), one obtains

    D_2 = (1/2)[25/3-20/3+8-8sqrt(3/8)]
        = 29/6-sqrt(6).

The truncation R<|s| is essential: half the UNTRUNCATED determinant-squared moment
would instead give29/6, an incorrect value. The shared scalar shift of the matrix
is likewise essential.

Substituting these moments and |S^1|=2pi, |S^2|=4pi into the exact parent coefficient
and simplifying 6^(2/3)/24^(1/3)=(3/2)^(1/3) gives

    c_d,ref = Gamma(7/6)*(3/2)^(1/3)*D_(d-1)
              / [2sqrt(3)*pi^(d-1)*sqrt(pi)], d=2,3.       (1)

This eliminates BOTH the angular and cone integrals for the reference expression.

## 2. A uniform, deterministic bound for every omitted periodic image

For q<=6 unit directional differentiations of phi(x)=exp(-|x|^2/2), product-rule
pairings bound the absolute derivative by

    e^(-|x|^2/2) sum_{j=0}^{floor(q/2)}
        q! |x|^(q-2j) / [2^j j! (q-2j)!].

At |x|>=1 this is at most76 |x|^6 e^(-|x|^2/2). The coefficient sum for q=6 is
1+15+45+15=76 and it bounds the smaller orders as well. At zero, every such
contraction has absolute value at most15.

For an integer point with maximum coordinate magnitude j>=1, there are at most
27j^3 possibilities for d<=3, |n|^6<=27j^6, and |n|^2>=j^2. Consequently

    sum_{n!=0} |n|^6 e^(-288|n|^2)
      <=729 sum_{j>=1} j^9 e^(-288j^2)
      <=1458 e^(-288).

The last step follows because successive terms have ratio at most512e^(-864)<1/2.
The code proves e^(288/125)>10 using the positive rational Taylor sum through20;
hence e^(-288)<10^(-125) and the geometric-ratio inequality follows too.

Normalizing by S=sum_n exp(-288|n|^2)>=1 costs the subtraction of
(S-1)D^q phi(0), which is included. Thus for EVERY unit-direction contraction,
including mixed directions, through total order6,

    |D^q K_24(0)-D^q K_infty(0)|
       < E := 1458*(76*24^6+15)*10^(-125)
        = 21175738586478 *10^(-125).                      (2)

This is a bound on the infinite image sum, not a sampled orientation or truncated
floating-point comparison. It applies to the exact variance-one normalization.

## 3. Full joint covariance and all orientations

Use the joint vector (G,t,svec H), where svec lists diagonal Hessian entries and
sqrt(2) times independent off-diagonal entries. Its dimension is at most10.
Each covariance entry differs from the reference by at most2E, by (2).
Hence the spectral norm of the difference is at most20E.

The reference Hessian block has eigenvalues d+2 on trace and2 on traceless matrices;
it is independent of the odd derivative block. The only nontrivial odd block is
[[1,-3],[-3,15]], whose eigenvalues are8 +/- sqrt(58). Its smaller eigenvalue exceeds
1/3: subtract I/3 and use first principal minor2/3 and determinant7/9.
All remaining gradient variances are1. Thus C_ref >= I/3 and

    (1-epsilon) C_ref <= C_24 <= (1+epsilon) C_ref,
    epsilon=10^(-108),   since60E<epsilon.                (3)

The inequalities are in the positive-semidefinite ordering and hold in every
orthonormal frame. No rotational invariance is asserted for C_24.

Multiplicative covariance inequalities pass to marginal linear observations and
to their conditional covariances. For the latter, the Schur-complement quadratic
form is the infimum over the conditioned-coordinate vector of the full quadratic
form, so taking infima preserves both inequalities in (3).
Therefore (3) controls the gradient density, the V density, tau^2, and the entire
conditional transverse covariance A|V=0 simultaneously.

## 4. Cone moments do not require perturbing the cone boundary

Let C be a centered covariance in n=m(m+1)/2 independent symmetric-matrix
coordinates, with (1-epsilon)C0<=C<=(1+epsilon)C0. Comparing Gaussian determinants
and inverse quadratic forms gives pointwise density bounds

    [(1-epsilon)/(1+epsilon)]^(n/2) phi_((1-epsilon)C0)
      <= phi_C
      <= [(1+epsilon)/(1-epsilon)]^(n/2) phi_((1+epsilon)C0).

Integrate the nonnegative function h(A)=det(A)^2 1{A<0}. Under covariance scaling
by a>0 its expectation scales by a^m, since h has degree2m and the cone is scale
invariant. Thus the exact conditional cone moment has the same bounds, multiplied
by (1-epsilon)^m D0 and (1+epsilon)^m D0 respectively. No continuity claim about
an indicator at a singular matrix is used.

Combining this with the two d-dimensional densities and tau^(4/3) yields an
integrand ratio between

    (1-epsilon)^a/(1+epsilon)^b and (1+epsilon)^a/(1-epsilon)^b,
    a=m+2/3+n/2,  b=d+n/2.

For d=2,3, a+2b<14 and2a+b<13. Since epsilon is far below1/28, elementary bounds
on log(1+epsilon), -log(1-epsilon) and exp give ratio between1-13epsilon and
1+28epsilon, hence safely between1-32epsilon and1+32epsilon. Angular integration
preserves these multiplicative bounds because every reference integrand is positive.
It follows that

    |c_d,24/c_d,ref -1| < 10^(-106), d=2,3.               (4)

The exact periodic constant is NOT claimed equal to the reference constant.
Equation (4) bounds it and justifies the numerical interval in this note.

## 5. Outward arithmetic and special-function remainder

`coefficient.py` uses only integer and Fraction arithmetic to obtain (1)-(4).
Every interval operation rounds outward to the rational grid10^(-80); negative
values use mathematical floor/ceiling rather than rounding toward zero. Floats and
booleans are rejected as inputs. Roots use exact integer root brackets.

Pi uses Machin's identity16atan(1/5)-4atan(1/239) with alternating-series remainder.
Log uses range reduction and the positive atanh series, with its geometric tail.
Exp uses reduction to[0,1/4], its positive Taylor series and a ratio-bounded tail;
negative arguments use reciprocal intervals.

Gamma(7/6) is obtained by shifting to z=199/6, using Stirling's LOG-gamma expansion
through B20, enclosing the remainder by the positive first omitted B22 term, then
subtracting the32 recurrence logarithms and exponentiating. The rigorous positive-
real remainder rule is NIST DLMF5.11(ii), and recurrence is DLMF5.5(i).
The periodization allowance in (4) is applied to exact rational endpoints BEFORE
the final outward rounding; it is not silently rounded to zero.
The displayed endpoints are outward rational decimal bounds, not rounded guesses.

The supplied unit tests and mutation runs are finite implementation/algebra checks.
They do not constitute formal verification or nonauthor analytic acceptance.
`ENCLOSURE.json` records the computed bounds and explicit scope.

## Review and remaining work

Nonauthor review should independently check the shared trace variance5/3, Rayleigh
cone integration, ordered-pair/gamma normalization inherited from the parent,
all-direction derivative bound, covariance and Schur-complement comparison, and
rational interval implementation. No frozen proof, review grade, RN/24-jet obligation,
P15 assertion or project completion flag is changed.

### External reconnaissance, 24 September 2026

Primary sources read: NIST DLMF5.11, especially equation5.11.1 and positive-real
remainder statement in5.11(ii); DLMF5.5 for the gamma recurrence. These are established
numerical-analysis inputs. The Gaussian covariance/cone and image comparison above
are derived here; no priority or novelty claim follows. Main issue63 and its exact
Drive source provide the project coefficient and conventions.
