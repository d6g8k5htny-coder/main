# Positive finite Gaussian spectra: the contact and regression interfaces

Actual author: OpenAI/Codex /root, task
01a11cbf-068d-7102-861e-75814e715c98, 8 October 2026.
This proof discharges named analytic interfaces for the ideal finite Gaussian
fields in the [fifty-law catalogue](../models.json). Its review is recorded in
[REVIEW.md](REVIEW.md). It is a source proposal, not a scientific-status
register, a Lean proof, or an admission to the full persistence theorem.

## 1. Source, field and exact scope

The consumed parent P is
[Math-@13af1089fd7991105a3d8828539bdd9a41ca87c3,
UNIFORM_MATRIX_CAP_AND_LIFETIME.md](https://github.com/d6g8k5htny-coder/Math-/blob/13af1089fd7991105a3d8828539bdd9a41ca87c3/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md):
Git blob dfed3b8d318a3ab1950957f393307733a4bef3f2, 40,261 UTF-8 bytes,
SHA256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7.
Root fetched its complete content and independently reproduced that byte count
and SHA256. The proof uses P's literal rows (3.1), target (3.3), transverse
block at M, endpoint identities (5.1)-(5.3), and the unnormalized intensity
convention (13.4). The finite support argument below replaces the named uses
of P §2; it does not inherit §2's arbitrary finite-jet assertion.

Read P with its [D1 reconciliation](https://github.com/d6g8k5htny-coder/Math-/blob/13af1089fd7991105a3d8828539bdd9a41ca87c3/reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md)
and its complete reading-rule identity set. The consumed congruence correction
is [E1](https://github.com/d6g8k5htny-coder/Math-/blob/13af1089fd7991105a3d8828539bdd9a41ca87c3/imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md),
blob 213594d6ca6a86fb938110f4d166d9ce275a02d0, 1,782 bytes,
SHA256 bad7ef609c4ad8c41ad6af562c1b6807921e19a9d556ed793ad1a0db6e202028;
root read and independently checked its complete bytes. D1 also names
[CAP](https://github.com/d6g8k5htny-coder/Math-/blob/13af1089fd7991105a3d8828539bdd9a41ca87c3/imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md),
blob 0633aca3c2a2882b0de4399da0a75d64c2e6b2e1, SHA256
0bf922b9203c29088b12388807aa0e2ecd020485eb0f6e919679841b5b2636fc,
and [E2](https://github.com/d6g8k5htny-coder/Math-/blob/13af1089fd7991105a3d8828539bdd9a41ca87c3/reviews/d1_section9_borel_repair_20260925/REPAIR.md),
blob fe9b9ce4999908bb3814b500ee2d0ceb0c6f704a, SHA256
845abf9f9c99d672c2a10a887b5a2e7206a3d2de3d876f35f75ff6e2dc13e62f,
plus its own W1/embedding correction. CAP and E2 are named dependencies for
the remaining bar-selection/counting obligations, not results transferred to
the finite model by this proof. Their identities here are read from D1.
Our explicit finite-coefficient coupling below supplies its stated convergence
in probability; it does not silently reuse the parent's unamended wording.

Fix L>0, omega=2 pi/L, an integer K>=2, and the square support

    S_K = {-K,...,K}^2.

Take fixed real weights q_n>0 with q_n=q_-n and sum q_n=1.
Let P_K contain one member of each nonzero pair {n,-n}. For independent
standard real Gaussians C_0, C_n, S_n, define the ideal field on the square
torus X=R^2/(L Z^2) by

    F(x) = sqrt(q_0) C_0
         + sum_{n in P_K} sqrt(2 q_n)
             [C_n cos(omega n.x) + S_n sin(omega n.x)].       (1)

There are (2K+1)^2 real Gaussian coordinates. F is centered, stationary and
variance one. Translation rotates each independent sine/cosine pair
orthogonally; this proves stationarity. General spatial rotations need not
preserve its covariance.

At L=24 and K=2 or 3, each of the ten catalogue spectral formulas is strictly
positive and symmetric on this support. Thus the results apply to the ten
ideal Gaussian coefficient laws, including gaussian__gaussian. The constants
can be chosen uniformly across this finite set of twenty fixed profiles.
They are not uniform over all possible positive spectra, K, L or dimension.
No coefficient draw or floating PRNG realization is certified by (1).
The rounded weights retained by models.py and its continuum-admissibility
label remain unchanged.

Let (u,v) be any orthonormal frame, including reflected frames, and x any
basepoint. For 0<r below the torus injectivity scale, put

    M=x-r u/2, S=x+r u/2,
    F(M)=b, F(S)=b-k r^3, grad F(M)=grad F(S)=0, k>0.

The last line denotes continuous Gaussian regression on these observations;
it is not an event of positive probability. All conclusions below specify
the exact conditioning coordinates.

We prove:

* Positive contact covariance of the seven named rows for every frame.
* Uniformly nonsingular small-r divided-difference observations, including
  the transverse Hessian at M.
* Uniform compact-mark Gaussian regression, independent residual and endpoint
  normalizer bounds.
* An integrable unnormalized all-mark intensity envelope.
* Positive original value-and-gradient covariance at every two distinct sites,
  and its uniform bound on a fixed off-diagonal domain.
* Finiteness and positivity of the local coefficient functional.

None of these conclusions supplies the actual-bar selection probability,
a global once-counted persistence identity, higher homology, a process limit,
or a usable numerical asymptotic window.

## 2. The named contact covariance

In frame coordinates, define the UNCONDITIONED observation vector

    J_0 = (F, F_u, F_uu, F_uuu, F_v, F_uv, F_vv)(x).

For real lambda in R^7, let a=n.u and c=n.v. The Fourier multiplier of the
corresponding derivative functional is

    P_lambda(n) =
        lambda_0 + i omega lambda_1 a - omega^2 lambda_2 a^2
        - i omega^3 lambda_3 a^3 + i omega lambda_4 c
        - omega^2 lambda_5 a c - omega^2 lambda_6 c^2.

The covariance from (1) gives exactly

    Var(lambda.J_0) = sum_{n in S_K} q_n |P_lambda(n)|^2.     (2)

This formula includes both members of each mode pair and the constant.
A basepoint phase has modulus one and does not change (2).

If the variance vanishes, every P_lambda(n) vanishes because all q_n>0.
As a polynomial in the original two frequency coordinates, P_lambda has
degree at most three in each coordinate. For every fixed second coordinate
on the grid it has at least five distinct roots in its first coordinate;
all coefficients in the first coordinate vanish on the second grid.
Applying the same root argument in the second coordinate shows that
P_lambda is identically zero. The invertible frame change then gives an
identity in the independent variables a,c. Its homogeneous cubic, quadratic,
linear and constant terms force, respectively,

    lambda_3=0;
    lambda_2=lambda_5=lambda_6=0;
    lambda_1=lambda_4=0;
    lambda_0=0.

Hence Gamma_0=Cov(J_0) is positive definite. Its entries depend continuously
on the frame. Compactness of O(2) gives a positive minimum of its least
eigenvalue. This argument preserves the actual directional covariance;
it never replaces a square-torus covariance by an isotropic one.

In particular the six-row subvector U_0 obtained by dropping F_vv has
positive covariance. The Schur complement proves

    Var(F_vv | U_0)>0.                                    (3)

Under regression U_0=(b,0,0,12k,0,0), the six conditioned coordinates are
deterministic. It would be false to assert positive covariance of all seven
rows under that conditioned law. Equations (2)-(3) concern unconditional
covariance and the remaining conditional scalar, respectively.

## 3. The literal divided differences and their contact limit

Use P's row order, with A_r the transverse curvature at the left pin:

    U_r = (
      [F(M)+F(S)]/2,
      [F(S)-F(M)]/r,
      [F_u(S)-F_u(M)]/r,
      (6/r^2)[F_u(M)+F_u(S)-2(F(S)-F(M))/r],
      [F_v(M)+F_v(S)]/2,
      [F_v(S)-F_v(M)]/r),
    A_r = F_vv(M),  J_r=(U_r,A_r).                         (4)

The exact target is

    v_r=(b-k r^3/2, -k r^2, 0, 12k, 0, 0).                (5)

The original six observations and U_r are related by an invertible linear
map for r>0, with absolute determinant 12 r^-5. No approximate target or
missing factor 12 is used.

Every difference row has an integral representation. With -1/2<=s<=1/2,

    (U_r)_1 = integral F_u(x+r s u) ds,
    (U_r)_2 = integral F_uu(x+r s u) ds,
    (U_r)_5 = integral F_uv(x+r s u) ds,
    (U_r)_3 = 6 integral (1/4-s^2) F_uuu(x+r s u) ds.       (6)

The fourth identity follows by integrating twice by parts:

    integral_M^S (t+r/2)(r/2-t) F_uuu(x+t u) dt
       = r[F_u(M)+F_u(S)]-2[F(S)-F(M)],

where t is measured from x. Its kernel has integral one and is nonnegative.
The other two U rows are endpoint averages. Thus J_r converges to J_0 in
mean square uniformly over basepoints and frames. For example,

    |(U_r)_3-F_uuu(x)| <= (3r/16) ||F||_{C^4,dir}.

The other rows follow from the fundamental theorem of calculus and endpoint
bounds. Here a coordinate C^q norm and its directional version differ by
fixed finite constants.

More explicitly, with xi the finite standard Gaussian coefficient vector,
write J_r=L_r(x,u,v) xi. Equation (6) gives a continuous extension L_0 at r=0
and uniform convergence on the compact set of basepoints and frames.
Consequently Gamma_r=L_r L_r^T converges uniformly to Gamma_0.
There exist r_*>0 and 0<m<=M<infinity such that

    m I <= Gamma_r <= M I,  0<=r<=r_*, all x,(u,v).         (7)

Shrink r_* below one and the injectivity scale. The covariance of U_r has
the same bounds. Its inverse converges uniformly. The Schur variance of
A_r given U_r lies between m and M: its quadratic form is the minimum of
the full quadratic form over the first six coordinates. It converges
uniformly to the contact Schur variance.

This is an existence proof for r_*,m,M. It does not evaluate a certified
radius or eigenvalue enclosure.

## 4. Regression, moments and the independent field residual

Let Q_{r,b,k,u,v} be regression on U_r=v_r. For marks in a compact rectangle

    b in B, 0<k_-<=k<=k_+,

the conditional coefficient mean is

    E_Q xi = L_U^T (L_U L_U^T)^-1 v_r,

and its centered covariance is
I-L_U^T (L_U L_U^T)^-1 L_U, a positive semidefinite orthogonal projection
bounded above by I. Equation (7) bounds the mean uniformly on this rectangle.

For every finite derivative order q, the finite expansion has a deterministic
bound ||F||_{C^q(X)}<=C_q ||xi||. Finite-dimensional Gaussian moments therefore
give, for every finite p>=1,

    sup_{r,x,frame,b in B,k in [k_-,k_+]}
        E_Q (1+||F||_{C^q})^p < infinity.                 (8)

Write mu_r(z)=E_Q F(z), a_r=E_Q A_r, and s_r^2=Var_Q A_r.
The Gaussian regression decomposition is an actual whole-field identity:

    F(z)=mu_r(z)+B_r(z)(A_r-a_r)+g_r(z),
    B_r(z)=Cov_Q(F(z),A_r)/s_r^2.                          (9)

The centered residual coefficient vector is orthogonal to the scalar
coefficient vector producing A_r-a_r. Joint Gaussianity makes the entire
residual coefficient vector, and hence the smooth field g_r, independent
of A_r. This is not a claim that its internal derivatives or evaluations
are independent.

Equation (7), the finite expansion and bounded derivative cross-covariances
bound B_r and all fixed-order derivatives uniformly. All C^q moments of g_r
are uniformly bounded. The law of A_r has variance bounded above and away
from zero and bounded mean, so on the compact mark rectangle

    density_Q(A)<=C exp(-c A^2).

For a fixed coordinate norm, J=1+||g_r||_{C^4} is independent of A_r,
has every finite moment uniformly bounded, and (9) yields

    M_3,M_4 <= C(J+|A_r|),                               (10)

where M_j bounds directional order-j derivatives on the local cylinder.
No inverse field-Hessian moment or division by a normalizer is used.

## 5. Compact-mark endpoint normalizer

Define H_i as the physical Hessian in frame coordinates at i=M,S,
and let

    alpha_i=F_uu(i)/r, beta_i=F_uv(i)/r, A_i=F_vv(i),
    H_i=[[r alpha_i,r beta_i],[r beta_i,A_i]],
    W_r=|det H_M| |det H_S|
          1{H_M negative definite}
          1{H_S has exactly one negative eigenvalue},
    Z_r=E_Q W_r.                                         (11)

All matrices are physical field Hessians, not observation covariances.
The exact pins and integral remainder identities give

    |alpha_i|,|beta_i|<=C M_3,
    |A_S-A_M|<=r M_3,
    |alpha_M+6k|,|alpha_S-6k|<=r M_4/2,
    det H_i/r=alpha_i A_i-r beta_i^2.                     (12)

These are P's endpoint identities specialized to dimension two. The two
gradient pins make the interval averages of F_uu and F_uv zero; subtracting
each average bounds its endpoint by the derivative Lipschitz bound.
The weighted third-derivative identity from (6) and target 12k give the
two axial limits with the displayed error. No simultaneous vector Rolle
zero is assumed.

Let A_0 have the conditional law of F_vv(x) given
U_0=(b,0,0,12k,0,0). Its variance is positive by (3).
Conditional means and covariances of A_r converge uniformly on compact marks.
One can couple the finite conditional Gaussian coefficient vectors using
their covariance square roots and a common standard Gaussian vector.
Continuity of positive semidefinite square roots gives uniform L^p convergence
for each finite p. Equations (8),(12) then show that the congruence-scaled
Hessians converge to

    diag(-6k,A_0), diag(6k,A_0).

Specifically the scaled matrix is diag(r^-1/2,1) H_i diag(r^-1/2,1);
its off-diagonal entry is sqrt(r) beta_i, as required by E1. On k>=k_->0, the limit is
nonsingular except when A_0=0. The uniform bounded Gaussian density gives
sup P(|A_0|<=epsilon)<=C epsilon. The type indicators therefore converge
uniformly in probability, and the limit of their product is 1{A_0<0}.
Uniform integrability follows from (8),(12). Hence

    Z_r/r^2 -> z_0(b,k,u,v)
       =36 k^2 E[A_0^2 1{A_0<0}],                        (13)

uniformly on the compact mark rectangle and frames. z_0 is continuous
and strictly positive; compactness yields a positive minimum. After reducing
r_* for this rectangle, Z_r/r^2 has a positive uniform lower bound.

The positive normalizer in this conclusion is compact-mark and small-r.
It neither certifies a specified band nor supplies a globally uniform
inverse-Z bound as b,k become unbounded.

## 6. The unnormalized all-mark envelope

The radius and covariance bounds in (7) do not depend on b,k.
For every b in R and k>0, equation (5), with r<=1, gives

    |v_r|<=C(|b|+k),  |v_r|^2>=c(b^2+k^2).

For the lower bound, b^2<=2(b-k r^3/2)^2+k^2/2 and the fourth
coordinate is 12k. Thus the six-dimensional Gaussian pin density satisfies

    pi_r(v_r)<=C exp[-c(b^2+k^2)].

The conditional coefficient mean is O(|b|+k) and its centered covariance
is bounded by I, so the finite-expansion proof yields

    E_Q (1+||F||_{C^3})^p <= C_p(1+|b|+k)^p.

Using only the gradient-average bounds in (12), r<=1, and
|A_i|<=C||F||_{C^2}, each |det H_i|/r is at most
C(1+||F||_{C^3})^2. Therefore

    0<=12 pi_r(v_r) Z_r/r^2
       <=C(1+|b|+k)^4 exp[-c(b^2+k^2)] =: H(b,k).         (14)

The product k^-2/3 H(b,k) is integrable over R x (0,infinity).
Near zero the exponent -2/3 is greater than -1; at infinity Gaussian
tails absorb the polynomial. This is exactly the unnormalized
typed-candidate amplitude envelope of P (13.4), with d=2.
It is not an estimate of a conditioned selection probability.
No division by k, by Z_r or by the transverse Hessian occurs in its proof.

## 7. Two distinct sites and the off-diagonal interface

Let x!=y as points of X. The unconditioned vector

    (F(x),grad F(x),F(y),grad F(y))

has positive covariance. To prove it, translate x to zero and choose a
coordinate in which x,y differ modulo L, say the first. Put
t=exp(i omega(y_1-x_1)), so t!=1. A zero-variance real combination would
annihilate every mode and, for each fixed n_2, have the form

    (a+b n_1)+(c+d n_1)t^n_1=0                           (15)

on the mode grid. Its coefficients include the nonzero phase from n_2
and the linear dependence on n_2 of the transverse gradient coefficient.

Four consecutive integers n_1 suffice to distinguish these four sequences.
Indeed the matrix with columns 1,j,t^j,j t^j, j=0,1,2,3, has determinant
t(t-1)^4!=0. Starting at another integer only gives invertible shifts
and nonzero factors. K>=2 supplies such a consecutive block.
Thus a=b=c=d=0 for each fixed n_2. Taking two different n_2 values
then forces both value and both transverse derivative coefficients to
vanish, as well as the two axial coefficients. The combination is zero.

The same argument in the other coordinate covers a difference only there.
This is a full two-site first-jet proof, not a sampled covariance check.
For torus distance at least a fixed rho>0, compactness yields a uniform
positive covariance floor. Conditional finite coefficient moments then
grow polynomially in the original height target (b,b-ell), and its
density decays like exp(-c b^2) for 0<ell<=1.

Consequently the analytic pin-density-times-determinant integrand for the
off-diagonal candidate height density is integrable and bounded uniformly
in ell in (0,1], exactly the analytic input to P §14. Identifying its
integral with a count still requires the applicable marked Kac-Rice
counting statement; this proof does not silently import that statement
or the selected-bar identity.

For K>=3 the bounded argument also proves independence of both full
two-site jets through order two: the two polynomials in n_1 then have
degree at most two, requiring six consecutive frequencies. The confluent
Vandermonde matrix for roots 1,t, each of multiplicity three, is invertible.
Three different n_2 values force every remaining polynomial coefficient
to vanish. This extension must not be asserted for K=2; Section 9 gives
a counterexample.

## 8. The local jet coefficient functional

The parity of the covariance in (1) makes odd-order jets independent
of even-order jets. Let

    G=(F_u,F_v), V=(F_uu,F_uv), T=F_uuu, A=F_vv,
    tau_u^2=Var(T|G=0), s_u^2=Var(A|V=0).

The seven-row covariance proves tau_u^2>0 and s_u^2>0.
Both are continuous in the frame. Conditional A given V=0 is a centered
scalar Gaussian, so its negative-curvature cone moment is exactly

    D_u=E[A^2 1{A<0}|V=0]=s_u^2/2.                      (16)

The local coefficient functional obtained from P's exact contact ledger is

    C_loc = Gamma(7/6)/(24^(1/3) sqrt(pi))
              integral_{S^1} p_G(0) p_V(0)
                       tau_u^(4/3) (s_u^2/2) d sigma(u). (17)

Every factor is finite, continuous and positive, so 0<C_loc<infinity.
The factor 24 in (17) is the source's Jacobian/gamma factor, not L.
The angular integral must retain the actual directional covariance.

Equation (17) follows from parity, integrating the birth mark by Gaussian
disintegration, and P §15's scalar integral. Here d sigma is ordinary
unnormalized arclength on the unit circle, with total mass 2 pi. In particular
A conditioned on F=b,V=0 need not be centered; (16) applies after the birth
disintegration, to A conditioned only on V=0. Equivalently (17) is the finite
positive contact functional

    4 integral k^-2/3 pi_0(v_0) z_0 db dk d sigma(u),

using (13)-(14). These definitions specify the same local functional.
They do not identify C_loc with the coefficient of actual finite bars.
That identification requires the marked count, actual selection and
complement estimates listed below. No numerical enclosure, SIDE24 digits,
or universal model-independent coefficient is claimed.

## 9. Negative boundaries and remaining admission obligations

Finite support cannot inherit P §2's arbitrary finite-jet independence.
For the gaussian__gaussian K=3 field, in a fixed coordinate direction,

    D_x(D_x^2+omega^2)(D_x^2+4 omega^2)
       (D_x^2+9 omega^2) F = 0,

or

    F_xxxxxxx +14 omega^2 F_xxxxx
       +49 omega^4 F_xxx +36 omega^6 F_x = 0.             (18)

Every retained axial frequency is a root. The relation is a literal
premise obstruction to reusing the all-mode proof; it is not a
counterexample to the persistence exponent. The seven named contact
rows remain independent. Unsupported K=1 provides a sharper negative
control: at an axial frame F_uuu=-omega^2 F_u, so those rows are dependent.

At K=2, two sites differing only in x have six univariate observations
F,F_x,F_xx at the two sites but only five axial frequencies. Their
joint covariance is singular. Thus the stronger two-site second-jet
statement requires its own support condition; original first jets and
the seven-row contact result do not license arbitrary residual lists.

At exact contact, regression imposes F_uu=F_uv=0, so the physical Hessian is

    [[0,0],[0,A]],  det H=0.

The [literal C210 source_coordinates interface](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6066554517)
requires the actual source physical Hessian determinant to be nonzero.
It cannot be instantiated at this exact fold contact. Finite-r typed
endpoints may satisfy it, but their eigenvalues degenerate as r tends
to zero; a uniform fold-family chart requires a separate argument.
Positive observation covariance is not a nonzero physical Hessian.

Before any of these catalogue laws is admitted to the full persistence
law, supply source-bound evidence for:

1. The marked Kac-Rice/Borel counting interface on this ideal field.
2. Every enlarged conditioning list actually consumed by cap, remote
   critical-count, collision and factorial-moment proofs, with its own
   finite-support rank argument. P's arbitrary-list premise is unavailable.
3. Actual determinant-weighted local selection at finite r, a once-counted
   global bar identity and the leading-order complement bound.
4. Same-field sampling/coupling, usable remainder/radius/bin windows and
   field-level statistics for a numerical confirmation.
5. Critical Lean statement/source alignment and the project's applicable
   scientific acceptance records.

The proof establishes the named covariance, regression, normalizer and
unnormalized integrability inputs. It does not fill these five obligations,
and it changes no existing applicability array, model status or graph node.


## 10. A finite-r physical Hessian bridge and its falsifier

The nonzero-Hessian hypothesis of C210 can be discharged pointwise on an
explicit derivative-controlled part of the pinned finite field. This is a
conventional hypothesis proof; no Lean instantiation was executed.

Use the actual scalar endpoint bounds (12) with
|alpha_i|,|beta_i|<=M_3/2. Suppose, for numbers a>0, k>0,

    A_M<=-a,
    r M_3<=a/2,  r M_4<=k,  r M_3^2<=a k.                (19)

Then A_S<=-a/2. For A_i<0 the Schur pivot of H_i is

    s_i=r alpha_i+r^2 beta_i^2/(-A_i),
    det H_i=A_i s_i.

At M, (12) and (19) imply

    s_M/r<=-6k+k/2+r M_3^2/(4a)<=-21k/4.

At S the same bounds give s_S/r>=6k-k/2=11k/2.
Thus H_M is strictly negative definite and H_S has one negative eigenvalue;
both physical Hessian determinants are nonzero. More explicitly,

    det H_M >= (21/4) a k r,
    -det H_S >= (11/4) a k r.                            (20)

This uses literal physical derivatives of the conditioned field and actual
criticality from the pins. Fixed positive a,k windows and finite derivative
bounds supply a sufficient small-r band through (19). They supply no
r-independent physical Hessian floor. Exact source-mode encoding, derivative
identification and a Lean application of C210 remain separate formal work.

Height/gradient pins and negative transverse curvature alone do not suffice.
For fixed r,k>0, let z=x+r/2 and consider the deterministic local polynomial

    f(x,y)=b+(20k/r^2)(z^5/5-r z^4/4)-y^2/2.              (21)

Its axial derivative is (20k/r^2) z^3(z-r). At M, z=0,
f=b and grad f=0; the leading axial term is -(5k/r)z^4.
Thus M is a strict degenerate local maximum, with f_xx(M)=0 and
det H_M=0 despite A_M=-1. At S, z=r, f=b-k r^3,
grad f=0 and f_xx(S)=20k r, so S is a saddle with A_S=-1.
The fourth-derivative bound grows like 1/r and (19) fails.
This is a deterministic local falsifier of the proposed implication,
not a positive-probability counterexample for any catalogue Gaussian law.

Even the exact generic cubic fold forbids a uniform ordinary Morse-chart
radius about M as r goes to zero. Take

    g(x,y)=b+2k x^3-(3/2)k r^2 x-k r^3/2-y^2/2.          (22)

Its two critical points are M,S, with H_M=diag(-6k r,-1)
and H_S=diag(6k r,-1). With z=x+r/2 measured from M,

    g(x,0)-b=-z^2(3k r-2k z).

The exact axial Morse coordinate v=z sqrt(3k r-2k z) has derivative
zero at z=r. More generally, a diffeomorphic chart transforming g-b
to a nondegenerate quadratic form has only one critical point in its
domain, so it cannot contain S. Any physical ball about M contained
in that domain therefore has radius at most r.

For the Hessian normalization H_pullback=2J used by C210, choose the
canonical positive axial scaling P_soft=(3k r)^(-1/2). The other
critical point has normalized axial distance sqrt(3k) r^(3/2).
Any normalized source ball contained in a Morse-chart domain has
radius at most that distance. These bounds are for this explicit
canonical normalization and example; they do not evaluate C210's
returned radius or prove a bound for every permissible normalizer.
They demonstrate that a nonzero finite-r hdet cannot justify a
uniform ordinary Morse chart through the fold collision.
A fold-family chart with an explicit rescaling and parameter domain
is a different interface, still to be proved.
