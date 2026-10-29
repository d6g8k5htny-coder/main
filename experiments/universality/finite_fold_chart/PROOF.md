# Uniform derivative-controlled fold charts: research preparation

Actual author: OpenAI/Codex, agent `/root/final_contract_review`, coordinated by
`/root`, 8 October 2026. Incorporated by `/root` in a separate source proposal
after merged PR319. This preserves a new analytic derivation; it is not a Lean
proof, scientific acceptance, or admission of a catalogue law to a full
persistence theorem. No existing applicability array, model status, or graph
node is changed by this source proposal.

Organizational independence of the author and peer checks: zero. The peer
`/root/benchmark_formal_audit` independently checked the matrix branch algebra
and derivative constants within the same agent team before this file was
written. That was a bounded matrix check, not a full saved-file review or
nonauthor acceptance. The [review record](REVIEW.md) separately binds full
saved-preparation and incorporated-source reviews.

## Source exposure and preservation boundary

The final proved C210 source block was consumed from an earlier successful
read of [Math#193 comment 6066554517](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6066554517).
Its handoff-relative path is `lean-project/SourceCoordinates.lean`: 8,161
UTF-8 bytes, 160 lines, SHA256
`b92d2936525bd120f41badb2bd9018714546d986b6ced9c4326bee9e0b1939da`.
The already cached block was independently rehashed in this audit. No new
C210 fetch, owner-checkout read, or compiler execution was performed. The
source exposes `source_coordinates`, its actual physical `hdet` and
actual-source `hcrit` premises, the same-field Taylor constructor, and the
generic `theta`, congruence, and local-coordinate proof. It does not expose
the complete underlying `MorseCongruence` or `TaylorSymLift` implementation.

The literal parent P was read from the local scratch copy
`finite-contact-parent-source.md`: 40,261 bytes, SHA256
`9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`,
Git blob `dfed3b8d318a3ab1950957f393307733a4bef3f2`, pinned at
[Math-@13af1089fd7991105a3d8828539bdd9a41ca87c3](https://github.com/d6g8k5htny-coder/Math-/blob/13af1089fd7991105a3d8828539bdd9a41ca87c3/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md).
This derivation consumes its literal finite height/gradient pins and scalar
endpoint identities, not its candidate global selection or persistence
conclusions. The correct scaled Hessian below is derived directly by the
chain rule. It does not copy P's erroneous `diag(sqrt(r),I)` wording.

The previously reviewed PR319 [finite Gaussian contact proof](../finite_gaussian_contact/PROOF.md) was 25,367
bytes, SHA256
`7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474`.
That proof is frozen and unchanged by this preparation. Its final section
contains the simpler finite-r determinant bridge, quintic falsifier, and
canonical shrinking-radius example. This file is the next proposed analytic
increment and is outside that merged proposal.

No deterministic sample bounds are inferred from uniform Gaussian moments.
The derivative windows below are explicit assumptions on the actual pinned
field realization. No probability estimate for these windows is claimed.

## Theorem and exact assumptions

Work in an actual orthonormal physical frame `(u,v)` about a midpoint `x_*`.
Let `Q=(-1,1)^2`, and let `f` be the actual ideal finite Fourier field, or any
`C^4` field, on a neighborhood of the embedded physical box

    x_* + R D_r Q,
    R=(u v),  D_r=diag(r,r^(3/2)).

For a torus field, this is an embedded local coordinate box. The supplied
frame is part of the parameter data; no rotational invariance of the field
law or global eigenvector choice is assumed. All norms below are Euclidean
vector norms, corresponding matrix operator norms, and full Frechet
derivative operator norms, except where the symmetric-matrix Frobenius norm
is explicitly specified.

Assume `0<r<=1`, the exact finite-separation pins

    M=x_*-(r/2)u,  S=x_*+(r/2)u,
    f(M)=b,  f(S)=b-kappa r^3,
    grad f(M)=grad f(S)=0,

and fixed windows

    0<kappa_-<=kappa<=kappa_+,
    a_0<=-A_M<=a_1,  a_0>0,
    A_M=f_vv(M).

On the physical box suppose

    ||D^3 f||<=D_3,  ||D^4 f||<=D_4,

where `D_3,D_4` are fixed finite nonnegative bounds on all derivatives of
the stated orders. These are not silently identified with unspecified
partial-block cap norms. Require

    r D_3<=a_0/2,
    r D_4<=kappa_-,
    r D_3^2<=a_0 kappa_-.                                (A)

Define the actual rescaled field and its critical points by

    Fhat_r(X,Y)=[f(x_*+R D_r(X,Y))-b]/r^3,
    q_M=(-1/2,0),  q_S=(1/2,0).

Define constants depending only on the fixed windows:

    c_0=sqrt(kappa_-/a_0),
    p=(1+c_0) max{sqrt(8/(21 kappa_-)), 2/sqrt(a_0)},
    Lambda=(sqrt(2)/6) p^3 D_3,
    N_2=(sqrt(2)/12) p^4 D_4,
    rho=min{1/(4p), 1/[16(1+Lambda)]}>0,                  (B)
    K_2=4 Lambda+rho(6 Lambda^2+2 N_2).

For each endpoint `i` there is an explicit real linear isomorphism `Phat_i`
and a `C^2` diffeomorphism

    e_i:B_rho(0) -> e_i(B_rho(0))

such that, with `J_M=-I` and `J_S=diag(1,-1)`,

    Phat_i^T Hess Fhat_r(q_i) Phat_i=2 J_i,
    ||Phat_i||<=p,
    e_i(0)=0,  De_i(0)=I,
    Fhat_r(q_i+Phat_i y)
       =Fhat_r(q_i)+e_i(y)^T J_i e_i(y),                  (C)
    B_(rho/2)(0) is contained in e_i(B_rho(0)).           (D)

The following bounds hold uniformly over actual field realizations, radii,
birth heights, gap marks, midpoints and supplied frames satisfying the
assumptions:

    ||De_i-I||<1/4,
    ||D(e_i^-1)||<=4/3,
    ||D^2 e_i||<=K_2,
    ||D^2(e_i^-1)||<=64 K_2/27.                          (E)

The inverse bounds hold throughout the image, and in particular on the
common target ball `B_(rho/2)`. This is a deterministic local theorem on
explicit derivative-controlled windows. It asserts neither a window
probability nor global pairing.

## 1. Uniform regularity of the actual rescaled field

For `r<=1`, `||D_r||=r`. The chain rule gives

    ||D^3 Fhat_r||<=r^-3 ||D_r||^3 D_3=D_3,
    ||D^4 Fhat_r||<=r^-3 ||D_r||^4 D_4=r D_4<=D_4.

Thus the rescaling loses no derivative in these estimates. Lower derivative
bounds follow from the exact critical pin and endpoint Hessian bounds. A
valid bound on the Hessian throughout `Q` is

    B_2=(13/2) kappa_+ +a_1+a_0/2+3D_3.

Indeed the endpoint Hessian estimate established in Section 2 is at most
`(13/2)kappa_+ +D_3+a_1+a_0/2`, and every point of `Q` is at Euclidean
distance less than two from `q_M`. Integrating the third derivative bound
along that segment adds at most `2D_3`. Since
`Fhat_r(q_M)=0` and `D Fhat_r(q_M)=0`, another integration gives

    ||D Fhat_r||<=2B_2,  |Fhat_r|<=2B_2.

Consequently its full `C^4` norm, defined as the maximum of derivative
suprema through order four, is at most `max{2B_2,D_3,D_4}`. All segments used
here remain in the convex box. These are bounds on the actual finite-pinned
field, with no contact constraints imposed at finite `r`.

## 2. Explicit endpoint normalizers and their bounds

Write the physical Hessians in the supplied frame as

    H_i=[[r alpha_i,r beta_i],[r beta_i,A_i]],
    alpha_i=f_uu(i)/r,  beta_i=f_uv(i)/r,  A_i=f_vv(i).

The scalar pin identities give

    |alpha_M+6kappa|, |alpha_S-6kappa|<=rD_4/2,
    |beta_i|<=D_3/2,
    |A_S-A_M|<=rD_3.

The average of `f_uv` on the pin interval is zero, since both transverse
gradients are zero. Subtracting this average from an endpoint value and
integrating the derivative Lipschitz bound gives `|f_uv(i)|<=rD_3/2`.
The analogous axial average and the height identity give the displayed
axial estimates. The third-derivative weighted mean is the finite pin
identity equal to `12kappa`, not a substituted midpoint condition.

The rescaled endpoint Hessian is exactly

    Hhat_i=D_r^T H_i D_r/r^3
          =[[alpha_i,C_i],[C_i,A_i]],
    C_i=sqrt(r) beta_i.

Put `B_i=-A_i` and `tau_i=alpha_i+C_i^2/B_i`. Conditions (A) imply

    a_0/2<=B_i<=a_1+a_0/2,
    tau_M<=-(21/4)kappa,
    tau_S>=(11/2)kappa,
    |tau_i|<=7kappa_+.

For the maximum, the positive mixed-square correction is at most
`rD_3^2/(4a_0)<=kappa_-/4`; its axial error is at most `kappa_-/2`.
For the saddle the correction is nonnegative and at most
`rD_3^2/(2a_0)<=kappa_-/2`. These observations prove all the stated pivot
bounds. In particular both physical endpoint determinants are nonzero,
the maximum is negative definite, and the saddle has one positive and one
negative eigenvalue.

Define

    T_i=[[1,0],[-C_i/B_i,1]],
    Hhat_i=T_i^T diag(tau_i,-B_i) T_i,
    Phat_i=T_i^-1 diag(sqrt(2/|tau_i|),sqrt(2/B_i)).

This proves `Phat_i^T Hhat_i Phat_i=2J_i`. Moreover

    |C_i/B_i|<=sqrt(r)D_3/a_0<=c_0,

so `||T_i||,||T_i^-1||<=1+c_0` and the bound `||Phat_i||<=p` in (B)
follows. There is also a uniform inverse bound

    ||Phat_i^-1||<=
       (1+c_0) max{sqrt(7kappa_+/2),
                    sqrt((a_1+a_0/2)/2)}.                (F)

In particular `||Hhat_i^-1||<=p^2/2`, so the rescaled endpoint Hessians
have a uniform spectral gap. The normalizer uses actual scalar pivots in
the supplied frame and requires no eigenvector selection or field-law
orientation invariance.

## 3. The actual integral Taylor matrix

Fix an endpoint and abbreviate `Phat_i=P`, `J_i=J`. Define

    M(y)=integral_0^1 (1-t)
         P^T Hess Fhat_r(q_i+tPy) P dt.

Since `p rho<=1/4`, all the integration segments for `|y|<=rho` stay inside
`Q`: the critical points have boundary distance `1/2`. Exact criticality
and Taylor's integral formula give

    Fhat_r(q_i+Py)-Fhat_r(q_i)=y^T M(y)y.                 (G)

Also `M(0)=P^T Hhat_i P/2=J`. The matrix field is `C^2` because the actual
field is `C^4`. For the symmetric-matrix Frobenius norm,

    ||DM(y)||<=sqrt(2) p^3 D_3 integral_0^1 (1-t)t dt
             =Lambda,
    ||D^2M(y)||<=sqrt(2) p^4 D_4 integral_0^1 (1-t)t^2 dt
               =N_2.

The factor `sqrt(2)` converts a two-dimensional symmetric bilinear-form
operator bound to Frobenius norm. The two integrals are `1/6` and `1/12`.
Consequently

    ||M(y)-J||_F<=Lambda |y|<1/16  for |y|<=rho.          (H)

No assumption `DM(0)=0` is used.

## 4. Explicit congruence compatible with the theta structure

For `J=diag(sigma,-1)`, `sigma` either `-1` or `1`, and symmetric `M` with
`||M-J||_F<=1/8`, define

    Z=JM,
    s_0=sqrt(det Z),
    t_0=sqrt(tr Z+2s_0),
    R_theta(M)=(Z+s_0 I)/t_0.                            (I)

The symbol `R_theta` is a new explicit analytic branch in this preparation;
it is not an identification with the particular branch selected by the
existing C210 implementation. On the stated neighborhood,

    3/4<=det Z<=41/32,

so both square roots and the denominator are positive and the formula is
smooth. Cayley-Hamilton gives

    (Z+s_0 I)^2=(tr Z+2s_0)Z,
    R_theta(M)^2=Z.

Since `Z^T J=JZ`, the same identity holds for the polynomial expression
`R_theta`; hence

    R_theta(M)^T J R_theta(M)=J R_theta(M)^2=JZ=M.        (J)

At `M=J`, `R_theta(J)=I`. Define

    s_theta(M)=J(R_theta(M)-I).

Then `s_theta(M)` is symmetric, `s_theta(J)=0`, and

    R_theta(M)=I+J s_theta(M),
    M-J=2s_theta(M)+s_theta(M)J s_theta(M).

For symmetric Frobenius inputs and output operator norm, safe explicit
bounds are

    ||DR_theta(M)||<=2,
    ||D^2R_theta(M)||<=6.                                (K)

Here are the scalar estimates proving them. On this neighborhood,

    s_0>=3/4,  s_0<=8/7,  t_0>=7/4,
    ||Z||op<=9/8,
    |D det Z|<=8/5,  |D^2 det Z|<=1,
    |Ds_0|<=16/15,  |D^2s_0|<=9/4,
    |Dt_0|<=21/20,  |D^2t_0|<=2,
    ||Z+s_0 I||op<=127/56,
    ||D(Z+s_0 I)||op<=31/15.

For example the first determinant derivative is bounded by
`||adj Z||_F=||Z||_F<=sqrt(2)+1/8<8/5`. Its second derivative is the
two-by-two determinant bilinear form, of norm at most one. Differentiating
the two scalar square roots yields

    |D^2s_0|<=1474/675<9/4,
    |Dt_0|<=109/105<21/20,
    |D^2t_0|<=146749/77175<2.

The first and second quotient derivatives in (I), using these inequalities,
are bounded respectively by

    2879/1470<2,
    43878/8575<6.

Thus (K) follows without a sampled matrix calculation.

For comparison an upper-triangular LDL branch is also explicit. Write

    M=[[a,b],[b,c]],  d=c-b^2/a,
    R_LDL(M)=[[sqrt(sigma a),sigma b/sqrt(sigma a)],
              [0,sqrt(-d)]].

On the same neighborhood `sigma a>=7/8` and `-d>=6/7`. Direct multiplication
gives `R_LDL^T J R_LDL=M` and `R_LDL(J)=I`. Its safe bounds are

    ||DR_LDL||<=1,  ||D^2R_LDL||<=4.

For the first bound, on a symmetric perturbation
`E=[[alpha,beta],[beta,gamma]]`, the absolute output-entry derivatives are
bounded by the following matrix acting on
`(|alpha|,sqrt(2)|beta|,|gamma|)`:

    [[4/7,0,0],
     [32/343,6/7,0],
     [1/84,1/8,7/12]].

Its maximum row sum is `326/343` and maximum column sum is `55/56`, both
below one, so its Euclidean operator norm is below one. For the second
bound, let `s=sqrt(sigma a)`, `w=sigma b/s`, `z=-c+b^2/a`, `q=sqrt(z)`.
For unit symmetric-Frobenius perturbations, direct differentiation gives
`|D^2s|<1/2`, `|D^2w|<2`, `|Dz|<=3/2`, `|D^2z|<=2`, and hence
`|D^2q|<=7/6+3087/3456<3`. The output Frobenius norm is therefore at most
`sqrt((1/2)^2+2^2+3^2)<4`. The theorem uses (I), since that branch also
has the symmetric `s_theta` and literal structural form `I+Js_theta`.

## 5. Chart, common inverse ball, and C2 estimates

Define

    e(y)=R_theta(M(y))y.

Equations (G) and (J) prove the exact energy identity (C). At the origin,
`e(0)=0` and `De(0)=I`, since `M(0)=J` and the derivative of the matrix
factor is multiplied by `y=0`.

Integrating (K) from `J` to `M(y)` and using (H) gives

    ||R_theta(M(y))-I||<=2Lambda |y|.

Differentiating the product yields

    ||De(y)-I||<=4Lambda |y|
                <=4Lambda rho<1/4.                     (L)

The radius choice in (B) makes the last inequality strict. Therefore
`e-I` is Lipschitz with constant `q<1/4` on the convex ball, and

    (3/4)|y-z|<=|e(y)-e(z)|<=(5/4)|y-z|.

In particular `e` is injective. For `|v|<rho/2`, solve `e(y)=v` using the
contraction

    y -> v-(e(y)-y)

on the closed radius-rho ball. It maps that ball into the radius-`3rho/4`
ball and has contraction constant below `1/4`. It has a unique fixed point
with

    |y|<=(4/3)|v|<2rho/3.

This proves the common inverse-image ball (D). The derivative bound (L)
makes every derivative invertible. The inverse function theorem therefore
makes `e(B_rho)` open and its inverse `C^2` throughout that image. The
injectivity above identifies the local inverse charts with one inverse.

A second differentiation of the product gives

    ||D^2e||<=4Lambda+rho(6Lambda^2+2N_2)=K_2.

The first and second derivatives of `e(e^-1(v))=v` then give

    ||D(e^-1)||<=4/3,
    ||D^2(e^-1)||<=(4/3)^3 K_2.

This proves all estimates in (E). Neither nonsingular pin covariance nor
Gaussian moment bounds are substituted for these deterministic estimates.

## 6. Parameter regularity and the contact caution

The explicit endpoint normalizer, integral Taylor matrix, and branch (I)
are continuous functions of the actual rescaled field in `C^4`, of endpoint
data, and of the supplied frame. The pivot signs are fixed by (A), so there
is no branch change. On the common target ball `B_(rho/2)`, continuity of
the inverse follows from the uniform contraction bound; its first and
second derivative formulas then give continuity in `C^2`.

Precisely, a family whose actual rescaled fields and other input data vary
continuously in the stated norms has continuous parameter dependence of
these spatial `C^2` charts and inverses on the common domains. If the
corresponding data have stronger joint smoothness, the explicit formulas
and parameter inverse function theorem give the corresponding local joint
smoothness. This is an implication from actual input regularity, not a new
axiom about the field coupling.

Uniform spatial `C^2` bounds plus continuous parameter dependence do not
imply joint `C^2` dependence in `r` through `r=0`. The rescaled Hessian has
mixed entry `sqrt(r) beta_i`, and anisotropic rescaling can introduce that
regularity obstruction. The theorem is stated for `r>0`. A contact extension
in a parameter such as `tau=sqrt(r)` requires proof of the appropriate
joint regularity of the actual field family; changing the parameter alone
does not prove that assumption. No such extension is asserted here.

Uniform Gaussian `C^q` moments also do not imply deterministic uniform
sample bounds `D_3,D_4`. This theorem applies on the derivative-controlled
window specified in its assumptions; probabilistic window removal remains
separate work.

## 7. Exact transport to the same physical field

Define

    P_i_phys=r^(-3/2) R D_r Phat_i,
    E_i(u)=r^(3/2) e_i(u/r^(3/2)).

The physical Hessian in fixed coordinates is transported by the supplied
frame `R`, so

    P_i_phys^T Hess_phys f(x_i) P_i_phys=2J_i.

Using (C) and the degree-two homogeneity of the quadratic form gives

    f(x_i+P_i_phys u)
       =f(x_i)+E_i(u)^T J_i E_i(u)                      (M)

on `B_(rho r^(3/2))`. Its image contains
`B_(rho r^(3/2)/2)`, and the inverse energy identity holds there. This uses
the actual endpoint value `f(x_i)`, which is `b-kappa r^3` at the saddle,
not a substituted common birth value.

The bounds transport exactly:

    DE_i(0)=I,  ||DE_i-I||<1/4,
    ||D^2E_i||<=r^(-3/2)K_2,
    ||D(E_i^-1)||<=4/3,
    ||D^2(E_i^-1)||<=r^(-3/2)(64/27)K_2,
    ||P_i_phys||<=p r^(-1/2).

The physical source is the actual field on the ellipsoid
`x_i + R D_r Phat_i B_rho`, with axial scale `r` and transverse scale `r^(3/2)`.
The uniform inverse bound (F) supplies a corresponding inner rescaled
ellipsoid. These are chosen height-window scales, not a claim about every
maximal physical Morse domain.

For the same integral Taylor construction,

    M_i_phys(u)=M_i_rescaled(u/r^(3/2)).

This follows by inserting `P_i_phys` in the physical integral Taylor
matrix and applying the Hessian chain rule; the factor `r^-3` cancels the
rescaled height factor. Therefore (M) has the exact structural expression

    E_i(u)=[I+J_i s_theta(M_i_phys(u))]u.

No new realization, coefficient vector, unrelated function, or renamed
source is introduced by this transport.

## 8. Structural theta compatibility and literal branch limits

The construction has the algebraic structure of the cached
`SourceCoordinates.theta` definition. It does not establish equality to
the particular branch, local inverse, or domains selected by the existing
`MorseCongruence` proof or the returned C210 witness.

There is a useful analytic local uniqueness statement. If symmetric `s,t`
solve the same equation

    M-J=2s+sJs=2t+tJt,

then, writing `d=s-t`,

    2d+dJs+tJd=0,
    2||d||<= (||s||+||t||)||d||.

Thus `s=t` whenever `||s||+||t||<2`. Our explicit branch is small. An
existing continuous branch with value zero at `J` therefore matches it
on some sufficiently small common neighborhood. That observation does
not reveal or quantify the existing branch's actual neighborhood or the
chosen inverse source. It supplies no bound for the radius returned by
C210. Equality to that literal chosen witness and its domains remains a
formal alignment obligation.

For the explicit preparation branch, one can regard `s_theta` as a function
on the open Frobenius ball about `J`, or extend it as a total symmetric
matrix-valued function outside that neighborhood. Only its smoothness and
identities on that open ball are used. Such an extension changes none of
the chart or energy statements on the proved source.

## 9. Formal API consumption and remaining boundaries

The actually exposed reusable lemma names are

- `SourceCoordinates.quadratic_congruence`;
- `SourceCoordinates.apply_hasFDerivAt_zero`;
- `TaylorSymLift.sourceSymMatrix_quadratic`.

A strengthened congruence-chart constructor must add the explicit radius,
contraction and inverse bounds above to the generic local-coordinate
construction. To consume the actual I4 source, it also needs:

1. Exact finite source-mode encoding and equality to the physical finite
   field, with the actual period, frame and coefficient conventions.
2. Physical derivative/Hessian identification for that same source and the
   exact critical pins.
3. A verified equality between `TaylorSymLift.sourceSymMatrix` and the
   integral Taylor matrix used in this preparation, or derivative bounds
   derived from its exposed actual definition.
4. The quantitative chart constructor, with the explicit chosen branch and
   proved source/target domains. Identification with a pre-existing chosen
   C210 branch or inverse must be proved if that identification is used.

Finite support supplies summability after the exact source-mode encoding
is provided. `TaylorSymLift.sourceSymMatrix_quadratic` then supplies the
actual-source energy identity. `SourceCoordinates.exists_coordinates`
already gives existential local coordinates, but its current exposed
statement does not provide the quantitative radius proved here.

No underlying formal implementation was fetched for this preparation, and
no Lean application or theorem compiling these new bounds was executed.
The result remains an analytic proof with named formal alignment work.

The exact-contact physical Hessian is still singular. The theorem constructs
uniform charts after rescaling at the two finite-r critical points; it does
not construct an ordinary Morse chart at the degenerate contact itself.
It also supplies no marked Kac-Rice/Borel count, window-tail removal,
determinant-weighted selection, global once-counted bar identity, complement
bound, blind confirmation, coefficient enclosure, or scientific admission.
