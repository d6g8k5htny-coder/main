# Weighted removal of deterministic fold-chart windows: conventional analytic source

Actual author: OpenAI/Codex, agent `/root/universality_review`, coordinated by
`/root`, 8 October 2026. This conventional analytic source incorporates the
complete external coordinate-successor preparation, preserved at 14,478 UTF-8
bytes, SHA256 `52edff0a32322bec3f8250e06341f3767caab439cb8e9b5ca3da823820dae3f8`.
The original preparation and correction history remain in the
[review record](REVIEW.md). This source leaves both PR321 proofs unchanged.
It is not a Lean proof, scientific promotion, model admission, or a change
to an applicability or scientific-status record.

The author previously reviewed the finite Gaussian contact proof and authored
the candidate-lifetime preparation. The author also performed a full-source
nonauthor review of the fold-chart proof. The present derivation therefore has
substantial proof-route and source exposure. Organizational independence is
zero.

Before this file was saved, the same-team agent `/root/benchmark_formal_audit`
checked the determinant estimate giving the moment-only
`O(a_0^2+r^2)` small-curvature bound and the order of the window choices. That
was a bounded check of a communicated derivation. It was not a full review of
the later saved file, nor a check of every displayed constant or the optional
sharper bound. The original preparation subsequently received a full exact-file
nonauthor mathematical review by `/root/final_contract_review`; its physical
coordinate clarification received a separate successor-delta review. Reviews of
this incorporated source bind its own bytes in the companion record. No compiler,
sampler, empirical draw or seed was used for the analytic derivation; repository
publication and scientific admission are separate actions.

## Exact consumed source versions

The finite Gaussian contact proof was read at
[the frozen finite Gaussian contact proof](../finite_gaussian_contact/PROOF.md):

```text
25,367 UTF-8 bytes
SHA256 7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474
```

The fold-chart proof was fully read at
[the frozen rescaled fold-chart proof](../finite_fold_chart/PROOF.md):

```text
21,678 UTF-8 bytes
SHA256 8b0ab87e9516857c1e92483bf9e39c8664af4fa76f72c6a54c4eb729f763fdba
```

Both hashes were freshly confirmed during the derivation. The contact proof's
Sections 3-5 supply the actual finite-pin Gaussian regression, uniform
conditional coefficient moments, whole-field residual decomposition, and
compact-mark positive floor for `Z_r/r^2`. The fold proof supplies the explicit
deterministic theorem on derivative and transverse-curvature windows. The
present proof consumes those exact statements; it does not use an arbitrary
finite-jet independence premise or a global selected-bar identity.

## 1. Fixed field, marks, and normalized law

Use the same actual ideal finite Gaussian field as in the consumed contact
proof. The working case is `gaussian__gaussian`, `K=3`, physical period `L=24`,
on the flat square torus `X=(R/24Z)^2`. Its covariance is

    Cov(F(x),F(y)) = sum_{n in {-3,...,3}^2} q_n exp(i omega n.(x-y)),
    omega=2pi/24,
    q_n=exp(-|n|^2) / sum_{m in {-3,...,3}^2} exp(-|m|^2).

The weights are strictly positive, symmetric, and sum to one. The real field
has an actual finite standard Gaussian coefficient vector `xi` of dimension
49. Equivalently, choose one representative of each nonzero pair `{n,-n}`
and use

    F(x)=sqrt(q_0) xi_0
         +sum_n sqrt(2q_n)[xi_n,c cos(omega n.x)+xi_n,s sin(omega n.x)].

The proof below only needs the already established finite Gaussian inputs for
this fixed law. It does not assert constants uniform over arbitrary spectra.

Fix a compact set `B` of birth marks, a gap-mark interval

    0<k_-<=k<=k_+<infinity,

and `epsilon>0`. At midpoint `x`, supplied orthonormal frame `(u,v)`, and
positive separation `r`, put

    M=x-(r/2)u,  S=x+(r/2)u.

Use the actual divided differences in contact Section 3 and define `Q` to be
the finite coefficient Gaussian regression on

    U_r=v_r=(b-k r^3/2, -k r^2, 0, 12k, 0, 0).

For `r>0`, their exact invertible relation to the original observations means
that, `Q`-almost surely,

    F(M)=b,  F(S)=b-k r^3,  grad F(M)=grad F(S)=0.          (1)

This specifies a Gaussian regression law even though the unconditioned exact
pin event has probability zero. The proof does not treat that event as having
positive probability.

Let `H_M,H_S` be the physical field Hessians in the supplied frame, and define

    W_r=|det H_M| |det H_S|
          1{H_M negative definite}
          1{H_S has exactly one negative eigenvalue},
    Z_r=E_Q W_r,
    V_r=W_r/r^2,  z_r=Z_r/r^2.

The compact-mark normalizer theorem gives constants `r_0>0,z_*>0` such that

    z_r>=z_*                                                     (2)

uniformly for `b in B`, `k in [k_-,k_+]`, all midpoints and supplied frames,
and `0<r<=r_0`. Reduce `r_0` below one. All subsequent moment suprema are over
this initial parameter family, before choosing any of the derivative windows.

Define the normalized weighted law by

    Q^W(E)=E_Q[V_r 1_E]/z_r.                                     (3)

This is a probability law by (2) and finite moments. It gives zero mass to
`W_r=0`. Thus `Q^W`-almost surely, `H_M` is negative definite and `H_S` is
nonsingular with one positive and one negative eigenvalue. In particular,

    A:=F_vv(M)<0  Q^W-almost surely.                              (4)

The possible zero eigenvalue allowed by the wording "one negative eigenvalue"
has zero weight because its determinant vanishes. No claim about other
critical points or global Morse behavior is required for (4).

All events below are Borel events of the actual finite coefficient field. In
particular, the global derivative norms are continuous functions of the
coefficients, because the finite expansion is a continuous linear map into
every fixed finite-order smooth-function norm.

## 2. Uniform moments before and after weighting

Fix the physical coordinates and their Euclidean norms. Define the full
`C^4` norm as the maximum, through order four, of the global suprema of the
Fréchet derivative operator norms, including the order-zero field norm. Set

    R=1+||F||_{C^4(X)}.

Finite support gives a deterministic bound

    R<=C(1+||xi||).

Under `Q`, the coefficient mean is uniformly bounded on the compact mark
family, and its centered covariance is an orthogonal projection bounded
above by the identity. Consequently, for every finite `j>=1`,

    M_j:=sup E_Q R^j < infinity.                                  (5)

If the contact proof's coordinate derivative norm is used, a fixed
finite-dimensional norm-conversion constant gives exactly (5) for the full
Fréchet norm defined here. These are moments of the actual global field, not
unspecified partial-block cap norms.

The exact gradient pins imply that the interval averages of `F_uu` and `F_uv`
are zero. Their derivatives along the interval are bounded by `R`. For each
endpoint `i=M,S`, write

    alpha_i=F_uu(i)/r,  beta_i=F_uv(i)/r,  A_i=F_vv(i).

Subtracting an endpoint value from the corresponding zero average gives

    |alpha_i|<=R/2,  |beta_i|<=R/2,
    |A_i|<=R,  |A_S-A_M|<=rR.                                    (6)

For example,

    |F_uu(M)|
      <=r^-1 integral_M^S R dist(M,t) dt
      =rR/2.

The same computation handles the other endpoint and the mixed derivative.
The transverse difference in (6) uses `F_uvv`, also bounded by the full
third-derivative norm.

Since

    H_i=[[r alpha_i,r beta_i],[r beta_i,A_i]],
    det H_i/r=alpha_i A_i-r beta_i^2,

equation (6), with `r<=1`, gives

    |det H_i|/r<=R|A_i|/2+rR^2/4<=3R^2/4.

Dropping type indicators therefore gives the pointwise bound

    0<=V_r<=R^4.                                                 (7)

In particular `Z_r` is finite. Equations (2),(3),(5),(7) yield

    E_{Q^W} R^p <= M_{p+4}/z_*

for each finite `p>=1`. A useful joint large-curvature and derivative-tail
bound is

    Q^W(R>T)
      <=E_Q[R^4 1{R>T}]/z_*
      <=M_6/(z_* T^2),  T>0.                                    (8)

Since `R` is global, its bound controls every rotated physical box in the
fold theorem, together with `|A_M|`. No probabilistic passage from a moment
to a deterministic bound is being made: the deterministic bound is imposed
on the event whose probability is estimated in (8).

## 3. A moment-only small-transverse-curvature bound

For a positive number `a`, retain the more precise endpoint estimate. Put

    x_1=R|A|/2,  y_1=rR^2/4.

At `M`, the determinant factor is at most `x_1+y_1`. At `S`, (6) gives
`|A_S|<=|A|+rR`, so its factor is at most `x_1+3y_1`. Consequently

    V_r<=(x_1+y_1)(x_1+3y_1)
        <=(x_1+3y_1)^2
        <=R^2 A^2/2+(9/8)r^2 R^4.                                (9)

On `|A|<=a`, averaging (9) and dividing by the scalar normalizer floor gives

    Q^W(|A|<=a)
      <=[a^2 M_2/2+(9/8)r^2 M_4]/z_*.                            (10)

Thus this region has uniformly `O(a^2+r^2)` weighted mass. No inverse of `A`,
no random inverse Hessian moment, and no random inverse normalizer occur.
The positivity needed for division is precisely the compact-mark scalar
floor (2).

## 4. Explicit choice and exact quantifier

Let

    eta=min{epsilon,1/2},
    a_0=min{1/2, sqrt(eta z_*/(2M_2))},
    T=max{1, sqrt(2M_6/(eta z_*))},
    a_1=D_3=D_4=T.                                               (11)

All of these numbers are chosen from the initial uniform moments and
normalizer floor. In particular they are chosen before reducing `r`.
They satisfy `0<a_0<=a_1<infinity`.

Choose a fixed positive geometric radius `r_geom` ensuring the boxes
`x+R_frame diag(r,r^(3/2))(-1,1)^2` are embedded for all midpoints and frames.
Here `R_frame=(u v)` denotes the frame matrix, and is distinct from the
random norm variable `R`. For a square torus one may, for example, take
`r_geom=min{1,L/(4sqrt(2))}`. Then define

    r_*=min{
      r_0, r_geom,
      a_0/(2T), k_-/T, a_0 k_-/T^2,
      sqrt(2eta z_*/(9M_4))
    }>0.                                                         (12)

For every `0<r<=r_*`, the two contributions on the right of (10), with
`a=a_0`, are each at most `eta/4`. Equation (8) is at most `eta/2`. Therefore

    Q^W(R<=T and |A|>=a_0)>=1-eta>=1-epsilon.                      (13)

On this event, the weighted support sign (4) gives

    a_0<=-A_M<=a_1,
    sup_X ||D^3F||<=D_3,
    sup_X ||D^4F||<=D_4.                                         (14)

In particular the physical-box suprema satisfy (14). The radius choice
also ensures

    rD_3<=a_0/2,
    rD_4<=k_-,
    rD_3^2<=a_0 k_-.                                             (15)

With the exact pins (1), the gap marks `k in [k_-,k_+]`, the transverse
window (14), and the derivative bounds (14)-(15), every hypothesis of the
explicit fold-chart theorem holds on this event.

Precisely, for the fixed field law and the fixed compact mark family,

    for every epsilon>0,
      there exist a_0,a_1,D_3,D_4,r_*>0,
        for every b in B, k in [k_-,k_+], midpoint and supplied frame,
          for every 0<r<=r_*,
            Q_{r,b,k,x,u,v}^W(actual fold-window event)>=1-epsilon. (16)

The suprema over failure probabilities are at most `epsilon`. This is a
uniform marginal statement for the family of normalized laws. It does not
assert that one coupled sample satisfies the event simultaneously for all
marks, locations, frames, or radii.

## 5. Optional sharper small-curvature estimate

The actual whole-field Gaussian residual decomposition in contact Section 4
also gives a stronger estimate. Under the original regression law `Q`,

    F=mu_r+B_r(A-a_r)+g_r,

where `g_r` is independent of the scalar `A`, all its global `C^4` moments
are uniformly bounded, and `mu_r`, `a_r`, and all fixed-order derivatives of
`B_r` are uniformly bounded on the compact mark family. It follows that

    E_Q[R^j | A=a]<=C_j(1+|a|)^j.                                 (17)

The conditional scalar Gaussian variance of `A` is bounded below by a
positive constant, so its density is uniformly bounded. Integrating (9),
using (17), over `|a|<=a_0<=1` gives

    E_Q[R^2 A^2 1{|A|<=a_0}]<=C a_0^3,
    E_Q[R^4 1{|A|<=a_0}]<=C a_0.

After division by `z_*`,

    Q^W(|A|<=a_0)<=C(a_0^3+r^2 a_0).                              (18)

The constant here depends on the fixed field law and compact mark family.
The powers come directly from integrating `a^2 da` and `da`, respectively.
Independence is used under `Q`, before applying the determinant weight; no
independence of `g_r` and `A` under `Q^W` is asserted. The moment-only choices
(11)-(12) already prove (16), so (18) is an optional strengthening rather
than an additional premise needed to close the statement.

## 6. Chart consequence and remaining boundaries

On the event in (16), set the fold theorem's windows to
`kappa_-=k_-`, `kappa_+=k_+`, and the constants in (11). Its explicit formulas
then give the actual endpoint normalizers, spatial `C^2` charts and inverses,
a common rescaled source radius `rho>0`, a common rescaled target ball
`B_(rho/2)`, and its stated uniform derivative bounds. The chosen `rho` and
bounds depend on the compact marks, `epsilon`, and the fixed field law.

Transport back to the same physical field has source-ball radius
`rho r^(3/2)` in the Hessian-normalized physical coordinates `u` of the fold
theorem, where physical points are `x_i+P_i_phys u`. The actual spatial
source is the ellipsoid `x_i+R_frame D_r Phat_i B_rho`, with axial scale `r`
and transverse scale `r^(3/2)`. The physical second-derivative and normalizer
bounds have the powers of `r` stated in the consumed fold theorem. This
argument supplies no positive physical radius uniform through the collision.

The fold theorem's continuous parameter dependence remains conditional on
continuity of the actual rescaled field family in the stated `C^4` topology
and continuity of the other input data. Statement (16) neither constructs
a parameter coupling nor establishes joint `C^2` contact regularity. Each
individual ideal finite field is smooth, which suffices for the spatial
chart theorem applied here.

The normalized law is the actual typed determinant-weighted pin regression.
It is not an elder-selection law. This window result does not identify
candidate pairs with bars or supply a global selection, complement, or
once-counted persistence identity. The lower gap cutoff `k_->0` and compact
birth window are essential consumed hypotheses; no all-mark inverse-`Z`
estimate is claimed.

Statement (16) follows from the exact already proved finite Gaussian inputs
recorded above. The [review record](REVIEW.md) separates preparation review,
coordinate-delta review and review of these incorporated bytes. Formal
implementation and scientific admission remain separate obligations; no formal
or scientific acceptance follows from repository incorporation.
