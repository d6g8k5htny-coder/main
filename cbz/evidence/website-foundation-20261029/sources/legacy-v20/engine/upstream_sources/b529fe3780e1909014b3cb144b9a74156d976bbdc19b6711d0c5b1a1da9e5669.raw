# One soft transverse direction: an elder-failure lower event in every fixed dimension

Object: OA-ELDER-DIMENSION-LIFT-20260928-v1.
Author: OpenAI / ChatGPT, foreground research continuation, 28 September 2026.
Disposition: AUTHOR-SIDE MATHEMATICAL CANDIDATE; source-bound nonauthor review required.
Scientific effect NONE. No existing proof, review verdict, registry, lemma flag or prize is changed.

## 1. Model, precise claims, and dependence boundaries

Fix d>=2, L>0, a compact birth interval B, and a compact gap interval
K=[k_-,k_+] with k_->0. Use exactly LP's variance-one periodized Gaussian field,

    Cov(f(x),f(y)) = sum_{n in Z^d} exp(-|x-y+Ln|^2/2)
                    / sum_{n in Z^d} exp(-|Ln|^2/2).

All constants may depend on d,L,B,K. The dimension is fixed: no uniform-in-d
claim is made. All orthonormal pin frames R are allowed, without assuming the
periodized field is invariant under arbitrary rotations. Set u=R e_1 and

    M=-(r/2)u, S=(r/2)u;
    f(M)=b, f(S)=b-k r^3, grad f(M)=grad f(S)=0.

Q_r denotes continuous Gaussian regression on these original 2(d+1) observations.
Let F_j(H)=|det H| for a nonsingular Hessian with j negative eigenvalues, and zero
otherwise. Retain the ORIGINAL determinant tilt and FULL endpoint normalizer:

    W_r=F_d(H_M) F_(d-1)(H_S), Z_r=E_Qr W_r,
    dQ_r^W=(W_r/Z_r)dQ_r.

There is no adjacency conditioning. Let p_r(b,k,R) be the probability that S is
M's global ordinary superlevel elder-death partner, exactly as in LP Section 1.
On its generic locus the death height is the maximin from LP Section 8:

    d_f(M)=sup_{gamma(0)=M, f(gamma(1))>b} min_t f(gamma(t)).       (A1)

The already integrated planar source EL proves an explicit path obstruction in
d=2. This note extends that mechanism to all fixed d. The new issue is the raw
Gaussian mass of a nearly flat transverse direction coupled to d-2 stable
negative directions. Requiring every extra matrix entry to be O(r) would lose
the desired probability power. A Schur-complement coordinate chart avoids that.

**Theorem A (direct lower event).** For all sufficiently small r, uniformly over
b,k,R in the specified ranges, there is an event E_r such that

    Q_r^W(E_r)>=c r^3,
    E_r implies d_f(M)>=b-k r^3/4>f(S).                          (A2)

In particular

    1-p_r(b,k,R)>=c r^3.                                        (A3)

The event can be chosen also to contain two distinct additional critical points
of index d-1 and heights strictly between b-k r^3 and b, within distance C_d r
of the midpoint. If N_r counts all additional critical points in that height
window, then

    Q_r^W(N_r>=2)>=c r^3,
    E_Qr^W[N_r(N_r-1)]>=c r^3.                                 (A4)

The second expectation may be extended-valued; this argument supplies no finite
upper bound. For d>2 it does NOT establish Q(N_r>=2)=Theta(r^3), a uniform lower
bound on Q(N_r>=2 | N_r>=1), or a global first-moment upper bound. Those would
need their own inputs. A torus-wide O(r^5) factorial bound is excluded by (A4).
The event lower bound is likewise an obstruction of order at least r^3 to any
approximation supported solely on empty and singleton configurations.

**Corollary B (separate parent-dependent compositions).** Consuming LP Theorem A
at its stated scope gives, in every fixed d>=2,

    c r^3<=1-p_r<=C r^3.                                        (A5)

Consuming additionally LP's compact-mark density identity, for positive-length
B,K, gives

    nu_cand(ell)-nu_eld(ell)=Theta(ell^(2/3))                     (A6)

for its COMPACT-MARK densities. The leading constant, a numerical c or r_*,
unrestricted-mark differences, and any identification with historical q-symbols
remain outside this statement. The d=3,L=24 specialization changes none of those
limitations. The direct lower (A3) does not consume the parent upper theorem.

**Corollary C (sharp rejected-pair inverse-moment boundary).** Under the same
compact-mark density premises, let the nonselected counting measure count
candidate pairs which are not ordinary elder pairs. Directly on that measure,

    E sum_{nonselected, 0<ell<=t} ell^(-p)
      is finite exactly for p<5/3,                                (A6a)

for sufficiently small t>0. In the finite regime it is Theta(t^(5/3-p)). At
p=5/3 the lower-cutoff divergence is logarithmic; above it it is a power. This
is not subtraction of two infinite individual inverse-moment expectations.

The proof below uses only the positive spectrum/contact-frame and uniform
regression facts explicitly displayed in LP Sections 2-4, elementary Taylor
estimates, a stable negative block, and the original maximin definition. It does
not consume the pending remote pair/distance limits or a SARD-G theorem.

## 2. Complete physical third-order jets after the pin frame

Write the orthonormal local coordinates as (x,y), with x axial and y in R^m,
m=d-1. LP's nonsingular frame U_r is an invertible re-expression of the exact pins
for r>0; it has bounded target v_r and converges, uniformly in covariance and
fixed derivative cross-covariances, to

    U_0=(f,f_x,f_xx,f_xxx,f_y1,f_xy1,...,f_ym,f_xym) at 0.        (A7)

Append the physical midpoint vector J consisting of:

- all independent entries A=D_y^2 f(0) of the transverse Hessian;
- all independent third-order derivatives T_alpha=partial^alpha f(0), |alpha|=3,
  except f_xxx (which is already in U_0).

At r=0 the entries of (U_0,J) are exactly every independent derivative of orders
0 through 3, once each. Their count is binom(d+3,3). No repeated symmetric entries
are presented as independent coordinates.

Every Fourier weight of the periodized covariance is positive. A zero-variance
linear combination of these derivatives would give a polynomial vanishing at
every point of the rotated dual lattice. Pulling back by the invertible rotation
gives a polynomial vanishing on Z^d. Induction on dimension, using the univariate
root theorem on each lattice line, makes it identically zero. Thus the complete
jet covariance is positive definite in every frame.

Continuity of U_r and compactness of O(d) give a uniform covariance floor and
ceiling for (U_r,J), for r in [0,r_1]. Schur complementation then gives a uniform
covariance sandwich for the Gaussian vector J under Q_r. Its mean is bounded
because v_r is bounded. Consequently, for EVERY fixed compact set of physical
J-targets, there is a uniform positive lower bound on its Lebesgue density there.
This is a bound for RAW jets, not for a falsely Gaussian nonlinear eigenvector
or normalized-Schur variable.

## 3. A unit-Jacobian chart around one soft transverse direction

For d=2 use the planar argument, or interpret the empty stable block below with
determinant one. We now give the new construction for d>=3. Write y=(z,w),
w in R^n, n=d-2, and decompose

    A = [[a, v^T], [v, D]],
    sigma=a-v^T D^(-1)v.                                        (A8)

Restrict D to a fixed positive-volume box of symmetric matrices near -I_n, with

    -3I_n/2 < D < -I_n/2.

Restrict v to a sufficiently small fixed box of positive volume, independent of
r, such that q=D^(-1)v satisfies ||q||<=1/9. The possible physical jets remain
bounded on this domain. Use the deterministic linear shear, frozen separately
for each realization of these midpoint jets,

    y = S(z,w) = (z,w-qz),
    S=[[1,0],[-q,I_n]], det S=1,
    S^T A S=diag(sigma,D).                                      (A9)

The full coordinate transformation is L=diag(1,S) after the orthonormal pin
frame. Its determinant has absolute value one. It preserves the axis x and the
original pin points. Both L and its inverse have uniformly bounded norms. It is
not generally orthogonal, and no rotation-invariance argument is applied to it.

Define tilde f(x,z,w)=f(R L(x,z,w)). Here R,L are CONSTANT in spatial variables
within each realization. Thus differentiating tilde f does not differentiate a
randomly varying frame. Its midpoint transverse mixed quadratic terms are zero
exactly, its stable Hessian is D, and its soft second derivative is sigma.

Let tau be the third derivatives of tilde f, except tau_xxx. For fixed v,D the
map from the original third derivatives to tau is invertible. Explicitly

    partial_z(tilde f)= (partial_z-q.partial_w)f,
    partial_w(tilde f)=partial_w f, partial_x(tilde f)=partial_x f.

On symmetric third-derivative coordinates this substitution is triangular with
unit diagonal when ordered by z degree. The pure xxx coordinate is fixed and
uncoupled; deleting it leaves determinant one. This observation is about finite
coordinate transformations, not the law of the resulting random variables.

The map

    (sigma,v,D,tau) -> (a,v,D,T),
    a=sigma+v^T D^(-1)v                                      (A10)

therefore has absolute Jacobian one. Its derivative is block triangular: the
Hessian part has derivative one in sigma and identity in the other coordinates;
the cubic part has determinant one, although it depends on v,D. Only D is
inverted; the chart remains regular when A is singular at sigma=0.

This is the step preventing a dimension-dependent loss. Neither v nor the other
n(n+1)/2 stable entries has a shrinking width. The raw entry a itself need NOT
be O(r): it follows the O(1) surface v^T D^(-1)v within an O(r) band.

## 4. The raw rare set has volume and probability of order r

Fix a small epsilon>0, eventually independent of r, and retain the fixed D,v
boxes. In the chart require

    |sigma+(3k/2)r|<epsilon k r,
    |tau_xxz|<epsilon k,
    |tau_xzz+2k|<epsilon k,
    |tau_zzz|<epsilon k,                                      (A11)

and put EVERY other independent cubic coordinate of tau except xxx in a box of
width 2epsilon k centered at zero. Restricting all these finitely many cubic
entries costs a constant depending on d,epsilon,K, not another power of r.
Let q_3=binom(d+2,3)-1 be their total number. Since (A10) has unit Jacobian, the
raw physical-J volume of this set is exactly

    (2epsilon k r) |D_box| |v_box| (2epsilon k)^(q_3).           (A12)

All inverse-chart targets lie in one fixed compact set: D^(-1),v and the shears
are bounded, sigma is bounded for r<=r_1, and k is compact. Section 2's Gaussian
density floor thus gives

    Q_r{J in this raw set}>=c_J r.                             (A13)

The sheared jets need not be jointly Gaussian. We integrate the RAW Gaussian
density using the explicitly justified Jacobian; we never assign an independent
or rotation-invariant distribution to sigma, v, D or tau. Replacing sigma by
sigma/r without its r Jacobian would be incorrect.

## 5. Remainder control is conditioned after ALL the physical jets

Regress the full smooth field on V=(U_r,J) at each target (v_r,j) in the raw set.
For one unconditioned copy F the conditional realization is

    F(z)+Cov(F(z),V)Cov(V)^(-1)((v_r,j)-V).                       (A14)

The normalized pin rows are integral averages of derivatives of order at most
three. Positive-spectrum rapid Fourier summability supplies uniform derivative
cross-covariance bounds through C^4, and all finite C^4 supremum moments of F.
The joint inverse covariance from Section 2 is bounded. The targets j lie in a
fixed compact set. Therefore, for each fixed finite p,

    sup E[||f||_C4^p | U_r=v_r,J=j] < infinity.                  (A15)

Choose K_4 finite so the conditional probability of ||f||_C4<=K_4 is at least
1/2 at EVERY target in the raw rare set. Integrating (A15) over that set yields

    Q_r{rare jets AND ||f||_C4<=K_4} >= (c_J/2)r.                 (A16)

The bounded shear converts this to a uniform local bound on ||tilde f||_C4.
There is no assertion of independence between the jet event and the remainder.
Subtracting a fixed unconditioned tail from a mass of order r is not used.

The constants can be selected without circularity. Fix the D,v boxes and a
compact raw-target container valid for all epsilon<=1 and small r. Choose a
uniform K_4 from that container. Then choose epsilon for the planar stability
margins, and finally reduce r_* using K_4 and those margins. The positive c_J
may shrink with epsilon, but is fixed independently of r.

## 6. Exact pinned planar normal form on the sheared plane

Restrict to w=0, and put

    F_r(X,Z)=[tilde f(rX,rZ,0)-b]/r^3,
    s=sigma/r, a_3=tau_xxz, beta=tau_xzz, c_3=tau_zzz.

The original axial height and gradient pins, including the transverse gradient
pins in the sheared direction, are still exact. Taylor expansion as in the
planar source gives

    F_r = 2kX^3-3kX/2-k/2+(s/2)Z^2
           +(a_3/2)(X^2-1/4)Z+(beta/2)XZ^2+(c_3/6)Z^3
           + O_C2(B3)(r K_4),                                (A17)

with a dimension-dependent constant absorbing the shear norm. For clarity the
potentially dangerous coefficients follow from the pin identities:

    tilde f_xxx(0)=12k+O(rK_4),
    tilde f_x(0)=-(r^2/8)tilde f_xxx(0)+O(r^3K_4),
    tilde f_z(0)=-(r^2/8)tilde f_xxz(0)+O(r^3K_4),
    tilde f_xx(0), tilde f_xz(0)=O(r^2K_4),
    tilde f(0)-b=-(r^3/24)tilde f_xxx(0)+O(r^4K_4).

The second-derivative z^2 coefficient is sigma, not the raw a. No stable
quadratic term survives on w=0 because (A9) eliminates it exactly.
The term -a_3 Z/8 is necessary for the transverse gradient pins.

On (A11), the polynomial part is O(epsilon k)-close in C^2(B3) to

    G_k(X,Z)=k(2X^3-3X/2-1/2-3Z^2/4-XZ^2).                     (A18)

Choose epsilon and then r_* so the C^0 difference is strictly below k/64 and the
C^2 difference meets the endpoint and saddle stability tolerances below.
The higher-dimensional directions have introduced no extra r^-1 remainder.

## 7. Stable path above the death pin

Use the fixed polygonal path in the sheared plane

    (-1/2,0) -> (-3/4,0) -> (-3/4,9/4) -> (-1,9/4).              (A19)

Dividing by k, the restrictions of G_k to its three consecutive segments are

    -3t^2/16-t^3/32,
    -7/32,
    -7/32+51t/64-9t^2/32-t^3/32,     0<=t<=1.

Their minimum is -7/32 and the final value is 17/64. The first is nonincreasing;
the third has derivative at least 9/64. Every segment is inside B(0,3), since
the maximal endpoint squared norm is 97/16<9 and the ball is convex.
A perturbation below k/64 leaves the path above -15k/64>-k/4 and its endpoint
above k/4. After applying the fixed shear, rotation and physical r scaling, the
path starts at the actual M and ends at a point above b, entirely above
b-k r^3/4. It remains inside an embedded O(r) ball by reducing r_*.

This is a path in the physical torus, not a separatrix claim. It proves the
second statement in (A2) for every field in the event of (A16). The use of a
plane chosen from midpoint jets causes no probabilistic problem: existence of
ONE such path suffices, and the chart is a smooth measurable function of those
jets on the chosen stable-block domain.

## 8. Full endpoint Hessians: only two directions are soft

In the (x,z;w) shear frame, the physical endpoint Hessians have blocks

    H_i^tilde = [[r P_i+O(r^2 K_4), C_i], [C_i^T,D_i]],
    ||C_i||<=C r K_4, D_i=D+O(r K_4),                         (A20)

where P_i is the Hessian of the polynomial in (A17) at the corresponding scaled
endpoint. To verify the cross-block bound, at the midpoint tilde f_zw=0 exactly
by the shear, and tilde f_xw=O(r^2K_4) by the zero endpoint w-gradients. Moving
to either endpoint changes these entries by O(rK_4). The stable block remains
uniformly negative with bounded inverse for small r.

The exact Schur complement is

    K_i=r P_i+O(r^2 K_4)-C_i D_i^(-1) C_i^T,
    K_i/r=P_i+O(r K_4+r K_4^2).                              (A21)

The correction must be included. The fact that it is smaller than r, rather
than zero, is what makes the construction stable. The target two-dimensional
endpoint Hessians are diag(-6k,-k/2) and diag(6k,-5k/2), with determinants 3k^2
and -15k^2. Uniform smallness of epsilon and r preserves their indices and
strict determinant floors after (A21).

Block congruence gives inertia(H_i^tilde)=inertia(D_i)+inertia(K_i), and

    det H_i^tilde = det D_i det K_i.                           (A22)

Hence the physical endpoints have precisely the required indices d and d-1,
and EACH determinant has magnitude at least c r^2. The d-2 stable eigenvalues
remain O(1), not O(r). Because the shear determinant has absolute value one,
physical and sheared Hessian determinants agree; general invertible congruence
preserves their inertia as well. Thus on the event

    W_r >= c_W r^4.                                           (A23)

The r^4 power is independent of fixed d. Replacing it by r^(2d) would correspond
to incorrectly shrinking every spatial direction's curvature.

For the FULL normalizer, the original endpoint gradient pins give zero averages
of H u along the axial segment. Thus ||H_i u||<=C r ||f||_C3. Hadamard yields
|det H_i|<=C r (1+||f||_C3)^d, so

    Z_r <= C_Z r^2                                           (A24)

by the endpoint-conditioned moment bounds of LP Section 4. The positive event
(A16),(A23) ensures Z_r>0. No restricted normalizer is used. Combining gives

    Q_r^W(E_r) >= (c_J c_W/(2C_Z)) r*r^4/r^2 = c r^3.          (A25)

Together with the path, this proves (A2)-(A3), in every fixed d>=3; EL covers
d=2. No parent upper selection theorem entered this argument.

## 9. Two additional saddles with a stable negative complement

A planar restriction critical point is NOT automatically a full-dimensional
critical point. We explicitly correct the stable coordinates.

Near either planar saddle P_+ or P_- of G_k, with

    P_+/-=(-3/4,+/-sqrt(15/8)), G_k(P_+/-)=-7k/32,
    det H_2 G_k(P_+/-)=-15k^2/2,

write x=rX, z=rZ, w=r^2 W. Consider the rescaled FULL gradient map

    H_r(X,Z,W)=r^(-2) grad tilde f(rX,rZ,r^2W).                 (A26)

On bounded (X,Z,W) sets, Taylor and the pin identities give in C^1

    H_r = (grad P(X,Z), D W+Q(X,Z)) + O(r C(K_4)),             (A27)

where P is the two-dimensional polynomial in (A17), and the stable-vector
quadratic polynomial is

    Q_j(X,Z)=(tau_xxwj/2)(X^2-1/4)
                +tau_xzwj XZ+(tau_zzwj/2) Z^2.                (A28)

In particular the -tau_xxwj/8 term comes from tilde f_wj(0), and is retained.
All its coefficients lie in fixed small boxes. The absence of an order-r
linear z term in the stable gradient uses tilde f_zw(0)=0 exactly. The two soft
gradient components have no O(1) dependence on W because their physical mixed
Hessian is O(r); varying w by r^2 W contributes only O(r) after /r^2.
The C^1 estimates require at most the uniform C^4 bound, not an assumed exact
cubic random field.

For epsilon small, P has one nondegenerate saddle near each P_+/- by ordinary
C^2 stability. At such a point p, the limit vector map in (A27) has a zero at
(p,-D^(-1)Q(p)). Its derivative is block triangular with diagonal blocks H_2P(p)
and D, so its inverse is uniformly bounded over the compact jet boxes. These
zeros persist for the C^1-small perturbation (A27), uniformly after reducing r_*.
This can equivalently be obtained by a quantitative inverse-function argument
on fixed disjoint neighborhoods. The solutions have w=O(r^2); they are not
assumed to lie exactly in the w=0 plane.

Their physical heights are b+r^3[G_k(P_+/-)+O(epsilon k)+O(r C(K_4))], strictly
inside the height window for small epsilon,r. Their full Hessian has a negative
stable block and a two-dimensional Schur complement with one negative and one
positive eigenvalue, just as in (A20)-(A22). Both therefore have index d-1.
Their physical positions lie in an O(r) ball and are distinct from each other
and the original pins. This proves the additional event in (A4); the factorial
inequality follows from N(N-1)>=2 on that event.

Uniformity here is on the compact family of jets and stable blocks for a FIXED
d. It does not assert a uniform number of all critical points or the persistence
pairing of either extra saddle. The separate strict path gives the elder
obstruction; mere extra-point existence would not do so.

## 10. Parent-dependent selection and density consequences

LP Theorem A gives the opposite inequality 1-p_r<=C r^3. When consumed with
(A3), it proves (A5) and shows the Gaussian order r^3 is sharp at every fixed
d, not merely for the matrix relaxation or the planar model.

For LP's ordered compact-mark candidate/elder densities, its exact radial ledger
is r A_r(b,k,u) dr db dk d sigma(u), with A_r uniformly positive and bounded,
and ell=k r^3. Consequently the density difference is

    ell^(-1/3)/3 * integral k^(-2/3) A_((ell/k)^(1/3))(b,k,u)
                       [1-p_((ell/k)^(1/3))(b,k,u)] d sigma dk db.

The lower factor (A3) contributes ell/k. The upper factor is supplied separately
by LP Theorem A. Both bounding integrands are proportional to
ell^(2/3) k^(-5/3), integrable and of positive integral on positive-length compact
B,K and the ordinary sphere measure. This proves (A6). Integrating over
0<ell<epsilon yields a compact-window cumulative rejected intensity of order
epsilon^(5/3), as a further composition, not a new unrestricted density claim.

More generally, integrate the nonnegative rejected-pair density directly with
weight ell^(-p). The two-sided bound (A6) compares this expectation with
integral_0^t ell^(2/3-p) d ell. It is finite precisely when 2/3-p>-1, proving
(A6a). At p=5/3, truncation to eta<ell<=t gives Theta(log(t/eta)); if p>5/3 it
gives Theta(eta^(5/3-p)) as eta decreases to zero for fixed small t. In the
finite regime the result is Theta(t^(5/3-p)), not a claimed leading constant.
The ordinary candidate and elder inverse-moment threshold 2/3 from LP Section 12
is different. Here a rejected-pair measure is used, never infinity minus infinity.
For d=2 this is already a consequence of the integrated planar lower and LP;
for d>2 it additionally uses the new dimensional lower proof.

For d>2 we do not import the PLANAR global first-moment estimate (IW I5) to
claim a global multiple-occurrence upper bound or conditional nonuniqueness
bounded away from zero. Only (A4) and the corresponding unconditioned TV lower
obstruction follow without another global upper input. This is intentionally
weaker than those extra planar conclusions in LOCAL_MULTIPLICITY.md.

## 11. What is new, what is checked, and what remains for review

The planar cubic, path and raw-jet conditioning mechanism are credited to the
unchanged EL/LOCAL sources. The new step is the full-dimensional raw Schur chart:
one shrinking scalar band, determinant-one data-dependent shear, all other
coordinates in fixed-volume boxes, and a stable negative complement. The
Gaussian law is never changed to an independently rotated or sheared ensemble.

The exact suite checks nonorthogonal shear cancellation, the chart through a
singular contact Hessian, unit cubic-jet Jacobians, the complete nonduplicated
jet list, a finite polynomial-lattice unisolvence illustration, exact planar
pins, the path, full-dimensional Hessian determinants/inertia, the cross-block
Schur correction, stable-coordinate displacement scaling, and exponent ledgers.
It deliberately rejects replacing r by r^(d-1), r^4 by r^(2d), dropping the raw
band Jacobian, dropping transverse pin terms, and reversing the stable sign.

Finite checks do NOT prove covariance floors, uniform conditional C^4 estimates,
uniform inverse-function neighborhoods or the Gaussian probability inequality.
Those arguments are the written Sections 2-9 and require an actual nonauthor
analytic review. The finite Fourier illustration is not substituted for positivity
of the infinite exact periodized spectrum. The code is a new shear/Schur module,
not recovery or a retry of the blocked author algebra.py from #116.

Source identities, limited primary-source reconnaissance, and the precise role
of each parent are in SOURCE_MAP.json and RECONNAISSANCE.md. This candidate
changes no old source or canonical scientific status and claims no historical
priority. All constants and cutoffs in the Gaussian result remain existential.
