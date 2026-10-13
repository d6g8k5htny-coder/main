# Contact-conditioned global genericity for a finite Gaussian field — conventional analytic source

Prepared on 8 October 2026 by OpenAI/Codex `/root/benchmark_formal_audit`,
at `/root`'s request. This conventional analytic source incorporates the external
preparation preserved at 23,103 UTF-8 bytes, SHA256
`a06ec513ce4b83bdeb1e600787a0acfcf9bb20a217ab6009bb326df8c8fd6fe5`.
Its preparation and successor reviews are recorded in the
[review record](REVIEW.md). It changes neither the frozen finite Gaussian contact
source nor the frozen PR321 candidate/chart sources, and changes no
scientific-status or applicability record. It is not a Lean proof, numerical
experiment, persistence-bar selection proof, or assertion of organizationally
independent acceptance.

The author implemented the catalogue machinery and contributed earlier
mathematical input to the contact, candidate-count and fold-chart arguments.
This derivation is source-exposed and within the same agent team;
organizational-independence credit is zero. The algebra and proof below were
checked by the author before saving. Those checks are distinct from the
subsequent full exact-file nonauthor preparation reviews and the separately bound
incorporated-source reviews in the [companion record](REVIEW.md).

## 1. Exact source exposure and law

The actual consumed contact proof was read and independently rehashed at

[the frozen finite Gaussian contact proof](../finite_gaussian_contact/PROOF.md):
25,367 UTF-8 bytes, 554 lines, SHA-256
`7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474`.
Call these frozen bytes FG. Its Sections 1–3 supply the exact field convention,
positive seven-row contact covariance, its six-row subvector and canonical
conditioning; its Section 7 supplies the earlier unconditional two-site ranks.
This proof does not infer conditional ranks from those unconditional ranks.
It proves the new conditional rank statements below directly.

The implementation identities were freshly rehashed:

* `experiments/universality/models.py`: 12,385 bytes, SHA-256
  `7ba088158b5e97ba6e7d715e66e6f9d14268ac3a82a6d187a776821936cc6686`.
  Lines 22, 31–41, 55–69 and 153–201 specify the spectrum/coefficient convention.
* `experiments/universality/models.json`: 12,914 bytes, SHA-256
  `10d334b27cac985d61506b4f26118699cd1689a82a0f6e454cef8d00df21e595`.

The literal parent P had earlier been completely read from the author's preserved
external scratch `finite-contact-parent-source.md`: 40,261 bytes, SHA-256
`9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`,
Git blob `dfed3b8d318a3ab1950957f393307733a4bef3f2`,
[pinned at Math- 13af1089fd7991105a3d8828539bdd9a41ca87c3](https://github.com/d6g8k5htny-coder/Math-/blob/13af1089fd7991105a3d8828539bdd9a41ca87c3/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md).
Its contact target is consumed through FG's literal row contract, not through
P's stronger arbitrary-finite-jet premise or its global selection conclusion.
No new source fetch, compiler run, field draw or seed consumption was performed
for this preparation.

Fix an integer K>=2, L>0, omega=2*pi/L, and the full square frequency support

    S_K={-K,...,K}^2.

Fix real q_n>0 with q_n=q_-n and sum q_n=1. For one representative from each
nonzero pair {n,-n}, take independent standard real Gaussian coordinates
C_0,C_n,S_n and put

    F(z)=sqrt(q_0) C_0
       + sum_pairs sqrt(2q_n)
           [C_n cos(omega n.z)+S_n sin(omega n.z)].           (1)

The coefficient vector xi has dimension N=(2K+1)^2. This is an ideal continuous
Gaussian law on the square flat torus X=R^2/(L Z^2), not a claim about the
finite-word sampler. The specific intended gaussian__gaussian K3 side24 case
has L=24, omega=pi/12 and exact mathematical weights

    q_n=exp(-(n_x^2+n_y^2))/sum_{m in S_3} exp(-(m_x^2+m_y^2)).

The proof uses positivity and symmetry only, so it also covers each fixed
positive symmetric spectrum with ideal Gaussian coefficients. It does not
certify the floating normalization or sampler as a realization of (1).
The catalogue software accepts only cutoffs 2 and 3; the mathematical statement
for arbitrary fixed integer K>=2 does not enlarge that software contract.

Fix x in X, an actual orthonormal frame (u,v), b in R and k>0. Condition on
exactly the six contact rows

    U_0=(F,F_u,F_uu,F_uuu,F_v,F_uv)(x)
       =(b,0,0,12k,0,0).                                  (2)

These are physical directional derivatives, with their omega factors.
This is canonical Gaussian regression, not an event of positive probability.
Let A be the six-row coefficient matrix. FG proves rank A=6 and positive
covariance of (U_0,F_vv(x)) for every frame. If B is an orthonormal matrix
whose columns span ker A, then the conditional field has coefficients

    xi=m+B zeta,
    m=A^T(AA^T)^(-1)(b,0,0,12k,0,0)^T,
    zeta~N(0,I_d),  d=N-6.                                 (3)

All reasoning about probability below takes place in this full-density
R^d residual coordinate space. In particular d=19 at K2 and 43 at K3.

## 2. Precise result and quantifiers

For every fixed law (1), target (2), basepoint and supplied frame, under its
canonical conditional law the following statements hold with probability one:

1. The known contact x is an isolated critical point, with Hessian
   diag(0,A_0) in the supplied frame, A_0=F_vv(x)!=0, and F_uuu(x)=12k.
2. Every other critical point of the actual conditional field is Morse.
3. The whole critical set is finite.
4. All other critical values are mutually distinct and are distinct from b.

Thus the contact is the one forced degenerate critical point. This does not
say that the entire conditional field is Morse. No simultaneous probability-one
event over an uncountable collection of different targets or frames is claimed.
Each fixed conditional law has the stated conclusion for all its spatial
points and pairs at once. No exceptional generic frame is excluded.

At K>=3, conditional gradient covariance is positive definite at every p!=x,
and the conditional variance of F(p)-F(q) is positive for every p!=q.
At K2 each statement has genuine covariance exceptions, classified below.
At those exceptions the target's nonzero cubic pin forces a nonzero derivative
or a nonzero height gap, respectively. Those are inhomogeneous constraints;
calling their covariance positive would be false.

## 3. Real rows, complex modal multipliers and residual variance

By translation of the sine/cosine pairs one may take x=0 for algebra. This is
stationarity, not rotational isotropy: u and v still enter the actual modal
polynomials. Translation is an invertible orthogonal transformation on each
Gaussian pair and preserves the law.

For a real linear observation ell, its conditional variance under (3) is zero
if and only if its real coefficient row belongs to the real row span of A.
Since all q_n are positive, the mode-weight map is invertible. The conversion
from real sine/cosine rows to conjugate complex rows on the full S_K is also
invertible over the appropriate real vector space. Consequently a real
row-span identity implies the corresponding identity of complex multipliers
on every n in S_K. Complex identities below are used first as necessary
conditions. At the listed exceptions we verify the actual real row identity,
so no reality or parity inference is hidden.

Writing a=n.u and c=n.v, a real pin combination with coefficients lambda has
the exact multiplier

    lambda_F+i omega lambda_u a-omega^2 lambda_uu a^2
      -i omega^3 lambda_uuu a^3+i omega lambda_v c
      -omega^2 lambda_uv ac.

Thus the complex span of the pin multipliers, after absorbing the nonzero
powers of i and omega into the coefficients, is

    P_0=span_C{1,a,a^2,a^3,c,ac}.                           (4)

Every member is a polynomial of degree at most 3 in each original frequency
coordinate. Polynomial restriction to the K2 grid is injective for this class:
five distinct roots in the first coordinate, followed by five in the second,
force a polynomial of coordinate degrees at most 3 to vanish identically.
This also holds on every larger square support. Homogeneous cubic terms in
(4) are multiples of a^3 only. This last fact will determine the exceptional
frame orientation, rather than rotating the square law.

## 4. Conditional gradient rank at K>=3: constructive proof

For any p!=x choose a coordinate i for which p_i!=x_i modulo L and define

    H(z)=[1-cos(omega(z_i-x_i))]^2.

It vanishes to order at least 4 at x, so it and all contact derivatives in (2)
vanish at x. It has H(p)>0 and frequency degree 2 in coordinate i.
For j=1,2 the real trigonometric polynomials

    h_j(z)=H(z) sin(omega(z_j-p_j))                         (5)

have support in {-3,...,3}^2, vanish in all six pin rows, and satisfy

    grad h_j(p)=omega H(p) e_j.

They are therefore genuine residual directions of (1) at every K>=3.
The two gradient observations are independent on ker A. Conditional gradient
covariance is positive definite at every p!=x. The same construction plus H
also supplies independent value/gradient directions at p, although only the
conditional gradient rank is consumed by the Morse proof.

## 5. Conditional gradient exceptions at K2: exact fourth differences

Suppose a nonzero real w=(alpha,beta) has D_w F(p) in the pin row span. Let
h=p-x and t_i=exp(i omega h_i). Its multiplier, divided by i omega, has form

    (alpha j+beta l) t_1^j t_2^l=P(j,l),
    P in P_0,  -2<=j,l<=2.                                 (6)

For forward difference Delta in j,

    Delta^4[(alpha j+c)t^j]
      =t^j(t-1)^3[(t-1)(alpha j+c)+4alpha t].               (7)

If t_1!=1, evaluating (7) at the single available start j=-2 and using
Delta^4 P=0 gives, for every l,

    2alpha(t_1+1)+beta l(t_1-1)=0.

The cases l=0 and1 imply beta=0 and alpha(t_1+1)=0. Nonzero w therefore
forces alpha!=0 and t_1=-1. Applying fourth differences in l to (6) then
forces t_2=1. Thus p=x+(L/2)e_1 and w is parallel to e_1. The coordinate-
interchanged argument gives the only other possibility:
p=x+(L/2)e_2 and w parallel to e_2.

On j=-2,...,2 the exact interpolation identity is

    j(-1)^j=-(5/3)j+(2/3)j^3.                              (8)

Uniqueness of polynomial restriction makes the polynomial in (6) a nonzero
multiple of the right side of (8), independent of the other coordinate.
Its cubic part can belong to (4) only if u is parallel to that coordinate
axis. This proves there are no further exceptional frames, points or deficient
row directions. At such an axial frame the conditional gradient covariance
has rank 1, not rank 0, because the only deficient row direction is axial.

Restoring the actual i and omega factors yields the real identity

    F_u(x+(L/2)u)
      =-(5/3)F_u(x)-(2/(3omega^2))F_uuu(x)
      =-8k/omega^2 !=0.                                   (9)

The statement holds for either sign of the coordinate-axis u, and for either
perpendicular orientation v. Thus the only rank-defective offcontact point
cannot be critical under target (2). The b pin does not enter (9).

## 6. Conditional Morse regularity: the actual affine incidence map

The argument now uses the residual law (3), not the unconditional coefficient
law. On any torus chart away from x and the possible K2 exceptional point,
write

    grad F_zeta(p)=g(p)+D(p)zeta.

Sections 4–5 prove rank D(p)=2. Choose a locally invertible two-column minor
D_C. Split zeta=(alpha,eta) in R^2 x R^(d-2), and solve gradient zero as

    alpha=a(p,eta)=-D_C(p)^(-1)[g(p)+D_D(p)eta].             (10)

The smooth map

    Psi(p,eta)=(a(p,eta),eta)

has domain and target dimension d. Differentiating the actual gradient
constraint at its zero, keeping the reconstructed field realization fixed
in the partial p derivative, gives

    D_p a=-D_C(p)^(-1) Hess F_zeta(p),
    |det D Psi|=|det Hess F_zeta(p)|/|det D_C(p)|.           (11)

A non-Morse offcontact critical point therefore makes zeta a critical value
of Psi. The explicitly imported equal-dimensional Sard theorem says that
critical values of a C-infinity map between open subsets of R^d have
Lebesgue measure zero. Countable torus/minor charts and bounded localizations
cover the incidence domains. The full Gaussian density of zeta in (3) then
makes all these exceptional coefficients a conditional null set.

At the excluded K2 point, (9) forbids criticality for every residual coefficient.
At x, the gradient is pinned and the Hessian is singular by design; neither
point is incorrectly put through this rank 2 argument.

Classical import: Arthur Sard, [The measure of the critical values of
differentiable maps, Bulletin AMS 48 (1942),883–890](https://doi.org/10.1090/S0002-9904-1942-07811-6).
Only the stated equal-dimensional C-infinity theorem is imported. The affine
incidence map and determinant computation (10)–(11) are established here.
This preparation does not report a fresh successful full-paper fetch or a
new proof of classical Sard. The author's earlier single PDF opener failed
and was not retried or rerouted. A peer's indexed primary-statement exposure
is a separate actor's read, not this author's full-paper exposure.

## 7. Isolated contact and finiteness

FG's positive seven-row covariance implies

    Var(A_0|U_0)>0,  A_0=F_vv(x).

A_0 is a scalar Gaussian with positive variance under (3), so A_0!=0 almost
surely. In local (u,v) coordinates, F_v(s,t)=0 can then be solved as

t=h(s) near contact. The exact pins give h(0)=0 and h'(0)=0. Put

    g(s)=F_u(s,h(s)).

Then

    g(0)=0, g'(0)=0, g''(0)=F_uuu(x)=12k.

Thus g(s)=6k s^2+o(s^2)>0 for sufficiently small nonzero s. Both gradient
components vanish only at contact in that neighborhood. This proves contact
is isolated; it is a cubic fold with one nonzero transverse Hessian pivot,
not an ordinary Morse critical point.

All other critical points are isolated by Section 6. The critical set is a
closed subset of a compact torus. An infinite critical set would have an
accumulation point which is also critical, contradicting isolation at that
point. The whole critical set is therefore finite almost surely.

For later branch arguments it is useful to note openness. At one such good
residual coefficient, A_0 stays nonzero in a nearby coefficient neighborhood.
The transverse implicit solution h(s,zeta) is jointly smooth. All contact
constraints remain exact, so g(0,zeta)=g'(0,zeta)=0 and
partial_s^2 g(0,zeta)=12k. Shrinking the neighborhood makes g(s,zeta)>0
uniformly for small nonzero s. The finitely many remaining Morse critical
points have smooth local branches by the implicit function theorem.
On the compact complement of these neighborhoods the original gradient has
a positive lower bound, which persists for sufficiently close coefficients.
Hence no new critical point appears there. The good coefficient set has
open neighborhoods with a fixed finite list of noncontact Morse branches.

## 8. Conditional height differences at K>=3: constructive proof

For any p!=q, the row F(p)-F(q) is nonzero on ker A. If p=x, use a function
H from Section 4 nonzero at q; its values at x and q differ. Otherwise choose
H vanishing to order 4 at x and nonzero at p. Choose a coordinate j in which
p_j!=q_j modulo L. Then

    h(z)=H(z)[1-cos(omega(z_j-q_j))]                        (12)

has support in {-3,...,3}^2, vanishes in every pin row, and has h(p)>0,
h(q)=0. It is a residual direction with unequal values. This proves positive
conditional variance of every distinct-point height difference at K>=3.

This is variance conditional on the six contact pins. It is not a claim about
variance after additionally fixing both offcontact gradients. That stronger
Schur-rank list is not consumed by the branch argument in Section 10.

## 9. Conditional height-difference exceptions at K2

Suppose F(p)-F(q) lies in the contact row span, with p!=q. Translate x=0 and
write theta_p,i=omega p_i, theta_q,i=omega q_i modulo 2pi. Choose a coordinate
in which these angles differ, say coordinate 1. Its multiplier identity is

    t_p^j exp(i theta_p,2 l)-t_q^j exp(i theta_q,2 l)=P(j,l),
    P in P_0,  -2<=j,l<=2.                                 (13)

Fourth differences in j at start -2 and Delta^4P=0 give

    t_p^(-2)(t_p-1)^4 exp(i theta_p,2 l)
      =t_q^(-2)(t_q-1)^4 exp(i theta_q,2 l).                (14)

For t=exp(i theta),

    t^(-2)(t-1)^4=16 sin^4(theta/2).

Using l=0,1 in (14), neither differing coordinate angle can be 0; both scalar
factors are positive and equal, and theta_p,2=theta_q,2. Equality of sin^4
implies equal cosines. The two differing angles must therefore be theta and
-theta, with theta neither 0 nor pi modulo 2pi. Fourth differences in the other
coordinate, using j=1 where 2i sin(theta)!=0, force the common second angle
to be 0. The pair is (p,q)=((s,0),(-s,0)) relative to x, with distinct sites.

On the five first-coordinate frequencies, exact odd interpolation gives

    2i sin(j theta)=i[A(theta)j+B(theta)j^3],
    A(theta)=(2/3)sin(theta)[4-cos(theta)],
    B(theta)=(2/3)sin(theta)[cos(theta)-1].                  (15)

The cubic coefficient is nonzero. Polynomial injectivity and (4) then force
u along that coordinate axis. The coordinate-interchanged case yields the
other axis, and these are the only dependent height-difference pairs.
Conversely (15) is an actual real observation identity. With signed physical
s along u, theta=omega s, it gives

    F(x+su)-F(x-su)
      =[A(theta)/omega]F_u(x)-[B(theta)/omega^3]F_uuu(x)
      =8k omega^(-3) sin(theta)[1-cos(theta)] !=0.          (16)

Distinctness excludes theta=0 or pi. Changing the ordered pair reverses the
gap sign. The target b cancels; its constant mode cannot turn this nonzero
gap into a collision. A contact-versus-other height difference has no such
exception: if one selected coordinate phase is1, (14) would force the other
to be 1 as well, contradicting the selected coordinate difference.

Thus, at K2, every distinct-point height difference either has positive
conditional variance or is a deterministic nonzero quantity under (2).
This statement does not replace rank deficiency by a falsely positive
covariance.

## 10. Distinct critical values through finite critical branches

Work in a good residual coefficient neighborhood from Section 7, with smooth
noncontact critical branches p_i(zeta). Let

    H_i(zeta)=F_zeta(p_i(zeta)).

For a residual variation delta zeta, actual criticality eliminates the
moving-site term in differentiation:

    dH_i[delta zeta]=delta F(p_i).

Therefore

    d(H_i-H_j)[delta zeta]=delta F(p_i)-delta F(p_j).        (17)

At K>=3, Section 8 makes this derivative nonzero whenever i!=j. At K2 it is
nonzero unless the sites form one of Section 9's dependent pairs. At such
pairs (16) already makes H_i-H_j nonzero, so they cannot lie on its zero
level. Every possible critical-value collision hence has a nonzero residual
derivative. The implicit function theorem represents its zero set locally
as a smooth hypersurface, which has d-dimensional Lebesgue measure zero.
Countably many local charts of that level set suffice. Countably many good
coefficient neighborhoods and their finite branch pairs then give a null
set for any collision. This is not an uncountable union of fixed-site null
sets.

For a collision with contact height b, the derivative is delta F(p_i)-
delta F(x), because delta F(x)=0 in ker A. Sections 8–9 show this functional
is nonzero for every p_i!=x, so the same regular-level argument excludes
H_i=b almost surely. Together with Sections 6–7, this proves all conclusions
in Section 2.

Crucially, varying residual coefficients also moves a Morse critical point.
The implicit function theorem realizes every nearby residual coefficient as
that branch, with its site adjusted to keep its gradient zero. One must not
replace (17) by a conditional covariance after fixing the sites' gradients
as extra coefficient constraints. Such a stronger rank statement might
fail and is neither assumed nor proved here.

## 11. Uniform and nonuniform boundaries

The covariance matrices in Sections 4 and8 depend continuously on frames and
sites. For each fixed positive spectrum and K>=3, compactness gives:

* a positive conditional gradient covariance floor on dist(p,x)>=delta>0,
  uniformly over basepoints and all supplied frames;
* a positive conditional height-difference variance floor on
  dist(p,q)>=delta>0, uniformly over p,q,x and frames.

For the second statement p or q may equal x; Section 8 includes that case.
These are existence statements from proved ranks and compactness, not numeric
eigenvalue evidence or estimates for a usable radius. Across any fixed finite
set of positive spectra and cutoffs K>=3, their minima remain positive.

At K2, no all-frame gradient covariance floor on a generic away-from-contact
domain containing the axial opposite site exists: Section 5 gives a genuine
rank defect there. No all-frame height-difference variance floor on a domain
containing the symmetric axial pairs exists either. Floors hold on compact
subsets separated from the respective rank-defect strata. A fixed nonaxial
frame has no such offcontact defects, but that does not yield a common floor
while frames approach an axis. The nonzero affine means (9) and (16) suffice
for qualitative genericity; this preparation derives no uniform probability
rate or integrable marked-Rice bound near these strata.

The statements hold for each fixed q,K,L,b,k with k>0. They therefore cover
the twenty fixed ideal Gaussian catalogue cases at L24,K2/3, all supplied
frames, without using arbitrary jets. No common probability-one event over
all these continuously varying conditional parameters is asserted. No
uniformity over q tending to a simplex boundary, unboundedK, variableL,
k tending to 0 or unbounded marks is claimed.

Absolutely continuous finite reweightings of this six-pin conditional law
preserve its null sets. In particular a nonnegative contact weight depending
on A_0 and finite derivative windows, with finite positive normalizer,
preserves the qualitative conclusion. This does not prove that normalizer,
a window-removal rate, or closure after arbitrary additional fixed pins.

## 12. Remaining obligations and preservation

The result proves a specific contact-conditioned global genericity statement
for the actual finite ideal Gaussian field. It does not prove that a nearby
maximum/saddle pair is an actual elder-rule bar, continuity of a globally
selected pairing, a once-counting identity, a complement estimate, a
factorial-moment or all-point conditional rank list, or a persistence limit.
Finite-r two-point conditioning is a different observation list and has not
been discharged by this proof. Qualitative contact genericity alone does
not imply uniform domination for any of those consumers.

The conditional contact itself remains degenerate; it cannot be passed to
C210's nonsingular physical-Hessian premise as an ordinary Morse point.
The frozen candidate theorem needs none of the new global genericity
arguments, and the frozen derivative-controlled chart theorem is unchanged.
No sampler certification, numerical confirmation, formal kernel execution,
scientific admission, publication acceptance or external uptake follows from
this analytic result. The [review record](REVIEW.md) binds this source and retains
the source-exposed same-team roles stated at the beginning.
