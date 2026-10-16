# Marked iid-copy Poisson limit for actual short H0 bars

Mathematical source by OpenAI/Codex universality_review,8October2026;
repository composition and current-source binding by OpenAI/Codex root,
task01a11cbf-068d-7102-861e-75814e715c98, standing owner authorization,
exact seven-path claim6070584604. Initial base is
main2c84170712c59d9de580c172815bd30bac5d93cd. Mathematical Sections1-8
are byte-identical to the frozen complete process preparation. The
[review record](REVIEW.md) binds original and current exact-byte verdicts.

The result superposes independent whole copies of the fixed torus law.
Its actual weighted cluster-avoidance proof is a substantive extra step
beyond the first moment. It supplies no spatial expanding-domain law,
regional witness closure, global second-factorial bound, numerical window,
sampler certificate, Lean proof or scientific promotion. All named
performers are AI agents with substantial source and route exposure;
organizational independence is zero.

## 1. Scope: fixed law, empty one-field limit, nonempty independent-copy limit

Fix an integer d>=2 and a side length L>0 before taking any limit. Set
X=R^d/(L Z^d), with its flat metric and volume L^d. The actual centered,
variance-one stationary Gaussian field has covariance

    K_L(z)=sum_(j in Z^d) exp(-|z+Lj|^2/2)
             /sum_(j in Z^d) exp(-|Lj|^2/2).                 (1)

Its exact Fourier weights are q_n proportional to
exp[-(2pi/L)^2 |n|^2/2] on every n in Z^d. It has its random constant
mode and the smooth version of IA below. It is neither a finite-cutoff
catalogue law nor a spatially unbounded field.

Use ordinary finite superlevel H0 bars, with decreasing level and the
elder rule. Exclude the single essential global-maximum class. On the
Morse/distinct-critical-value locus, a finite bar has a unique younger
maximum M and its unique merging index-(d-1) saddle S, with

    b=F(M), ell=F(M)-F(S)>0.

Let N_t be the number of these bars with 0<ell<=t. The all-bar first
moment proved in IA is

    E N_t ~ (3/2) L^d c_(d,L) t^(2/3), c_(d,L)>0.          (2)

Consequently, a single field's process on any fixed compact set of
rescaled lifetimes tends to the empty process, not a nondegenerate
Poisson process. Indeed P(N_(Tt)>0)<=E N_(Tt)->0 for each fixed T.
More strongly, almost every one field has finitely many finite bars,
all with positive lifetimes, so N_(Tt)=0 eventually along every t->0.
If there are no finite bars this assertion is immediate. Rescaling
lifetimes by t alone makes existing bars escape to infinity.

The nondegenerate regime considered here is instead **iid superposition
of independent whole field copies on this same fixed torus**. Let
t_n->0 and let n be the number of copies, with

    n t_n^(2/3) -> lambda, 0<lambda<infinity.               (3)

The copies, rather than bars within one copy, are independent. The
result below is a Poisson random measure for their superposition, with
its actual correlated birth/gap/direction mark law. This is a narrower
independent-copies theorem. It is not a spatial expanding-domain or
mixing theorem, nor a joint L->infinity/lifetime limit.

The new substantive input beyond (2) is the weighted cluster-avoidance
estimate

    E[N_(Tt) 1{N_(Tt)>=2}] = o(t^(2/3)) for every fixed T. (4)

Section 4 proves (4) from the same physical contact coupling and global
critical-point isolation. A first moment alone does not imply (4);
Section 8 gives an explicit counterexample. We do not replace (4) by
an unproved second-factorial assertion.

## 2. Borel bar counts and precise marks

Let G be the open locus of C4 functions on X which are Morse with
distinct critical values. The actual law gives G probability one by
IA's finite-block/tail incidence proof, with Sard before Rice. The
intrinsic Borel elder selector sigma(M,S,f), assigned zero outside G,
is the one in ST Section 2: the death threshold is the rational-level
open-superlevel connectivity supremum to a point of higher birth.
Morse handle attachment proves its unique actual H0 once-counting.

For completeness, N_t is a legitimate joint Borel mark in (f,t), not
an assumed measurable random root count. Around any f in G, implicit
critical branches enumerate all its finitely many critical points;
a compact gradient floor excludes additional roots outside their
neighborhoods. These branches are continuous in the C4 parameter and
their heights are continuous. The separable C4 function space has a
countable cover of such neighborhoods. On each, sum the finitely many
jointly Borel expressions

    sigma(p_i(f),p_j(f),f)
       1{0<f(p_i(f))-f(p_j(f))<=t}.

This gives N_t consistently on G. Assign it zero on the Borel complement.
The same finite sums show that all marked count evaluations below are
Borel. No global genericity at every fixed two-site conditional target
is assumed: the selector and count are defined even on bad fibers.

To define directions everywhere, use the Borel torus displacement
representative h(M,S) whose coordinates lie in [-L/2,L/2), choosing
this fixed convention at coordinate ties. Put

    r=|h(M,S)|>0, u=h(M,S)/r in S^(d-1), k=ell/r^3>0.       (5)

The cut-locus convention will not affect the limit. In the short spatial
band r<r_0<L/2 the displacement is unique and the literal pins are

    M=x-r u/2, S=x+r u/2,
    F(M)=b, F(S)=b-k r^3, grad F(M)=grad F(S)=0.           (6)

Thus u points **from the dying maximum to its killing saddle**. It is
an ordered direction, with no quotient by +/- and no angular factor 1/2.
The midpoint x is a torus point; the birth location is M. In the limit
M->x at fixed scaled lifetime and k. We retain the actual birth mark b
and physical gap variable k in the point process; omitting k is a later
projection, not a change of conditioning.
An orthonormal completion of u can be chosen Borel, with continuous
choices on local sphere charts. Transverse orthogonal changes only
rotate the contact rows with absolute determinant one and conjugate A;
the density, determinant/type weights and full-field pin law are
unchanged. No globally continuous frame over the sphere is required.

Define the finite random point measure

    Xi_t=sum_(finite bars) delta_(ell/t, b, k, M, u)       (7)

on E=(0,infinity)_a x R_b x (0,infinity)_k x X_x x S^(d-1).
The a coordinate denotes rescaled lifetime. E is locally compact and
second countable. Xi_t is zero on G's complement. In particular its
restriction to 0<a<=T has total count N_(Tt), though that window is
not compact in all its marks. Gaussian first-moment domination makes
its expected count finite. The window may include arbitrarily small
positive a; the limiting intensity will still have finite mass there.

## 3. The actual one-bar marked intensity, including all marks

All conditional versions here are the canonical actual Gaussian
regression versions from IA. Let U_r be its literal rescaled contact
observation, Sigma_r its covariance, and

    v_r=(b-k r^3/2,-k r^2,0,12k,0,...,0),
    pi_r(v_r)=the U_r density at this target,
    Q_r=the conditional full-field law given U_r=v_r.

Write H_M,H_S for the physical Hessians at (6),

    W_r=|det H_M det H_S|
          1{H_M negative definite, index(H_S)=d-1},
    V_r=W_r/r^2,
    z_0(b,k,u)=36 k^2 E_Q0[(det A_0)^2 1{A_0<0}].          (8)

A_0 is the full symmetric transverse Hessian block, of size d-1, and
index means number of negative eigenvalues. The indicator cannot be
replaced by a dimension-independent determinant-sign test. The contact
target is U_0=(F,grad F,Hess F u,F_uuu) with F=b, gradient and axial
Hessian column zero, and F_uuu=12k. The row ordering of IA/ST gives
the same density pi_0(v_0); an orthogonal permutation has absolute
determinant one.

IA proves, at every fixed b in R,k>0,u,x, under its actual common-field
coupling,

    F_r -> F_0 globally in C4,
    V_r -> V_0=36k^2(det A_0)^2 1{A_0<0},
    E_Qr[V_r sigma_r] -> z_0(b,k,u),                      (9)

and for all b in R,k>0 and common 0<r<=r_0,

    12 pi_r(v_r) E_Qr V_r <= H(b,k),
    H(b,k)=C(1+|b|+k)^(2d) exp[-c(b^2+k^2)],
    integral_Rx(0,infinity) k^(-2/3) H(b,k) db dk<infinity. (10)

These bounds include small k and large birth/gap marks. They are
unnormalized; there is no all-mark inverse normalizer or inverse random
curvature premise. The conditional field norm has all polynomial moments
needed in (9). IA separately gives an O(1) far lifetime density per
volume, so the expected number of far bars of lifetime <=Tt is O(t).

For every bounded continuous test h supported in a bounded lifetime
window 0<a<=T, the direct near-pair COUNT formula and radial substitution
give

    t^(-2/3) E[Xi_t h]_near
      =integral_(a,b,k,x,u) a^(-1/3)
         1{k>=ta/r_0^3}
         [12 pi_r(v_r)/(3 k^(2/3))]
         E_Qr[V_r sigma_r]
         h(a,b,k,x-r u/2,u) da db dk dx d sigma(u),
    r=(ta/k)^(1/3), 0<a<=T.                              (11)

The formula follows first for nonnegative Borel h by area formula,
Tonelli and an ordinary change of variables; bounded signed tests follow
by subtraction. It does not differentiate a cumulative asymptotic.
The exact factors are spatial r^(d-1), height r^3, observation
12r^(-(d+3)), and Hessian product r^2. They give r dr. The substitution
ell=k r^3 gives d ell and r dr=(ell^(-1/3)/(3k^(2/3))) d ell;
ell=ta then supplies t^(2/3). Midpoint spatial Jacobian is exactly one
in this flat embedded band. We keep dx rather than prematurely cancel
the whole-volume factor.

Use (9) at fixed a,b,k,u,x, and h(a,b,k,x-r u/2,u)->h(a,b,k,x,u).
The absolute integrand is bounded by

    ||h||infinity a^(-1/3) H(b,k)/(3k^(2/3)), 0<a<=T,      (12)

which is integrable over all birth/gap/frame/space marks. The far term
divided by t^(2/3) is O(t^(1/3)). Dominated convergence proves

    t^(-2/3) E[Xi_t h] -> integral_E h d eta,              (13)
    d eta=4 a^(-1/3) k^(-2/3) pi_0(v_0) z_0(b,k,u)
                      da db dk dx d sigma(u).             (14)

For compact-supported continuous h, (13) is vague mean-measure
convergence. The same argument for the full lifetime window h=1{a<=T}
gives its total mass, and (12) controls the noncompact b,k tails and
the a->0 tail. Hence the finite measures restricted to a<=T are tight
and converge weakly whenever their boundary at a=T is treated by a
continuous approximation. Eta has no atom at T.

Integrating k defines

    beta(b,u)=4 integral_0^infinity
                    k^(-2/3) pi_0(v_0) z_0(b,k,u) dk,
    d eta_(a,b,x,u)=a^(-1/3) beta(b,u)
                              da db dx d sigma(u).        (15)

The b,u integral of beta is c_(d,L), so

    eta{0<a<=T}=(3/2)L^d c_(d,L) T^(2/3),
    eta_(a)(da)=L^d c_(d,L) a^(-1/3) da.                  (16)

This is the whole torus intensity. Per-volume normalization would
divide it by L^d. Stationarity implies uniform x and its independence
from the other marks in this limiting measure. The lifetime factor
also separates after rescaling. No independence of b,k,u is asserted;
their joint regression correlations remain in (14).

For reference the same actual coefficient, not a numerical substitute,
is IA's exact functional

    c_(d,L)=Gamma(7/6)/(24^(1/3) sqrt(pi))
       * integral_S p_G(0) p_Vu(0) tau_u^(4/3)
          E[(det A_u)^2 1{A_u<0}|V_u=0] d sigma(u),       (17)

where G=grad F, V_u=Hess F u is the full d-coordinate axial column,
and tau_u^2=Var(F_uuu|G=0). Every covariance is from (1). Birth
integration removes F but retains V_u/A_u correlations. The numeral
24 in this prefactor is the Jacobian/gamma factor, not the side L.
Nothing here certifies old digits or old numerical intervals.

## 4. Global contact isolation supplies physical weighted cluster avoidance

We first state the deterministic endpoint fact used in the new argument.
Fix a contact-law realization F_0 satisfying IA's contact genericity:
the contact x is isolated; every other critical point is Morse; their
heights are mutually distinct and different from b. The critical set
outside x is finite. On V_0>0, A_0<0 and the literal cubic axial
derivative is 12k>0. IA/ST's local vector ridge has positive third
derivative, exact zeros at the pinned M and S, and no other local
critical point. Away from the contact neighborhood, implicit branches
persist from the finitely many Morse roots, and a compact gradient
floor excludes new roots. Therefore for all sufficiently small r,
the entire critical set of F_r consists of the pinned M,S and those
old branches, with their old distinct limiting heights.

The actual elder-transfer result makes M,S a finite actual paired bar,
not merely a typed candidate. Each maximum supplies at most one birth
bar, and each merging saddle kills at most one bar. Thus no other bar
can use M or S as an endpoint. Every other finite bar uses two old
branches. If there are at least two old branches, the minimum of their
finitely many distinct limiting height differences is positive. Their
perturbed differences are consequently bounded below by half that
minimum for all sufficiently small r. With fewer old branches, no
other finite bar can have two old endpoints. The pairing of old
branches need not be combinatorially fixed for this argument: every
possible two-old-endpoint gap has a positive limiting lower bound.
The essential class is excluded throughout.

Now fix a in (0,T], b in R, k>0, u and x, and put
r_t=(ta/k)^(1/3). Along this actual canonical conditional coupling,
the pinned bar has lifetime k r_t^3=ta<=Tt. On V_0>0 the preceding
endpoint fact proves

    N_(Tt)(F_(r_t))=1 eventually as t->0.                 (18)

The realization-dependent lower gap need not have an integrable
inverse, an explicit deterministic cutoff, or a uniform probability
rate. On V_0=0, V_(r_t)->0 regardless of the count or selector.
Consequently, for every fixed a,b,k,u,x,

    V_(r_t) sigma_(r_t)
          1{N_(Tt)(F_(r_t))>=2} ->0 almost surely,
    E_Q_(r_t)[V_(r_t) sigma_(r_t)
          1{N_(Tt)>=2}] ->0.                             (19)

The second line follows from the same polynomial field-norm domination
as (9), at these fixed marks. This is a statement about the specified
contact conditional coupling, not about independence of two contacts.
No disintegration of unconditional Morse genericity to all pin targets
is used; IA proves the required limiting contact genericity separately.

Apply COUNT to the actual bar selector with the additional nonnegative
Borel mark 1{N_(Tt)>=2}. Its left side is exactly
E[N_(Tt) 1{N_(Tt)>=2}]. In (11) replace E[V_r sigma_r] by

    E[V_r sigma_r 1{N_(Tt)>=2}].

This modification is permitted for the full-field global mark by IA's
all-Borel finite-tail area formula. Only one Hessian determinant
product enters. The integrand's pointwise limit is zero by (19), and
its unnormalized absolute bound is (12) with ||h||infinity=1. Dominated
convergence on a in (0,T], all b,k, frames and space gives

    t^(-2/3) E[N_(Tt) 1{N_(Tt)>=2}]_near ->0.             (20)

The far **anchor** contribution counts actual far bars with the same
bounded indicator; it is bounded by their unconditional expected
number O(t). It does not multiply the far count by another count.
Combining it with (20) proves (4).

In particular P(N_(Tt)>=2)<=E[N_(Tt)1{N_(Tt)>=2}]/2
=o(t^(2/3)). Also E N_(Tt)-P(N_(Tt)=1)
=E[N_(Tt)1{N_(Tt)>=2}]=o(t^(2/3)), so

    P(N_(Tt)=1) ~ (3/2)L^d c_(d,L) T^(2/3) t^(2/3).      (21)

Equation (4) is deliberately not the assertion
E[N_(Tt)(N_(Tt)-1)]=o(t^(2/3)). The factor N_(Tt)-1 is unbounded;
no proof here bounds it by the existing single-contact polynomial
envelope. Rare high multiplicities could distinguish the estimates.
The process theorem needs (4), and (4) has just been derived without
that stronger factorial premise.

## 5. Laplace functional of iid superposition

Take n independent copies of the full actual law and their measures
Xi_(t_n)^(j), all marked in the same torus X. Put

    S_n=sum_(j=1,...,n) Xi_(t_n)^(j).                     (22)

Let f>=0 be continuous with compact support in E. Its lifetime support
is contained in 0<a<=T for some fixed T. Write h=1-exp(-f), and

    q_t=E[1-exp(-Xi_t f)], m_t=E[Xi_t h].

For a realization with at most one point in this lifetime window,
1-exp(-Xi_t f)=Xi_t h. For any finite list of nonnegative f-values,
the union-product inequality gives

    0<=sum_i(1-exp(-f_i))-[1-exp(-sum_i f_i)]
       <=N_(Tt) 1{N_(Tt)>=2}.                            (23)

Thus (4) and (13) imply

    q_t=t^(2/3) [integral_E (1-exp(-f)) d eta+o(1)].       (24)

Only independence of whole copies is used in the next step:

    E exp(-S_n f)=(1-q_(t_n))^n
      ->exp[-lambda integral_E(1-exp(-f))d eta].          (25)

Indeed n q_(t_n)->lambda eta(1-exp(-f)), and
n q_(t_n)^2=O(n t_n^(4/3))->0. The right side is the Laplace
functional of the Poisson random measure with intensity lambda eta.

Here is an elementary finite-window description of the process
convergence, so (25) is not mistaken for a first-moment assertion.
Restrict to a<=T. By (4), the probability that any of the n copies
contributes two or more points is at most

    n P(N_(Tt_n)>=2)=o(n t_n^(2/3))->0.

Censor each copy which contributes two or more points, replacing its
measure by zero. The censored copies remain independent, each a Bernoulli
single point, and their superposition differs from the original with
probability tending to zero. The singleton intensity differs from its mean measure
in total mass by at most E[N_(Tt)1{N_(Tt)>=2}], so its conditional
singleton mark law converges weakly to eta restricted to a<=T,
normalized by m_T=(3/2)L^d c_(d,L)T^(2/3). Its success probability
is m_T t^(2/3)+o(t^(2/3)). The binomial number of successful copies
therefore converges to Poisson(lambda m_T), and their independent
marks converge to the normalized limiting mark law. This proves
finite-window marked Poisson convergence. Equivalently, apply (25)
to linear combinations of tests in that window. Tightness of the
marks follows from the all-mark domination in (12), including small
a and the b,k tails; counts are tight from their bounded expectations.

Increasing T yields vague point-process convergence on E, or,
equivalently, the usual jointly Poisson counts on disjoint relatively
compact eta-continuity sets with the displayed intensity. No new
point-process literature result beyond the elementary binomial-to-Poisson
limit and the standard characterization by these marked finite-window
laws is needed. All calculations above directly identify that law.

Projecting out k, or out b,k,u, gives the corresponding limits provided
the full bounded-lifetime window is used with the tail domination just
proved. In particular the rescaled lifetime process has intensity

    lambda L^d c_(d,L) a^(-1/3) da,

and its number of points with 0<a<=T is Poisson with mean
lambda (3/2)L^d c_(d,L)T^(2/3). A uniform birth location is a property
of this fixed-torus iid superposition, not a statement of spatial
independence within one field.

## 6. What a direct second-factorial attack proves, and what remains

For actual distinct finite H0 bars, overlapping endpoints are excluded
by the once-counting rule: distinct bars cannot use the same birth
maximum or the same killing saddle, and a maximum cannot be a saddle.
They therefore have four distinct critical endpoints. Candidate-pair
factorial counts can share endpoints and have a different incidence
decomposition; they must not be substituted for the actual count.

The actual infinite law does have the named four-site first-jet ranks.
For distinct sites p_1,...,p_4, take

    B_i(z)=product_(j!=i) theta_(p_j)(z)^2,
    theta_p(z)=sum_l[1-cos((2pi/L)(z_l-p_l))].

B_i kills all jets through order three at every other site and is
positive at p_i. The value/gradient maps of B_i and
B_i sin((2pi/L)(z_l-p_i,l))/(2pi/L) are triangular with positive
diagonal. Coordinate frequency cutoff seven suffices. The original
q_n of (1) is strictly positive for every used mode. Therefore the
joint values and gradients at these four distinct sites have positive
covariance; no independence of the contacts is asserted. The same
finite-block/tail area formula gives a four-point nonnegative Borel
factorial identity on separated domains.

Likewise, two distinct limiting contact centers x_1,x_2 have joint
contact rank. Multiply a local polynomial realizing a prescribed
third-order jet at x_i by theta_(x_j)^2. Since that factor is nonzero
at x_i, its Taylor-series inverse adjusts the prescribed jet through
degree three; it kills the other full third-order jet. The resulting
coordinate degree is at most five. This supplies the two named
contact rows and their independent transverse entries, not a rank
transferred from the finite K2 or K3 catalogue.

Fix a center separation eta_0>0. On its compact center/frame domain,
these ranks yield a joint contact covariance floor. For both radii
r_i<=r_*(eta_0), continuity keeps a common positive floor for the
joint U_(r_1),U_(r_2). Their Gaussian targets have a joint coercive
bound in (b_1,k_1,b_2,k_2). Regression supplies polynomial conditional
field moments; the product of the two normalized determinant weights
is bounded by a field-norm polynomial of degree 4d. Thus there is a
joint integrable Gaussian mark envelope on this separated band.
Each fold has the same r_i dr_i ledger. Applying both substitutions
ell_i=k_i r_i^3 and integrating 0<ell_i<=Tt proves that this
**separated-center, sufficiently small-radius twofold contribution**
is O_(eta_0)(t^(4/3)).

The qualifier is essential. As the centers coalesce, the covariance
floor degenerates; positivity at distinct sites gives no common lower
bound there. Controlling the shrinking center-separation range would
require an actual multiscale four-site regression/determinant domination
including its higher degeneracies. This preparation supplies neither
that bound nor a global second-factorial theorem. Nor is a term with
one far anchor multiplied by the remaining count bounded merely by
the far first moment. The bounded cluster mark in Section 4 avoids
exactly these two unsupported replacements. It controls probability
and first-weighted multiplicity, which suffice for (25), rather than
claiming every factorial moment or a quantitative convergence rate.

## 7. Conditional process interface and why it is actually discharged here

The iid Laplace conclusion itself uses only two interfaces at scale
s(t)=t^(2/3):

* Mean mark measures: s(t)^(-1) E Xi_t -> eta on each bounded-lifetime
  window, with finite mass/tight tails.
* Weighted cluster avoidance: E[N_(Tt)1{N_(Tt)>=2}]=o(s(t)) for each T.

Under those two interfaces and n s(t)->lambda, (23)-(25) prove the
Poisson theorem for any iid random point measures. If the second
interface were retained as an assumption, this would only be a
conditional Laplace theorem. For the actual law (1), Section 4 derives
it from the source-bound CONTACT, conditional global genericity,
actual elder transfer, all-Borel COUNT and ENVELOPE, with FAR for the
anchor complement. These are not supplied by covariance positivity
alone. The unbounded set of birth/gap marks is handled by the actual
H bound, rather than a normalized compact-mark probability claim.

The pointwise isolation proof does not require a common deterministic
gap for all realizations or an all-target samplewise generic event.
It fixes a,b,k,u,x first, uses that fixed contact law's probability-one
genericity, then integrates its weighted zero limit. All-mark dominated
convergence is the precise quantifier change. The proof uses the
canonical Q_r at each target, so it does not inherit a conclusion
from an unspecified conditional version on null fibers.

## 8. Explicit first-moment counterexample

This is an abstract comparison process, not a counterexample to the
actual Gaussian law. On a mark window with finite nonzero measure eta
of mass m, let P=eta/m. With probability m t^(2/3)/2, produce exactly
two independent P marks; otherwise produce no points. For small t
this is a valid random finite point measure Z_t, and

    E Z_t=t^(2/3) eta,
    E[N_t 1{N_t>=2}]=m t^(2/3).

Its first moment is exactly the same rare-event scale, but its iid
superposition under n t^(2/3)->lambda has Laplace functional

    exp{-(lambda m/2)
          [1-(integral exp(-f) dP)^2]}.

This is a Poisson process of two-point clusters, not the Poisson
random measure with exponent -lambda integral(1-exp(-f))d eta.
In particular its total count takes only even values. It also has
second factorial of order t^(2/3). The example demonstrates why the
new physical isolation/weighted-cluster argument is needed and why
the leading lifetime intensity by itself cannot identify a process law.

## 9. Existing lanes and residual scope

The actual PAPER1_NOTE lines 61-69 and the research README/completion
audit were consumed before this derivation. They distinguish the
regional shrinking additional-witness problem, rejected candidates,
replacement bars and actual bars, and explicitly warn that a global
factorial bound alone does not discharge a regional collision estimate.
This preparation does not close or rename that owner lane. Its random
count is all finite actual H0 bars of one unconditioned fixed field;
its asymptotic conditioning is the one-bar contact Palm kernel. The
refined-selector issue main#229 remains separate with its own radius,
mark, weighting and confinement requirements.

There is no expanding-domain process claim, covariance-only universality,
same-field spatial Poisson assertion, higher-homology result, all factorial
moment estimate, selected-density continuity theorem or quantitative
convergence rate. The realization-dependent old-height separation gives
qualitative dominated convergence, not a certified numerical t window.
No finite spectral/grid experiment measures the infinite field exactly.
No seed or held-out field is consumed. Source digests and exposed AI
technical checks do not establish external uptake, priority, publication
acceptance, blind confirmation or a compiled formal theorem.

The substantive advance is the explicit marked mean law14, the physical
weighted cluster-avoidance estimate4, and the marked iid-copy limit25 for
this fixed actual law. The current source bindings and separately recorded
verdicts below distinguish original preparations from incorporated proofs.

## 10. Current source bindings, historical exposure and residual scope

The current analytic dependencies are:

| Input | Current repository source | UTF-8 bytes / lines | SHA256 |
| --- | --- | --- | --- |
| IA | [Exact periodized Gaussian H0](../periodized_gaussian_h0/PROOF.md) | 34629 / 637 | c8bd489ca3fbb3760f7b2d53e7385131ac5336c51ba983e00e4d540fe2f00aaa |
| ST | [Regular-fold structural theorem](../fold_structural_universality/PROOF.md) | 43927 / 804 | 3d952a906553d578876456539a8f0dcd64aedd45099e8394cbe9db99866ec531 |
| S24 | [Actual physical-law source](../../../formal/sources/side24_v1/PROOF.md) | 10272 / 216 | c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769 |

IA preserves mathematical Sections1-8 of original34281B/639lines
SHAcd0abd19cfb0bdd398142c136729d3bf96724939b3e774a3773a683310078a7f,
common-core SHA430b123f42162279f391b02e8320823d8a82f8cab6e86700aede72e8ab05b6cc.
ST preserves Sections1-10 of corrected43761B/813lines
SHAb077e2465174cadf78f2df120a80992534f99b838e18b5ba15fed250f4c28cca,
common-core SHAe30bdc85cfdd2eebfbe26037ea00b5aa90257aadff541c4a16682208025c6323.
The original process author fully read those original saved sources and
reread IA's actual all-Borel,contact genericity,selection,envelope and
complement arguments, plus ST's Borel selector. The current rebind is
root's separately checked composition; original read extents are not
relabeled as fresh reads of later wrapper bytes.

Original process preparation31220B/627lines
SHAe69927b62c25f152b6e977fd409ff8890c86b8b43bf92336912c33d292fd2280
remains frozen in ordinary coordinator custody. Root read all627 actual
lines, challenged endpoint exhaustion/Borel counting, and checked the
actual incorporation difference. The earlier benchmark route attack was
bounded and prefile; it did not review that saved source. REVIEW.md
records the fresh full original and current-source verdicts separately.
A byte-preserved mathematical core does not automatically rebind a
whole-file verdict to new operational source text.

The original process derivation also recorded prior full reads of
SB25439B/501lines SHA5f529e5ed2632c90819746848146e3a24acda7e888cd3c9342df76edcbcbb7a3,
ET21138B/418lines SHA1cac9b525831b7e1f37f0d0561d3231fa73c8713c4c92283dca1d5b51b8a67ce,
and S24c06 at historicalroot6541/blob44b66f04f89fcd87383b3603fa69f1feb64cdddd.
SB preserves the density-version distinction; ET's original finite d2
law is not substituted for IA's actual higher-dimensional adaptation.
Current [SB](../finite_h0_lifetime/PROOF.md)5ea9 and
[ET](../finite_elder_transfer/PROOF.md)c9e4 preserve their mathematical
Sections2-8 and2-13 with their own complete source reviews. These are
history/scope inputs; IA's actual hypotheses and coupling drive the new
process argument.

The existing [PAPER1_NOTE](../../../docs/PAPER1_NOTE.md),6322B/69lines
SHA161ba87433879ee77ade047f1516d8c919fa3097812577cac54d7fe978a7ad63,
was previously fully read, with50-69 reread by the process author.
The existing [research plan](../../../docs/research-translation/20260930/README.md),
17156B/127lines SHAfcf43e5929b175ed9e9b046b9e92123d200c6cbff689d3b68196b3cca55228c7,
was read through95 with1-64 reread. The existing
[completion audit](../../../docs/research-translation/20260930/COMPLETION_AUDIT.md),
21094B/386lines SHA f6e557bb15706d614407daf160a8e44efce6d47775a3fbf078fff0a296093f89,
was read through140 with1-118 reread. Those bounded context reads do not
accept their historical imports, retire their nodes or transfer ownership.
The regional shrinking additional-witness and refined-selector main229
lanes remain distinct. Root independently read PAPER1/S24 in full and
checked the actual current source identities; no new primary-paper or
compiler claim follows from this composition.

IA records inherited bounded primary statements and the explicit classical
Sard,nonnegative-Borel area-formula,Morse/handle/critical-free-flow imports.
Gaussian regression,Tonelli/Fubini,smooth Fourier moments and DCT are
consumed with IA's proved actual hypotheses. The process calculation11-25,
including the new physical cluster estimate, is derived here rather than
attributed to a publication that did not state it.

The stated fixed-d,L iid-copy theorem requires no knowingly unresolved
cluster assumption beyond its source-bound analytic proofs and named
imports. Coalescing-center/global factorial and spatial/regional collision
questions remain unproved here. Changed dependency bytes require fresh
source binding and review. No blind seed,experiment,numerical coefficient,
formal target,model label or scientific status is changed.
