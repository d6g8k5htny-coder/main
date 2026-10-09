# Exact covariance-matched amplitude crossover and whole-barcode clusters

Repository incorporation on 8 October 2026. Mathematical author:
OpenAI/Codex `/root/universality_review`. Repository wrapper and bounded
integration contributor: `/root/final_contract_review`. Integration owner:
`/root`, task `01a11cbf-068d-7102-861e-75814e715c98`, under native claim
`6071350791`, lease through 00:30 UTC on 9 October 2026. The isolated
detached construction base is
`89657c7c48289378949dc59e7fbe6dc1a55d9c62`.

Sections 1-11 are preserved byte-for-byte from the corrected external
34,110-byte / 730-line source, SHA256
`9e8f10162e9fa21f308454df9463459cdd67ef39a2d8103a78a397b4ab82dd0d`.
The 28,494-byte mathematical core has SHA256
`f558dafcdb3841646c3d364056a95c07e7a56759ef5bc47f9b6033b85f5e0df0`.
The preserved original 33,421-byte / 720-line source, SHA256
`5b1859080a137aed9eeeba20e1b6f0c5a61387e5eb6fa20bb1b8704860fb73d2`,
received two separate full-source AMEND verdicts. Its missing critical
small-lifetime condition and ambiguous density/count table labels are
corrected in the successor; the original and its adverse review history
remain immutable.

The corrected preparation has a separate fresh FULL 730-line mathematical
PASS from `benchmark_formal_audit`, a full actual-source read by `root`,
and a bounded complete-correction-diff PASS from `final_contract_review`.
That bounded correction audit was not a new whole-730-line read. Those
completed reviews apply to their identified external sources; they do not
review these new repository wrapper bytes. A fresh incorporated-source full mathematical
review and metadata review are PENDING at this construction snapshot;
the [review record](REVIEW.md) preserves the separate identities and scopes.
Historical assignment-stage pending states retained in Sections 1-11 refer
to preparation chronology, not the current status of an input review.

All contributors are source/route-exposed OpenAI/Codex AI performers;
organizational-independence credit is zero. This is a conventional analytic
composition for the declared law, not numerical, formal, blind or scientific
acceptance. Current operational inputs and actual reading extents are
recorded in Section 12. No compiler, sampler, seed, installation, external
post or scientific-status change is part of this construction.

## 1. Base input, units and review boundary

Let H be the actual fixed Gaussian spectral law on X=R^d/(L Z^d),
d>=2, admitted by SG in the source table below. Write V=L^d and
beta=2/3. Ordinary finite superlevel bars use a fixed coefficient
field; essential intervals are excluded. A bar has positive lifetime
ell_i, degree p_i, birth point x_i and ordered direction u_i from
birth critical point to its killing critical point. Use the existing
Borel shortest-displacement convention outside the near band.

The following exact base premises are consumed:

* H is C4 almost surely and Morse with distinct critical values.
  It has a finite Borel barcode and finite expected total finite-bar
  count. These statements for H0 are proved by SG. Higher-degree
  counting is supplied by the separate HK preparation.
* For each chosen degree population I (one degree, H0, or all finite
  degrees), its expected lifetime measure per volume has a finite
  nonnegative Borel density nu_(H,I), with

      nu_(H,I)(s) ~ c_I s^(beta-1), c_I>0, s down to zero. (1)

* The projected short-bar process of H, keeping only degree,
  lifetime ell/s, birth location and ordered direction, has mean
  limit eta_H and satisfies the actual physical weighted-cluster
  estimate

      E[N_H(Ts) 1{N_H(Ts)>=2}]=o(s^beta), T>0.          (2)

Here the all-degree form uses HK. The H0 form alone uses SG and
needs no higher-degree admission. For the projected measure,

      eta_H{lifetime<=T}=V (c_I/beta) T^beta.           (3)

Its finer angular/location distribution is that proved by SG/HK,
after integration of the original birth and cubic-gap marks.
It is the whole-volume measure, not a per-volume measure.

The scalar transformation in Sections 2-5 needs only the finite
barcode, finite expected count and (1). The process conclusions
also consume (2) and the projected one-bar mean limit. All arguments
below are proved under these named premises, rather than inferring
them from covariance. At assignment the higher-homology source had
one reported full nonauthor PASS and its second full audit was still
pending. This preparation does not upgrade that review state or
claim a new full review of SG/HK. H0 is a complete base specialization
independent of the remaining HK review.

For a population I write

      B_H=# {finite H bars in I},
      m_H(t)=E N_H((0,t])/V=integral_0^t nu_H(s)ds,
      M_alpha=integral_0^infinity s^(-alpha)nu_H(s)ds
              =V^(-1) E sum_(i in I) ell_i^(-alpha).    (4)

We suppress I where harmless. The random sums below use actual
whole-field barcodes, not candidate pairs.

Finite E B_H deserves an explicit global argument. The one-site
finite-incidence/Kac-Rice identity in SG has a uniform gradient
covariance floor on the compact torus and finite conditional Hessian
determinant moments. It gives E#Crit(H)<infinity. Every finite bar
has two unique critical endpoint roles, so B_H<=#Crit(H)/2.
Therefore

      integral_0^infinity nu_H(s)ds=E B_H/V<infinity.    (5)

No high-lifetime asymptotic or global second factorial moment is
needed for the Mellin tail.

## 2. Exact amplitude law and covariance identity

Fix alpha>0. Let R be independent of H, with density

      p_R(r)=alpha r^(alpha-1), 0<r<1.

Set

      a=(E R^2)^(-1/2)=sqrt((alpha+2)/alpha),
      A=aR, F=A H.                                     (6)

Then A has density alpha a^(-alpha)u^(alpha-1) on (0,a),
is strictly positive almost surely and is bounded by a. Its second
moment is exactly one. Since H is centered and A is independent,

      E F(x)=0,
      Cov(F(x),F(y))=E A^2 Cov(H(x),H(y))=Cov(H(x),H(y)). (7)

The same holds for every derivative jet covariance that exists.
The field is stationary, with the same spectral covariance weights
as H. It is not declared Gaussian: one common random amplitude
multiplies the full coefficient vector and couples all sites.

For each realization with A>0, grad F=A grad H and Hess F=A Hess H.
Critical points, Hessian indices, critical-value order and actual
persistence pairings are unchanged. The superlevel identity
{F>=h}={H>=h/A} gives exactly

      ell_(F,i)=A ell_(H,i), b_(F,i)=A b_(H,i),
      x_(F,i)=x_(H,i), u_(F,i)=u_(H,i).                 (8)

The physical cubic mark k_i=ell_i/dist(M_i,S_i)^3 is also
multiplied by A. Thus F is C4, Morse/distinct almost surely, and
every positive C4-norm moment possessed by H remains finite:
E||F||C4^q<=a^q E||H||C4^q. Under SG's simple weighted-square-root
spectral verifier, H has all finite such moments, so F does too.
Under its weaker R4 premise only its supplied moments are needed.
No Fernique upgrade is presumed for that statement.

The total finite-bar count is unchanged, so E B_F=E B_H<infinity.
Ordinary persistence over any fixed coefficient field obeys (8).
This is an exact scalar field operation, not a simulation, a new
singularity class or a truncated approximate covariance identity.

## 3. Exact density and cumulative transformations

Condition on A. The pushforward of the actual expected bar measure
has density A^(-1)nu_H(t/A). Nonnegative Tonelli then gives

      nu_F(t)=E[A^(-1)nu_H(t/A)]
        =alpha a^(-alpha) t^(alpha-1)
                    integral_(t/a)^infinity s^(-alpha)nu_H(s)ds,
                                                    t>0. (9)

Indeed in the amplitude integral set s=t/A:
A^(alpha-2)dA changes to t^(alpha-1)s^(-alpha)ds with reversed
limits. This is the actual selected-bar density, not a candidate
surrogate. It holds for every chosen degree separately and for
their finite sum.

For every fixed t>0 the integral in (9) is finite by (5), since
s^(-alpha) is bounded on s>=t/a. The formula is unchanged by
altering the base density on a Lebesgue null set. It specifies
a locally absolutely continuous, in particular continuous, density
representative on (0,infinity): the lower-limit integral of a
locally integrable function is absolutely continuous on positive
compact intervals, and its prefactor is smooth there. This continuity
comes from amplitude integration. It is not an inherited continuity
claim for the discontinuous global selector's base density.

For the cumulative count, each bar contributes

      P(A ell_i<=t)=min{(t/(a ell_i))^alpha,1}.

Consequently the exact per-volume identity is

      m_F(t)=m_H(t/a)+(t/a)^alpha
                         integral_(t/a)^infinity s^(-alpha)nu_H(s)ds.
                                                        (10)

Equations (9)-(10), not differentiation of an asymptotic cumulative
formula, will establish all density constants. The cutoff t/a,
normalization a and volume factors remain explicit.

## 4. The full three-regime crossover

The base asymptotic (1) implies, for some delta>0, bounds
c_1 s^(beta-1)<=nu_H(s)<=c_2 s^(beta-1) on 0<s<=delta.
For s>=delta, (5) controls every negative-power Mellin tail.
Therefore

      0<M_alpha<infinity exactly when 0<alpha<beta;
      M_alpha=infinity when alpha>=beta.               (11)

Positivity follows from c_I>0. In the finite case it is an actual
global barcode Mellin moment, which includes all lifetimes.

### 4.1 Subcritical amplitude: 0<alpha<beta

The integral in (9) tends to M_alpha by monotone convergence.
Since m_H(t/a)=O(t^beta)=o(t^alpha), (9)-(10) yield

      nu_F(t) ~ alpha a^(-alpha) M_alpha t^(alpha-1),
      m_F(t) ~ a^(-alpha) M_alpha t^alpha.              (12)

Equivalently the whole-volume density coefficient is
alpha a^(-alpha) E sum_i ell_i^(-alpha), and its cumulative
coefficient is a^(-alpha) E sum_i ell_i^(-alpha).
The per-volume formulas divide those expectations by V.
The limiting coefficient is global, rather than determined by
the local contact jet functional alone.

### 4.2 Critical amplitude: alpha=beta=2/3

Split the integral at delta. On its lower piece,
s^(-beta)nu_H(s)=(c_I+o(1))/s. Elementary logarithmic averaging
gives

      integral_(t/a)^infinity s^(-beta)nu_H(s)ds
                               ~c_I log(1/t).          (13)

The finite upper tail and log a have lower order. Thus

      nu_F(t) ~ beta a^(-beta)c_I t^(beta-1)log(1/t),
      m_F(t) ~ a^(-beta)c_I t^beta log(1/t).             (14)

Here a=2 exactly. The density prefactor is beta 2^(-beta)c_I;
the cumulative prefactor is 2^(-beta)c_I. Changing lifetime units
changes the bounded term inside the logarithm, not its leading
coefficient. All lengths here use the fixed base field-amplitude
units declared by the covariance normalization.

### 4.3 Supercritical amplitude: alpha>beta

The lower integral diverges as

      integral_(t/a)^infinity s^(-alpha)nu_H(s)ds
           ~c_I/(alpha-beta) (t/a)^(beta-alpha).        (15)

For proof, trap nu_H between (c_I+-epsilon)s^(beta-1)
below delta and integrate the power explicitly; the fixed tail
is negligible compared with the divergent lower term. It follows
that

      nu_F(t) ~ c_I E A^(-beta) t^(beta-1),
      m_F(t) ~ (c_I/beta) E A^(-beta) t^beta,
      E A^(-beta)=a^(-beta)alpha/(alpha-beta).           (16)

The base m_H term in (10) must be retained: its coefficient
c_I a^(-beta)/beta combines with c_I a^(-beta)/(alpha-beta)
to give the second formula in (16). Dropping it would give the
wrong cumulative constant.

Because A is nonconstant, E A^2=1 and x->x^(-beta/2) is strictly
convex, Jensen gives E A^(-beta)>1 whenever that expectation is
finite. Thus even this restored exponent has a strictly changed
coefficient despite identical covariance. As alpha grows without
bound the multiplier tends to one; no varying-alpha uniform error
or interchange of alpha and t limits is asserted.

## 5. Exactly what the covariance-matched falsifier demonstrates

All realizations preserve the actual regular folds and Morse
pairings of H. No new higher-order singularity is introduced.
Nevertheless (12), (14) and (16) give, respectively, a different
power, a logarithm and a changed leading coefficient. Covariance,
positive smooth-norm moments, Morse genericity and a cubic local
height gap do not determine the selected lifetime law.

The proof identifies the missing distributional control. Negative
amplitude moments, not positive norm moments, distinguish the
three phases. Small amplitudes contract whole finite barcodes,
including bars whose physical endpoints are far apart. CONTACT
and ENVELOPE for a Gaussian covariance cannot be transferred to
this non-Gaussian coefficient law. Nor must the Gaussian FAR
O(1) bound survive.

For example, fix a positive physical distance cutoff and let
nu_H,far be its actual finite-bar density. SG/HK gives nu_H,far<=C
near zero and finite total mass. If this population is nonzero,
then its Mellin moment is finite for every alpha<1, and exactly
the same transform gives

      nu_F,far(t) ~ alpha a^(-alpha) M_(alpha,far)t^(alpha-1).
                                                        (17)

This applies even when beta<alpha<1: the restored full fold
asymptotic (16) coexists with an unbounded, but lower-order, far
density. The five original contracts are sufficient, not necessary,
and it would be false to assert that F has regained all their
literal bounds merely because its leading exponent returns.
For the P5 law, a robust good multi-maximum field provides a
positive-probability population at some fixed positive cutoff.
Section 8 establishes that actual support assertion.

These are exact results for this specified power-density amplitude
law. They are not a claim that every small-amplitude law or every
non-Gaussian field has an alpha universality class. Different
amplitude tails require their own integral analysis.

## 6. The process object and the exact one-copy Laplace transform

For a chosen population I define the finite Borel base configuration

      B(H)=sum_(i in I) delta_(p_i,ell_i,x_i,u_i).

The process space retained first is
I x (0,infinity) x X x S^(d-1); its lifetime coordinate is scaled
by t. For F=A H put

      Xi_(F,t)=sum_(i in I) delta_(p_i,A ell_i/t,x_i,u_i).

The whole finite barcode, critical points and directions are Borel
by SG/HK's actual counting charts. A single fixed F realization
has a finite positive lifetime list and hence tends to empty on
any bounded scaled-lifetime window. Its expectation tends to zero
in all three regimes above. A nontrivial iid-copy limit requires
the explicit normalizations below.

Take a nonnegative bounded continuous test f supported in lifetime
0<lifetime<=T, compact in that open lifetime space. Define

      Psi_f(z)=E_H[1-exp(-sum_i f(p_i,z ell_i,x_i,u_i))],
      K_f=integral(1-exp(-f))d eta_H.                  (18)

Nonnegative bounded Borel tests such as window-count/void marks
will also be used where their integrals are evaluated directly.
Set z=A/t. The exact one-copy Laplace deficit is

      D_t(f)=E[1-exp(-Xi_(F,t)f)]
        =alpha a^(-alpha)t^alpha
                           integral_0^(a/t) z^(alpha-1)Psi_f(z)dz.
                                                        (19)

This identity keeps each complete H barcode together. It is not
a replacement by an independent process of bars with the same
mean. The copies below are independent pairs (H_j,A_j); their
amplitudes are independent across copies and independent of their
own H_j. A single amplitude shared by all copies would be a
different model and is not covered.

Two elementary base bounds will be useful. From (1) and (5),
there is C_T such that, for every s>0,

      E N_H(Ts)<=C_T s^beta.                           (20)

Near zero this is the cumulative asymptotic; away from zero use
the finite total expected count and enlarge the constant. Hence
Psi_f(z)<=min{1,C_T z^(-beta)}. Moreover, the error between
E sum_i(1-exp(-f(p_i,z ell_i,x_i,u_i))) and Psi_f(z)
is nonnegative and at most

      E[N_H(T/z)1{N_H(T/z)>=2}].

The actual one-bar mean limit and (2) therefore imply

      z^beta Psi_f(z)->K_f, z->infinity.               (21)

This is the genuine single-copy cluster input. First moment alone
would not imply (21).

## 7. Subcritical process: exact full-barcode Poisson clusters

Suppose 0<alpha<beta. Equations (20)-(21) show

      J_f=integral_0^infinity z^(alpha-1)Psi_f(z)dz<infinity.

The small-z part is bounded by integral_0^1 z^(alpha-1)dz;
the large-z part by C integral_1^infinity z^(alpha-beta-1)dz.
Monotone convergence in (19) gives D_t(f)/t^alpha ->
alpha a^(-alpha)J_f. For n independent copies with

      n t_n^alpha -> lambda in (0,infinity),

their exact Laplace product tends to

      exp{-lambda alpha a^(-alpha)
        integral_0^infinity z^(alpha-1)
          E_H[1-exp(-sum_i f(p_i,z ell_i,x_i,u_i))]dz}.  (22)

Indeed D_t->0 and n D_t has the displayed finite limit, so
n log(1-D_t)=-n D_t+o(1). This is the Laplace functional of
the following concrete Poisson cluster random measure. Put a
Poisson parent measure on (z,H) with intensity

      lambda alpha a^(-alpha) z^(alpha-1)dz P_H(dH).

Each parent contributes the entire child configuration

      C_z(H)=sum_(i in I) delta_(p_i,z ell_i,x_i,u_i).    (23)

The parent measure is sigma-finite. Only finitely many parents
are active in any bounded lifetime window, as verified next, so
the resulting child process is locally finite. Elementary Poisson
conditioning/exponentiation gives (22). This construction is the
exact Lévy pushforward of the actual whole-field barcode law.

Write ell_min=infinity when the population is empty, and otherwise
the minimum positive finite lifetime. On the full window (0,T]
the active-parent rate is

      Lambda_T=lambda a^(-alpha)T^alpha
                           E[ell_min^(-alpha);B_H>0].  (24)

This follows by integrating z^(alpha-1) over z<=T/ell_min.
It is finite because ell_min^(-alpha)<=sum_i ell_i^(-alpha),
whose expectation is V M_alpha<infinity. The expected number
of child points on that window is

      m_T^*=lambda a^(-alpha)T^alpha E sum_i ell_i^(-alpha).
                                                        (25)

Therefore each bounded lifetime window has a compound Poisson
configuration law with rate (24), with its nonempty finite cluster
distribution obtained by conditioning the parent intensity on being
active. This terminology permits a singleton cluster; it does not
by itself claim strict non-Poisson behavior.

The full first intensity is, for these marks,

      mu(dv,dm)=lambda alpha a^(-alpha)v^(alpha-1)dv
                        E sum_i ell_i^(-alpha)delta_(m_i)(dm),
      m_i=(p_i,x_i,u_i).                                (26)

It agrees with (12), including whole-volume factors. The barcode
Mellin weights affect the spatial/directional marks too; their law
need not be the near-fold mark law eta_H. In particular the original
H pairs can have nonshrinking physical separation. This is an
actual global amplitude cluster mechanism.

If P(B_H>=2)>0, then m_T^*>Lambda_T. The process has void
probability exp(-Lambda_T), whereas an ordinary Poisson process
with the same first intensity would have exp(-m_T^*). Thus it is
strictly non-Poisson. Section 8 proves this condition for H0, and
hence for the all-degree population, in the actual P5 family.
For a population which almost surely contains at most one bar,
(22) does reduce to an ordinary Poisson measure. Strict clustering
is not falsely asserted for every arbitrary carrier or degree.

For the original F birth and cubic marks, the amplitude-natural
subcritical scaling is

      b_F/t=z b_H, k_F/t=z k_H.

Appending these as child marks in (23) gives the same exact
cluster Laplace law: f then acts on (p,z ell,z b,z k,x,u).
The integrability bounds depend only on its bounded lifetime
support. Unrescaled b_F,k_F tend instead to zero in this regime;
one must not retain eta_H's original birth/gap mark law by assertion.
For the collapse, at each fixed parent (z,H) the physical marks
are t z b_H,t z k_H and tend to zero. The active child-count
integrand is integrable by (25), so dominated convergence justifies
the assertion over the complete parent intensity, not merely at
bounded deterministic amplitudes.

## 8. Actual positive multi-bar probability in the P5 family

This support step establishes a strict process falsifier rather
than leaving a hypothetical multiple-bar condition.

Assume SG's full positive K5 block. The trigonometric function

      h(x)=cos(2 omega x_1)+cos(2 omega x_2)
                         +sum_(i=3)^d cos(omega x_i),
      omega=2pi/L,

has four nondegenerate maxima. All its critical points are
nondegenerate: its Hessian is diagonal there with nonzero entries.
Some critical heights coincide, but an arbitrarily small K5
coefficient perturbation makes them all distinct while preserving
the four maxima. This existence follows from SG's actual finite-
block Morse/distinct-value incidence proof: the K5 Gaussian block
has a positive coefficient density, the bad set has Lebesgue
measure zero, and every coefficient neighborhood of h has positive
measure. Choose one such deterministic good h_* in that neighborhood.

There is a C2, hence a C4, neighborhood of h_* in which the field
stays Morse/distinct with exactly the continued critical list and
at least four maxima. On a connected torus all maxima give H0
births; one class is essential and each other class has a finite
death. Thus every field in this neighborhood has at least three
ordinary finite H0 bars, over any coefficient field.

That neighborhood has positive probability for the *actual* full
Gaussian law, not just its normalized finite truncation. Split
H=H_B+T into the actual positive finite K5 block and independent
centered Gaussian C4 tail. The block has positive probability of
being arbitrarily close to h_* in C4. Also

      P(||T||C4<epsilon)>0 for every epsilon>0.          (27)

Here is a self-contained small-ball proof requiring no uncited
Banach support theorem. C4(X) is separable. Its countable covering
by open balls of radius epsilon/sqrt(2) has a ball with positive
tail probability. Two independent copies T,T' both in that ball
have difference norm less than sqrt(2)epsilon with positive
probability. Their difference has the same C4 law as sqrt(2)T:
the centered Gaussian identity holds on the countable dense jet
evaluations generating this Borel law. Consequently (27) follows.
If the tail is identically zero it is immediate. Independence of
block and tail then supplies the claimed positive probability
of the good h_* neighborhood. All arguments are theoretical;
no coefficient draw or seed is consumed.

We conclude P(B_(H,H0)>=3)>0, so (24)-(25) are strictly different
for H0 and the all-degree population. The alpha<beta limit is
strictly a cluster process for these actual admitted fields.

The same good neighborhood also has at least one finite bar whose
endpoints stay a fixed positive distance apart: its finite distinct
critical points have a positive minimum pair distance, and their
branches vary continuously on a sufficiently small neighborhood.
This justifies the positive far population used in (17) at some
fixed distance cutoff. No arbitrary uniform cutoff over all law
realizations is inferred.

## 9. Critical process: logarithmic singleton limit

Suppose alpha=beta. In (19), write z^(beta-1)Psi_f(z)=
z^(-1)[z^beta Psi_f(z)]. Equation (21) and a direct logarithmic
Cesàro argument give

      D_t(f)/(t^beta log(1/t)) -> beta a^(-beta)K_f.    (28)

For proof split at 1 and a large fixed Z. The bounded lower part
divided by log(1/t) tends to zero. Beyond Z, trap z^beta Psi_f
between K_f+-epsilon and integrate dz/z. This also covers K_f=0.
Since a is fixed, log(a/t)/log(1/t)->1.

Choose t_n -> 0 with n t_n^beta log(1/t_n)->lambda in (0,infinity).
The iid Laplace product yields an ordinary Poisson random measure
with intensity

      lambda beta a^(-beta) eta_H.                    (29)

The whole-lifetime marginal is
lambda V beta a^(-beta)c_I v^(beta-1)dv, matching (14).
This conclusion also has a genuine physical cluster check.
Changing variables z=A/t in the mixed multiple-bar expectation
gives beta a^(-beta)t^beta times the integral of

      z^(beta-1) E[N_H(T/z)1{N_H(T/z)>=2}].

For z>=1 the bracket times z^beta tends to zero by (2), and
is bounded by (20). Logarithmic Cesàro makes this piece o(log(1/t)).
For z<=1 the expected total count bounds the bracket by a finite
constant, so that piece is finite. Therefore

      E[N_F(Tt)1{N_F(Tt)>=2}]
                               =o(t^beta log(1/t)).    (30)

Whole-barcode contractions at A of order t contribute only O(t^beta)
and vanish at this logarithmic normalization. The leading events
span logarithmically many scales with both A->0 and t/A->0.
No second-factorial assertion or independent bars within one
field is used.

The marks in (29) are only the retained degree/lifetime/location/
direction marks of Section 6. For unrescaled physical b_F,k_F,
the leading critical mass collapses to b_F=0,k_F=0. To verify it,
amplitudes A>=delta contribute only O(t^beta), hence zero after
division by t^beta log(1/t). On A<delta, the base small-bar birth
and gap marks have the tight all-mark Gaussian limit in SG/HK;
their multiplication by A makes them smaller than delta times
a tight variable. First discard the bounded-base-scale O(t^beta)
part, then take the base mark-tail cutoff large and delta small.
This gives collapse for bounded continuous tests. It does not
give convergence in a mark space requiring k_F>0 bounded away
from zero. One may use a space including k=0 or retain marks
renormalized by the latent amplitude; the latter is a separately
specified observation convention, not the physical unscaled mark.

There is a precise optional latent mark for the logarithmic range.
Set Theta_t=log A/log t. For fixed 0<u<v<1, the event
Theta_t in [u,v] is A in [t^v,t^u], hence z in
[t^(v-1),t^(u-1)]. Both endpoints tend to infinity, and (21)
makes this interval's deficit integral asymptotic to
(v-u)K_f log(1/t). Amplitudes A>=1 or A<=t contribute O(t^beta)
by (20) or finite expected count. The weighted multi-point error
is still (30). Step-function approximation on Theta intervals
therefore proves the critical PRM with the additional latent mark

      Theta distributed uniformly on (0,1),
      intensity lambda beta a^(-beta)eta_H tensor dTheta.

This mark is asymptotically independent of the retained base fold
marks. It records an amplitude available in the generative model;
it is not substituted for the physical birth/gap marks. There is
no claim that all critical events occur at one amplitude scale.

## 10. Supercritical process: ordinary Poisson with tilted amplitude

Suppose alpha>beta. Now E A^(-beta)<infinity. Conditional on
A, Xi_(F,t) on the retained marks equals the base H process
with small threshold t/A. Its first mean limit, divided by t^beta,
is A^(-beta) eta_H. The uniform bound (20) gives an integrable
majorant C A^(-beta). Dominated convergence yields the mixed
mean limit E A^(-beta)eta_H. The same argument with (2) gives

      E[N_F(Tt)1{N_F(Tt)>=2}]=o(t^beta).               (31)

Thus for n t_n^beta -> lambda the independent-copy process tends
to the ordinary Poisson random measure with intensity

      lambda E A^(-beta) eta_H.                       (32)

The very small A of order t has probability O(t^alpha), lower
order than t^beta; this is also visible in the dominated bound.

Unlike the critical case, full physical birth/gap marks can be
retained in this regime. Their actual intensity is the mixture

      lambda E[A^(-beta) (T_A)_* eta_H^full],
      T_A(p,v,b,k,x,u)=(p,v,A b,A k,x,u),              (33)

where eta_H^full is the base full marked one-bar measure. The
same integrable majorant and bounded tests prove this statement.
It is the tilted amplitude distribution proportional to A^(-beta)
times its original law, not the original unweighted amplitude
distribution and not unchanged Gaussian birth/gap marks.
Explicitly its density on (0,a) is
(alpha-beta)a^(-(alpha-beta))A^(alpha-beta-1).

## 11. Summary of exact regimes and limits

Here nu and m are per-volume measures; eta_H is the whole-volume
base process measure. Let c=c_I and M_alpha be (4).

| Regime | Leading density asymptotic | Leading cumulative-count asymptotic per volume | iid normalization | Process limit on degree/lifetime/location/direction |
| --- | --- | --- | --- | --- |
| 0<alpha<beta | alpha a^(-alpha)M_alpha t^(alpha-1) | a^(-alpha)M_alpha t^alpha | n t^alpha -> lambda | Full-barcode Poisson clusters, (22)-(23) |
| alpha=beta | beta a^(-beta)c t^(beta-1)log(1/t) | a^(-beta)c t^beta log(1/t) | t_n -> 0 and n t_n^beta log(1/t_n) -> lambda | Ordinary PRM, lambda beta a^(-beta)eta_H |
| alpha>beta | c E A^(-beta)t^(beta-1) | (c/beta) E A^(-beta)t^beta | n t^beta -> lambda | Ordinary PRM, lambda E A^(-beta)eta_H |

The scalar identities are exact at every positive lifetime. The
asymptotics and process conclusions consume the stated actual base
premises. Below beta the Lévy law depends on the whole barcode,
so different degrees/locations can be linked within each parent
cluster. At and above beta the proved weighted-cluster bounds
remove multiple points from one copy at the relevant normalization.
Every limit concerns independent whole copies of one fixed law.
There is no spatial expanding-domain law, coalescing-contact
factorial estimate, within-one-field Poisson assertion or regional
witness discharge.

The Laplace limits also determine process convergence on finite
lifetime windows. On the retained mark space, degree/location/
direction are compact and a fixed window [epsilon,T] is compact.
The normalized expected total counts are bounded by the exact
cumulative asymptotics. The expected contribution from lifetime
at most epsilon tends to a constant times epsilon^alpha below
beta or epsilon^beta at/above beta, and vanishes as epsilon goes
to zero. This supplies tightness including the endpoint zero.
The limiting first measures have no lifetime atom at T, so the
window boundary does not create an exceptional truncation. For
the subcritical extra rescaled birth/gap marks, the active parent
measure and child first intensity are finite on each window;
their dominated pushforwards in (23) give the necessary mark-tail
truncation as well. No global factorial-moment criterion is used.

The covariance-matched mixture supplies an actual falsifier of
covariance-only universality. It does not contradict a Gaussian
spectral theorem that explicitly assumes the Gaussian coefficient
law and the needed contact/envelope/selection interfaces. The
amplitude density, independence, normalization and mark scaling
are necessary declared parts of this counterexample.

## 12. Incorporated source identities, actual read extents and review custody

The current repository sources below are the operational bindings. Their
full files were freshly rehashed in the detached construction checkout;
this identity check is not four new whole-proof reads by the constructor.

| Input | Current repository source | UTF-8 bytes / lines | SHA256 |
| --- | --- | --- | --- |
| SG | [spectral_gaussian_admission/PROOF.md](../spectral_gaussian_admission/PROOF.md) | 42478 / 826 | `6c0f34cb318d337b3b8c2c306f3b965f5e5a9d2fe38d5e001b8044de74b7f225` |
| HK | [higher_homology_fold/PROOF.md](../higher_homology_fold/PROOF.md) | 46174 / 911 | `87f43e6ed94c5c63c385ecf177f68212afa5db54b24062fa008f081f47a78999` |
| ST | [fold_structural_universality/PROOF.md](../fold_structural_universality/PROOF.md) | 43927 / 804 | `3d952a906553d578876456539a8f0dcd64aedd45099e8394cbe9db99866ec531` |
| PP | [iid_short_bar_process/PROOF.md](../iid_short_bar_process/PROOF.md) | 30935 / 618 | `320bc988c468bea4938bb777a2d05952c6ab8e0a384f33725b50de847aeec1eb` |

The mathematical author originally consumed and freshly hashed the external
SG preparation, 42,038 bytes / 825 lines, SHA256
`43e8180947b559e53f7afcf95ba61136f3fffcbe05a19e7a5953018fda27e294`,
and HK preparation, 44,258 bytes / 884 lines, SHA256
`de1cad971e113f097a8e29ba82fb295d93b23f57aeda5b265e6f3c88cef47f2e`.
ST and PP were consumed at the same current proof identities in the table.
These historical author inputs are retained separately from the current
repository bindings; no new whole current-SG or current-HK author read is
inferred from incorporation.

The original/current SG Sections 1-11 are identical: 36,656 bytes, SHA256
`4cad43d3f80a8e47da886762d402c4981fe6834e278579b36673055539d6cf32`.
The original/current HK Sections 1-12 are identical: 37,220 bytes, SHA256
`c770d4f7a28bff3d9f59a5baa004f052fb6db2c9e27a11799db0e7420d124471`.
The constructor compared the actual cores in this incorporation. The
current input wrappers have their own source-bound review records; core
identity does not turn an earlier review into a new whole-source review.

The author had prior full actual-byte reads of all four original inputs,
including the then freshly saved HK author read. During the amplitude
preparation task the author additionally reread ST lines 640-699 for the
historical amplitude diagnostic and HK lines 525-748 for density, cone
constants and all-degree endpoint/cluster premises. SG and PP hashes were
checked without claiming two additional whole reads. SG Section 4 supplies
finite expected critical count and all-Borel counting; Sections 7-8 supply
the H0 density and actual cluster/mean premises. HK Sections 7, 10-11 supply
the higher-degree counterparts. The earlier ST lower-bound diagnostic is
not falsely rebound to this stronger exact amplitude law.

At original assignment the second HK whole-source audit was pending, as
retained in Section 1. This is an explicitly historical statement. The
amplitude proof consumes the exact HK mathematical premises, with H0 a
specialization independent of the higher-degree premises. Later input
review records do not retrospectively replace that assignment chronology.

The original amplitude preparation is
`spectral-amplitude-crossover-preparation.md`, 33,421 bytes / 720 lines,
SHA256 `5b1859080a137aed9eeeba20e1b6f0c5a61387e5eb6fa20bb1b8704860fb73d2`.
`final_contract_review` freshly read all 720 saved lines in untruncated
ranges 1-240, 241-480 and 481-720, and reproduced the four declared input
hashes. `benchmark_formal_audit` separately completed a full 720-line
saved-source review. Both returned AMEND: the Section 9 normalization
alone admits a sequence approaching one, so it must also require
`t_n -> 0`; the table must label its density and count formulas as leading
asymptotics. No original-source verdict is rebound to its successor.

The corrected external preparation is
`spectral-amplitude-crossover-corrected-preparation.md`, 34,110 bytes /
730 lines, SHA256
`9e8f10162e9fa21f308454df9463459cdd67ef39a2d8103a78a397b4ab82dd0d`.
Its author completed a full saved-source readback. `benchmark_formal_audit`
made a fresh FULL 730-line mathematical read in untruncated ranges 1-365
and 366-730, read the complete predecessor diff and rehashed the four
inputs: PASS, no must-fix. `root` also completed a full actual-730-line
read. `final_contract_review` read the complete actual 720-to-730 diff and
the corrected blocks, reproduced the four input hashes, and returned a
bounded correction PASS. Reversing only the declared correction recovered
the original 720-line bytes exactly. That audit did not claim a new whole
730-line read. The constructor now preserves Sections 1-11 exactly, checks
all current source/core identities, and authors only the repository wrapper,
review custody and README addition. This is an integration contribution,
not a new nonauthor mathematical review of the incorporated file.

The coordinator supplied the positive multi-maximum P5 witness, void-rate
diagnostic and normalization/mark cautions. `final_contract_review`
contributed bounded prefile attacks of the critical logarithmic Cesaro /
singleton argument and subcritical Laplace integrability, including the
physical birth/gap collapse caution. These are prior proof-route
contributions, separately exposed from saved-source reviews. The same
reviewer also authored a separate fold-chart route and supplied carrier
witness attacks in earlier work; organizational independence remains zero.
The mathematical author authored the SG, ST, physical-law, process and HK
preparations. All performances are from the same OpenAI/Codex team.

No new primary-source fetch is credited to this amplitude construction or
its preparation reviews. The scalar identities, Mellin/logarithmic limits,
Gaussian tail small-ball argument and iid Laplace calculation are explicit
in Sections 1-11. Classical Morse/area/counting imports retain the bounded
and full read extents declared by the exact input proofs. These are not
reverified by a numerical test or a newly claimed full primary-book read.

The companion [review record](REVIEW.md) records construction-stage PENDING
states for the new incorporated full-source and metadata reviews. Publication,
attachment, hosted checks, merge and receipt bindings will be separately
recorded only when obtained. They are not established by this file or by
completed preparation reviews. No scientific-status promotion, critical
Lean instantiation, empirical sampler admission, computational/count coupling,
blind seed use or same-field spatial-process result is claimed.
