# A uniform local-norm route for finite-carrier diagonal counts

16 October 2026 UTC. Joint proposal by OpenAI/Codex `/root` (initial mathematical sketch) and `/root/core_theorem_audit_20261015` (independent reconstruction, explicit constants, falsification controls and this report). Actual organizational-independence credit **0**. This new document is pending separate nonauthor review. No production/remote/native write, ref claim, scientific-status transition, Lean execution or source admission occurred.

## 1. Result and the added family hypothesis

The global coefficient-norm argument carried an exponential prefactor `2^(N/2)` and a derivative constant depending on the whole coefficient vector. It did not give an increasing-domain bound. This proposal replaces that step by finite Gaussian Gram matrices of **localized physical Sobolev norms**. Their traces, rather than their ranks, control the exponential moment. A deterministic physical cover removes any need to condition at a field-selected maximum.

For the finite full-square Gaussian family defined below, assume

```text
L>=Lmin>0,
sum_n q_n=1,  q_n=q_-n>0,
sup_family sum_n q_n |(2pi/L)n|^12 <= M12 < infinity.    (U)
```

The last condition is a **uniform physical spectral moment**, in addition to the individual smoothness of each finite field. The normalization `sum q_n=1` is an essential separate uniform variance bound, including the DC mode. Without normalization one must also assume `sum q_n<=M0<infinity`; every occurrence of `1+M12` below is then replaced by `M0+M12`. M12 alone does not control the constant or low-frequency variance. These family assumptions are not asserted for every changing sequence of the project's spectra. Under (U), fixed positive transverse/cubic lower marks a,k_-, fixed 0<rho<1, and the endpoint/type/cutoff conventions below, constants c,C>0 independent of K,L and coefficient dimension give

```text
P(two distinct qualifying candidates form an eta-cluster)
       <= C (1+L)^2 exp(-c/eta),
E D(t,eta) <= C (16K^2)^4 (1+L)^2 exp(-c/eta).          (1)
```

The cluster requires center distance at most eta and both endpoint radii at most rho eta. Distinct candidates sharing their maximum or saddle are included; the identical-candidate diagonal is excluded. The count is by ordered endpoint identity, not by continuous frame labels. All constants and small-scale restrictions are supplied below.

With K=K(t), L=L(t), define

```text
H(t)=8log K(t)+2log(1+L(t))+4log(1/t),
eta(t)=c/H(t),       0<t<1.                            (2)
```

When `t^(1/3) H(t)->0`, the compact-mark lifetime radius bound lies below rho eta(t) eventually, and (1) gives

```text
E D(t,eta(t)) <= C 16^4 t^4,
t^(-4/3) E D(t,eta(t)) = O(t^(8/3)) -> 0.              (3)
```

In particular polynomial growth of K and L in 1/t satisfies the radius condition. A family with K/L uniformly bounded automatically satisfies (U), regardless of its positive normalized weights. Thus K and L comparable to `t^(-1/3)` are allowed by this diagonal estimate. No one-point intensity asymptotic, far-field mixing, global selected-bar identity, infinite-carrier result or spatial point-process limit is inferred.

## 2. Exact inherited source and review identities

The finite source convention is Main `85b928b326ff0586715a56ae91f51d3fbd89557a`, [experiments/universality/finite_gaussian_contact/PROOF.md](https://github.com/d6g8k5htny-coder/main/blob/85b928b326ff0586715a56ae91f51d3fbd89557a/experiments/universality/finite_gaussian_contact/PROOF.md), Git blob `5f345201309cfa1781e92d6292f542de4bcb2999`, 25,367 bytes, SHA-256 `7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474`. Its actual law and literal endpoint-distance normalization are preserved. Its original fixed-law arguments do not themselves establish family hypothesis (U).

The following local inputs remain unchanged:

| Input | SHA-256 |
| --- | --- |
| [QUANTITATIVE_FOLD_ISOLATION_PROPOSAL.md](QUANTITATIVE_FOLD_ISOLATION_PROPOSAL.md) | `827c82953845eb12005542ff6a4e1c99886b68d1ea9f1aeae16f981622ac6a30` |
| [NONAUTHOR_FOLD_ISOLATION_REVIEW.md](NONAUTHOR_FOLD_ISOLATION_REVIEW.md) | `f06c61e3f3e60c795a1a14e824640f1689d4143dba4f6698cada7a41929eec90` |
| [FINITE_FOURIER_CRITICAL_COUNT_DIAGONAL_ROUTE.md](FINITE_FOURIER_CRITICAL_COUNT_DIAGONAL_ROUTE.md) | `621cee96854aa440498cecf5452a8d1e23f90a4458779c3286c18eba106fe138` |
| [NONAUTHOR_FINITE_COUNT_DIAGONAL_REVIEW.md](NONAUTHOR_FINITE_COUNT_DIAGONAL_REVIEW.md) | `16152427f7013822ac1bba81142429800334e6bf0f354a6439b23a1997bc4524` |

The finite-count review was read in full. Its verdict is ACCEPT_SCOPED for that exact fixed-law argument. It explicitly does not supply expanding-domain uniformity. The accepted local chart and a.s. critical count are reused with their premises; their verdicts do not extend automatically to this new norm/cover argument. All reviews/authors are OpenAI/Codex, source-exposed, organizational-independence credit 0. The previous six author files and both review files have their frozen hashes checked by the new control program.

The classical analytic import is Fourier inversion, differentiation and Parseval/Plancherel on Schwartz functions. These were inspected in Leon Simon, [*Lectures on PDE*, Lecture 5, Fourier properties (a)–(h), PDF page 26, with its Sobolev proof on page 27](https://math.stanford.edu/~lms/lecs-on-pde.pdf). Simon uses the unitary convention adopted below. The explicit H6-to-C4 operator constants here are derived by a displayed Cauchy–Schwarz calculation, rather than imported as an unnamed uniform embedding constant. Only the relevant portions of Lecture 5 were read, not the entire text. A saved primary PDF and byte identity are in [uniform-local-norm-primary-sources/SOURCE_IDENTITY.json](uniform-local-norm-primary-sources/SOURCE_IDENTITY.json).

## 3. Uniform physical derivative variances

For each K>=2 and L>=Lmin, keep the actual ideal law

```text
omega=2pi/L,  S_K={-K,...,K}^2,
F(x)=sqrt(q_0) C_0
   + sum_nonzero_pairs sqrt(2q_n)
        [C_n cos(omega n.x)+S_n sin(omega n.x)],         (4)
```

with N=(2K+1)^2 independent standard real Gaussian coordinates Z. Weights may vary with K,L but obey (U). The law is centered, stationary and variance one. It need not be isotropic. All reasoning remains on the same F; there are no independent-copy replacements.

For a coordinate multi-index beta of order j<=6, the sine/cosine pair variances add exactly to give

```text
Var(partial^beta F(x))
 = sum_n q_n product_l ((omega n_l)^(2beta_l))
 <= sum_n q_n |omega n|^(2j)
 <= 1+M12.                                            (5)
```

The last inequality uses `r^(2j)<=1+r^12` for r>=0 and j<=6; it includes j=0. This bound is uniform in x,K,L. In covariance notation, `sum q_n|omega n|^12=(-Delta)^6 Cov(F(0),F(h))|_(h=0)`. Condition (U) controls this physical twelfth-order quantity. A cutoff K or period L by itself is not that condition.

A simple nonempty family with automatic (U) is

```text
K/L<=Lambda<infinity.
```

Then `|omega n|<=2pi sqrt(2)Lambda` on the whole square, and one can take `M12=(2pi sqrt(2)Lambda)^12`. No lower bound on the individual positive weights is needed for the uniform norm estimate. Positivity remains required separately for the existing real-parameter genericity/Bézout argument.

## 4. An explicit fixed physical cutoff and trace bound

Put

```text
b=min(1,Lmin/16)>0,    R0=b,    eta0=b/4.
```

Define the flat function and smooth step

```text
psi(t)=0 for t<=0,  exp(-1/t) for t>0,
J(t)=psi(t)/(psi(t)+psi(1-t)),
theta(s)=1-J(|s|-2),
chi_b(x1,x2)=theta(x1/b) theta(x2/b).                   (6)
```

The denominator is positive. The step is 0 on t<=0 and 1 on t>=1. Exponential flatness at zero makes it C-infinity. Although absolute value appears in theta, theta is identically one near s=0, so there is no nonsmooth point there. Thus chi_b is a fixed C-infinity cutoff, equals one on `[-2b,2b]^2`, vanishes outside `[-3b,3b]^2`, and has values in [0,1]. Since 6b<Lmin<=L, every translated support is embedded in the torus, with a periodic lift available on R^2.

Here are completely explicit, deliberately coarse derivative bounds, so cutoff regularity is not an unspecified constant. For j=0,...,6 let L_j(u) be the integer polynomials

```text
L_0=1,       L_(j+1)=u^2(L_j-L_j'),
A_j=sum_m |coefficient_m(L_j)| max(1,m^m),
E_0=8,
E_n=8 sum_(j=1)^n binom(n,j) (2A_j) E_(n-j),
G_n=sum_(j=0)^n binom(n,j) A_j E_(n-j),
d_0=6,      d_j=2G_j^2 for 1<=j<=6.                   (7)
```

Use 0^0=1 in the coefficient bound. On 0<t<=1, `psi^(j)(t)=exp(-u)L_j(u)`, u=1/t>=1. Since `sup_(u>=1) u^m exp(-u)<=max(1,m^m)`, its absolute value is at most A_j. On 0<=t<=1 the denominator in J is at least exp(-2), because at least one of t,1-t is at least 1/2. Its jth derivative is bounded by 2A_j. Differentiating the identity between that denominator and its reciprocal gives the recurrence E_n, since exp(2)<8. Then the Leibniz formula gives `|J^(n)|<=G_n`. For j>=1, theta's derivative is supported on the two intervals of total length 2; consequently

```text
integral_R |theta^(j)(s)|^2 ds <= d_j,
||partial^gamma chi_b||_L2(R^2)^2
          <= b^(2-2|gamma|) d_(gamma1)d_(gamma2).       (8)
```

For j=0, `0<=theta<=1` and support length 6 give d_0=6. The elementary exponential bound used above follows, for example, by bounding the tail of the e series after 1/4!: `e<=65/24+1/100<11/4`, hence e^2<8. These are finite explicit constants, not evaluated derivative suprema from sampled points.

Use the unitary Fourier transform on R^2,

```text
hat f(xi)=(2pi)^(-1) integral exp(-i x.xi) f(x) dx,
||f||_H6^2=integral (1+|xi|^2)^6 |hat f(xi)|^2 dxi.
```

By the derivative rule and Plancherel,

```text
||f||_H6^2=sum_(|alpha|<=6) w_alpha ||partial^alpha f||_2^2,
w_alpha=6!/[(6-|alpha|)! alpha1! alpha2!].              (9)
```

For any deterministic torus basepoint g, let `f_g(x)=chi_b(x)F(g+x)` using the periodic lift. The derivative Leibniz sum and weighted Cauchy–Schwarz give

```text
E ||partial^alpha f_g||_2^2
 <= (1+M12) 2^|alpha|
       sum_(gamma<=alpha) binom(alpha,gamma)
                         ||partial^gamma chi_b||_2^2. (10)
```

For clarity, the weight sum is `sum_(gamma<=alpha) binom(alpha,gamma)=2^|alpha|`. The random derivative in each summand has variance bounded by (5); no independence between summands is assumed. Since b<=1 and |gamma|<=6, `b^(2-2|gamma|)<=b^(-10)`. Define the fixed integer

```text
A_theta=sum_(|alpha|<=6) w_alpha 2^|alpha|
          sum_(gamma<=alpha) binom(alpha,gamma)
                             d_(gamma1)d_(gamma2)
       =430299201784277045424464881586844996,
T=(1+M12)b^(-10) A_theta > 0.                          (11)
```

Equations (8)–(10) prove `E||f_g||_H6^2<=T`, uniformly in K,L,g and weights satisfying (U). The constant is intentionally very loose and is not a usable numerical asymptotic-window claim. A smaller trace bound using the actual cutoff derivative integrals would also work, but is not needed to establish uniformity.

## 5. Dimension-free Gaussian exponential moment

For each individual finite field, write `f_g=sum_(j=1)^N Z_j phi_(g,j)` in H6, where every deterministic phi is a compactly supported smooth function. Let Q_g be its real H6 Gram matrix. It is positive semidefinite, and

```text
S_g^2:=||f_g||_H6^2=Z^T Q_g Z,
tr Q_g=E S_g^2<=T.                                   (12)
```

An orthogonal diagonalization in this finite real coefficient space gives nonnegative eigenvalues lambda_j with sum at most T. Gaussian rotational invariance and the one-dimensional Gaussian integral yield

```text
E exp(S_g^2/(4T))
 = product_j (1-lambda_j/(2T))^(-1/2)
 <= sqrt(2).                                          (13)
```

Indeed, for a_j=lambda_j/(2T)>=0 with sum at most 1/2, induction gives `product(1-a_j)>=1-sum a_j>=1/2`. The estimate includes zero eigenvalues and arbitrary finite rank. It uses neither an infinite-dimensional Gaussian theorem nor a rank-dependent prefactor. In particular,

```text
P(S_g^2>=v)<=sqrt(2) exp[-v/(4T)],    v>=0.             (14)
```

This is the unconditioned Gaussian law (4). We do not transport the almost-sure Bézout count or this trace bound to an arbitrary regression subspace; a conditional mean or an algebraically exceptional subspace would need its own analysis.

## 6. Direct Fourier derivative constants, including rotation

For any unit vectors v_1,...,v_j and j<=4, Fourier inversion and Cauchy–Schwarz give

```text
|D^j f(x)[v_1,...,v_j]|
 <= (2pi)^(-1) [integral |xi|^(2j)/(1+|xi|^2)^6 dxi]^(1/2)
                       ||f||_H6.
```

Polar integration and u=r^2, followed by v=u/(1+u), give exactly

```text
I_j=integral_R2 |xi|^(2j)/(1+|xi|^2)^6 dxi
   =pi integral_0^infinity u^j/(1+u)^6 du
   =pi j!(4-j)!/5!,        0<=j<=4.                   (15)
```

Thus the full multilinear operator norms, without a coordinate-frame inflation, satisfy

```text
sup ||D^3 f||op <= ||f||_H6/sqrt(80pi) <= p3 ||f||_H6,
sup ||D^4 f||op <= ||f||_H6/sqrt(20pi) <= p4 ||f||_H6,
p3=1/15,       p4=2/15.                               (16)
```

The rational majorants follow already from pi>3. Absolute convergence of these derivative Fourier integrals also gives continuity. For the present compact smooth f, it is ordinary classical differentiation; the display supplies the quantitative H6-to-C4 control. Arbitrary directions enter through `product |xi.v_l|<=|xi|^j`, so no isotropy of F or frame-wise coordinate norm assumption is used. Where chi_b=1, all these derivatives are the physical derivatives of F.

## 7. Deterministic cover and the local cluster obstruction

For each L choose `n=ceil(L/b)` evenly spaced points in each torus coordinate, giving n^2 deterministic grid centers. Their spacing is L/n<=b. Every torus point M has a grid center g for which its short lift satisfies

```text
|M-g|_infinity<=b/2,
n^2<=(1+L)^2/b^2.                                    (17)
```

Any rotated closed square of radius R0=b about M is contained in the region where chi_b(x-g)=1, since its coordinates relative to g are bounded by `b/2+sqrt(2)b<2b`. The cutoff support has side 6b<L, so these are embedded lifts even across a canonical torus seam. For eta<=eta0=b/4 and r_i<=rho eta, the first endpoint segment also lies within this inner region. Under the cluster condition `d(z_1,z_2)<=eta`, all endpoints are within the same short neighborhood; in particular `|M_2-M_1|<=(1+rho)eta<=b/2`. The constants in this paragraph cover every eigenframe orientation.

Candidate identity and physical types are exactly those of the accepted finite-count report: p=(M,S), distinct endpoints, both critical, M a nondegenerate maximum, S a nondegenerate index-one saddle, unique short displacement r,u, canonical v=Ju, positive gap ell=k r^3, k>=k_-, and F_vv(M)<=-a. Let D(t,eta) be its ordered sum over **different** candidates with gaps at most Tgap t, center distance at most eta and both radii at most rho eta. Here Tgap>0 is fixed; it is unrelated to the norm-trace constant T in (11). Bounded Borel additional selectors may reduce this count. We use the completed unconditioned Gaussian probability space and the finite-configuration measurability argument of the accepted report.

For one grid center write `S=S_g`, `W=1+S`, and use the local derivative bounds `B=p3 W`, `C=p4 W`. Define constants depending only on b,a,k_-,rho:

```text
tau=6k_-,
kappa=min(b, a/(32p3), tau/[32(p4+p3^2/a)]),
D0=p4/2+3p3^2/a,
gamma=min(a^2/(rho^2 p3^2 eta0),
          6k_-/(rho D0), kappa/(1+rho))>0.             (18)
```

Suppose two distinct candidates form the stated eta-cluster, and choose a covering grid center g for its first maximum M_1. Then

```text
(1+S_g)^2 >= gamma/eta.                               (19)
```

Here is the direct proof, keeping the localization premise visible. If W^2<gamma/eta, all of the first candidate's segment and its rotated radius-b chart have their third/fourth derivatives bounded by B,C, by (16)–(17). Its source endpoint stationarity and weighted cubic identity give, in its oriented soft eigenframe,

```text
F_eee(M_1)>=12k_1-r_1(C/2+3B^2/a).
```

The conditions needed for that conversion hold:

```text
r_1 B<=rho p3 sqrt(gamma eta)<=a,
r_1(C/2+3B^2/a)<=rho eta D0 W^2<rho D0 gamma<=6k_-.
```

Thus the first maximum has hard eigenvalue at most -a and soft cubic at least tau. Its accepted isolation radius is

```text
R=min(b,a/(32B),tau/[32(C+B^2/a)])
       >=kappa/W^2>(1+rho)eta.                        (20)
```

If the maxima are distinct, the second is at distance at most (1+rho)eta from the first, contradicting maximum uniqueness in the chart. If they share their maximum, distinct candidate identity forces two different saddles, each at distance at most rho eta<R, contradicting the chart's at-most-two-critical-points conclusion. A shared saddle with distinct maxima falls in the first case. Cross-role equality is impossible under the physical types. Identical candidates are excluded and would not be a contradiction. This proves (19).

The actual center separation may be zero or much smaller than either radius. No hypothesis `r_i<=rho times actual_center_distance` is introduced. The chart provides no full enclosing height barrier or global elder selection; the frozen cubic and mixed-cubic falsifiers remain attached to its use.

## 8. Unconditional union, integrated count and growing-domain cutoff

Let `eta_cap=min(eta0,gamma/4)`. For 0<eta<=eta_cap, (19) implies `S_g^2>=gamma/(4eta)` for at least one deterministic covering center. Consequently

```text
{D(t,eta)>0} subset union_(deterministic g)
                  {S_g^2>=gamma/(4eta)},
P(D(t,eta)>0)
 <=sqrt(2) n^2 exp[-gamma/(16T eta)]
 <=sqrt(2)b^(-2)(1+L)^2 exp(-c/eta),
c=gamma/(16T)>0.                                     (21)
```

Although the witnessing maximum and its covering center depend on F, the union is over a predetermined finite cover. Each probability is unconditioned. We do not condition a local norm at a chosen maximum or assume the different cubes are independent. The same trace bound T applies to each deterministic center.

The accepted algebraic argument gives `Ncrit<=16K^2` almost surely for **each** positive full-square finite law, including every member of a varying family. It needs no uniform positive lower weight, no uniform covariance inverse and no Morse premise. The candidate population contains at most Ncrit^2 endpoint pairs. Thus, almost surely,

```text
D(t,eta)<=(16K^2)^4 1{D(t,eta)>0}.
```

There is no need for a common probability-one event over an uncountable family of laws: this expectation estimate is proved for each member with the same displayed deterministic constants. Combining it with (21) proves (1), explicitly

```text
E D(t,eta)
 <= Ccov 16^4 K^8 (1+L)^2 exp(-c/eta),
Ccov=sqrt(2)b^(-2).                                  (22)
```

This is an integrated candidate factorial count, not a pointwise Rice-density bound. Setting the count to zero on the source-specific exceptional null set is harmless. No four-site observation covariance or Hessian determinant weight is inserted a second time.

For the full lifetime band 0<ell<=Tgap t with k>=k_-, let `A=(Tgap/k_-)^(1/3)`. Then every radius is at most A t^(1/3). Choose H(t) and eta(t) from (2). The exact algebra is

```text
exp(-c/eta(t))=K(t)^(-8)(1+L(t))^(-2)t^4.
```

Since H(t)>=4log(1/t), eta(t)<=eta_cap eventually. If additionally

```text
t^(1/3)H(t)->0,                                      (23)
```

then `A t^(1/3)<=rho eta(t)` eventually. Thus (13) of the accepted report, with this eta, counts every two distinct candidates in this compact-fold lifetime sector whose centers have distance at most eta(t). Equation (22) proves (3). A sufficient general growth condition is

```text
log K(t)+log(1+L(t))=o(t^(-1/3)).                      (24)
```

Polynomial K,L in 1/t satisfy it. More generally one may replace the logarithmic coefficients 8,2,4 by p>=8, q>=2, u>4/3 to obtain a vanishing t^(-4/3) normalization, with the corresponding radius condition. These are exact cancellations of the coarse count/cover powers, not an asymptotic independence argument.

For example take L=t^(-1/3) and K=ceil(L), with arbitrary positive symmetric normalized weights. For t sufficiently small, K/L<=2, so (U) holds with a fixed M12. The numerator/denominator scales in (2) give `H(t)=(22/3)log(1/t)+O(1)` and eta of logarithmic order; A t^(1/3)/eta tends to zero. The bound is therefore compatible with this increasing-domain regime. Calling this a constant-total-intensity regime additionally requires a **separate** one-point intensity asymptotic; none is proved or assumed here.

The displayed D is a total ordered endpoint count. Dividing it by L^2 or L^4, if a later explicitly stated convention requires per-volume or per-volume-pair normalization, can only reduce the bound uniformly up to an Lmin-dependent constant. No such division is hidden in (22). The requested t^(-4/3) factor is a twofold cumulative-lifetime scale. It is not a proof of lifetime density ell^(-1/3) or of cumulative mass t^(2/3).

## 9. Falsifiers and limits of reuse

**M12 alone does not control variance.** If normalization is dropped, take `F_A=A Z_0+G`, with Z_0 an independent standard real Gaussian and G a fixed finite full-square field with all nonconstant mode weights positive. Its twelfth physical spectral moment is independent of A because the DC mode contributes zero. But `E||chi_b F_A||_H6^2=A^2||chi_b||_H6^2+E||chi_b G||_H6^2` diverges. This explicitly falsifies a uniform local trace from M12 alone. It is outside (U), whose normalized total mass controls DC and every lower moment. A separate uniform M0 bound is the alternative stated above.

**Individual smoothness does not imply a uniform local trace.** Fix a physical period, let K grow, put a fixed positive fraction of the normalized spectral mass on the two axial frequencies +/-K, and distribute the remaining mass positively over the full square. Then M12 grows at least as K^12. On the inner cutoff square chi_b=1, so

```text
E ||chi_b F||_H6^2
 >= integral_inner E |partial_x^6 F|^2
 >= (positive fixed area and mass) times (omega K)^12.
```

A dimension-free bound with a fixed T would imply a uniform second moment and is impossible for this family. Even the fourth derivative at one fixed physical point has variance of order K^8. This is a literal obstruction to removing (U) merely because each field is a finite C-infinity Fourier sum. It is not a counterexample under (U).

**Very fast growth can defeat the radius comparison despite (U).** If `log K(t)` grows as t^(-1/2), then the cutoff (2) is at most of order t^(1/2), whereas the lifetime radius upper bound is of order t^(1/3). The latter is not o(eta). Uniform (U) can still be arranged by assigning sufficiently small positive weight to high physical frequencies, so this failure is distinct from the moment failure. It shows that the count/cover logarithmic route cannot cover the whole stated lifetime sector at that growth rate by this bound alone. It does not disprove the occurrence bound when its radius cutoff is actually imposed, or any point-process conjecture.

**Moving mark restrictions must be identified by what they affect.** Growing upper height or upper cubic-mark thresholds do not harm (21)–(22) when the fixed lower a,k_- and fixed lifetime cap Tgap remain: this proof uses no such upper bounds. Increasing lower transverse/cubic thresholds also only restricts the counted population, so fixed positive lower bounds still suffice. In contrast, allowing the lower transverse curvature or cubic mark to tend to zero removes the uniform gamma in (18). Allowing Tgap=Tgap(t) to grow changes (23) to `Tgap(t)^(1/3)t^(1/3)H(t)->0`; for example Tgap=1/t makes the radius bound constant, so it cannot be automatically below a cutoff tending to zero. Those are scope failures, not conclusions against the estimate with its actual premises.

The full-square assumption is used for the a.s. critical-count theorem, not for the local Gram trace. Infinite Fourier carriers have no finite polynomial Bézout cap supplied here. The same is true of arbitrary conditioned coefficient laws that may lie inside the algebraic bad set. Non-Gaussian coefficients need a different exponential-moment argument. The argument uses fixed positive geometric lower marks and a physical Lmin. Neither small-mark integration nor a source-law admission to (U) is filled by the word smooth.

Even under (U), the moving far-center complement, uniform one-point remainders/intensity, actual global H0 bar correspondence, essential classes, higher homology, mixing, increasing-domain process tightness and identification of a limiting point process remain separate. No independence follows from the deterministic union. The accepted K2/K3 enlarged height-covariance singularities are unchanged; this integrated route avoids that density for this sector rather than repairing it.

## 10. Controls and review custody

[uniform_local_norm_controls.py](uniform_local_norm_controls.py) and [UNIFORM_LOCAL_NORM_CONTROLS.json](UNIFORM_LOCAL_NORM_CONTROLS.json) use exact integers and Fraction arithmetic, with no floating arithmetic or field samples. They verify the beta-integral constants and rational operator majorants, the explicit cutoff recurrence and trace integer, multinomial/Leibniz sums, 256 finite product margins across varying dimensions, deterministic cover/rotation margins, logarithmic cancellation, the growing-lifetime-cap radius failure, and unchanged identities for the six prior author files and two accepted reviews. All checks passed.

The controls do not mechanically establish the continuous cutoff derivative estimates, Plancherel, all-parameter trace bound, Gaussian integral, local analytic chart, algebraic genericity, or asymptotic theorem. Those are the conventional derivations/imports stated in this report. The large explicit constants are not certified useful experimental windows. No Gaussian draws, approximate critical roots, PRNG seeds, Rice integrals, field-level experiments or process simulations occurred.

Current lineage is root sketch -> this agent's independent analytic reconstruction and falsification controls -> pending separate nonauthor review of these new bytes. The two previous ACCEPT_SCOPED reviews are consumed only within their frozen scope; this proposal needs its own review. All authors/reviewers so far are OpenAI/Codex, with organizational-independence credit 0. Previous files and reviews remain unchanged. These new local files and a primary-source copy have no production/native/remote or scientific-status effect.
