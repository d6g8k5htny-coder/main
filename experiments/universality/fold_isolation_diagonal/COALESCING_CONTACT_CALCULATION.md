# Coalescing two-contact calculation: a collinear cancellation and a finite-carrier obstruction

16 October 2026 UTC. Read-only mathematical analysis by OpenAI/Codex `/root/core_theorem_audit_20261015`. No native source write, source claim, PR, graph update, proof-status transition or process theorem. This report derives restricted-geometry estimates on one actual finite Gaussian field. The decimal calculations at the end are diagnostics only.

## 1. What changes the next action

There are two concrete results.

1. **A useful restricted estimate.** For four nearby critical sites lying exactly on one line, the joint eight-gradient density has a Vandermonde singularity `V^-2`. The product of the four physical Hessian determinant weights is bounded by `C V² (1+||F||C5)^8`. These factors cancel in the gradient-pinned four-point Rice kernel. For two disjoint small pairs with center separation eta, requiring a positive lower cubic-gap mark `k_i>=k_->0` forces `||F||C5 >= c k_-/eta²`; consequently the same kernel is at most `C exp(-c k_-²/eta^4)`. Any actual H0 selectors bounded by one can be retained in that upper bound. No global pairing or iid replacement is required.
2. **A real obstruction to the naive height-density route.** On a coordinate-aligned line, the twelve value/gradient observations at four distinct sites have rank 9 for full-square K2 and rank 11 for K3, rather than rank 12. Two limiting six-row fold contacts have the same deficient ranks. Full positivity of the finite square does not fix this. In K2, two distinct aligned exact positive cubic contacts are actually incompatible with the field's support. The infinite-law K7/K5 ranks in the current process source therefore cannot simply be inherited by these finite carriers.

The next mathematical step should use gradient-pinned Rice plus actual height-gap indicators and a geometry-dependent divided-difference analysis, or strengthen the carrier and stratify the height-observation singularities explicitly. It should not assume a uniform twelve-dimensional density for the K2/K3 catalogue. The collinear estimate still leaves transverse collision geometry and the small-k boundary unestimated; it is not a global factorial or spatial Poisson theorem.

## 2. Source, law, observable and scale

Pinned source: Main `85b928b326ff0586715a56ae91f51d3fbd89557a`.

- [iid_short_bar_process/PROOF.md §6](https://github.com/d6g8k5htny-coder/main/blob/85b928b326ff0586715a56ae91f51d3fbd89557a/experiments/universality/iid_short_bar_process/PROOF.md#L409) proves the separated-center, sufficiently small-radius twofold contribution `O_eta0(t^(4/3))` for the actual **infinite** periodized Gaussian law. Its four-site K7 and two-contact K5 interpolation are explicitly not transferred from finite K2/K3.
- [finite_gaussian_contact/PROOF.md §§2–5](https://github.com/d6g8k5htny-coder/main/blob/85b928b326ff0586715a56ae91f51d3fbd89557a/experiments/universality/finite_gaussian_contact/PROOF.md#L94) supplies the actual finite Fourier law, covariance multiplier identity, literal six observation rows, target and physical Hessian weights. Its joint seven-row covariance is a **single-contact** result.
- [finite_candidate_lifetime/PROOF.md](https://github.com/d6g8k5htny-coder/main/blob/85b928b326ff0586715a56ae91f51d3fbd89557a/experiments/universality/finite_candidate_lifetime/PROOF.md) and [finite_h0_lifetime/PROOF.md](https://github.com/d6g8k5htny-coder/main/blob/85b928b326ff0586715a56ae91f51d3fbd89557a/experiments/universality/finite_h0_lifetime/PROOF.md) keep candidate and selected populations distinct and put the Hessian determinant into the Rice weight once.

Fix K>=2, L>0 and omega=2pi/L. The field is the source's planar ideal Gaussian Fourier law:

The four input bodies were independently checked against Main's immutable Git tree:

| Input | Git blob | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| iid_short_bar_process/PROOF.md | `cd7ef8b8a364cb394be48208f55332658e9fff0b` | 30935 | `320bc988c468bea4938bb777a2d05952c6ab8e0a384f33725b50de847aeec1eb` |
| finite_gaussian_contact/PROOF.md | `5f345201309cfa1781e92d6292f542de4bcb2999` | 25367 | `7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474` |
| finite_candidate_lifetime/PROOF.md | `a27a84150b3e70bfd794672eeeed6915425a1da4` | 29190 | `ed31f562f0ffaaf6d4bd2ceb1ae3d234798e983a55d55c6983b128fce27133f5` |
| finite_h0_lifetime/PROOF.md | `7a3518edd9bd04adf7aac04f3a5a40bdaab78a48` | 27064 | `5ea91111fa5f3d78d7c4188ebe84b3bc2c452b05dfdfc62c74b03123cb2696d1` |

```text
S_K = {-K,...,K}²,
q_n=q_-n>0, sum q_n=1,
F(z)=sqrt(q_0) C_0
  + sum_nonzero_pairs sqrt(2q_n)
      [C_n cos(omega n.z)+S_n sin(omega n.z)],
```

with independent standard real Gaussian coefficients, one representative per pair. No cutoff, weights, coefficient distribution or source normalization is changed in the analysis. Constants may depend on this fixed law and on L,K. They are not uniform as positive weights approach zero, over unbounded K, or over variable L.

Choose an orthonormal frame (u,v), a short embedded line segment on the torus and four ordered distinct sites

```text
z_i = z + x_i u,                 x_1<x_2<x_3<x_4.
M_1=z_1, S_1=z_2, M_2=z_3, S_2=z_4,
r_1=x_2-x_1, r_2=x_4-x_3,
eta=(x_3+x_4-x_1-x_2)/2.
```

Assume `r_i<=rho eta` for a fixed `0<rho<1`, for example rho=1/4, and `(1+rho)eta` small relative to the injectivity scale and to a fixed covariance-normalization radius. Define

```text
V = product_(i<j)(x_j-x_i)>0,
D_i = product_(j!=i)|x_i-x_j|,
G = (F_u(z_1),...,F_u(z_4), F_v(z_1),...,F_v(z_4)),
H_i = Hess F(z_i),
ell_i=F(M_i)-F(S_i),       k_i=ell_i/r_i³.
```

The exact kernel estimated below is

```text
R(x_1,...,x_4;B)
 = p_G(0) E[ product_i |det H_i| * B(F) | G=0 ],
```

where B is any Borel mark in [0,1]. For the compact-positive-gap bound it includes `k_1>=k_-` and `k_2>=k_-`; it may also include `ell_i<=Tt`, maximum/saddle Hessian types and the two actual elder selectors. These marks are all from the **same F**. The proof only bounds selectors by one and never assumes they are a bijection or that contacts are independent.

This is an integrand on a collinear position stratum. That stratum has zero full spatial measure; a bound there alone cannot be integrated as a bound on all four-site configurations.

## 3. Uniform gradient-jet rank at K2 already suffices for the collinear density

Let T_x send four scalar values at x_1,...,x_4 to their successive Newton divided differences of orders 0,1,2,3. Its triangular diagonal gives

```text
|det T_x|=1/V.
```

Apply T_x separately to the four F_u and four F_v evaluations. With `U=diag(T_x,T_x)G`,

```text
|det(G -> U)|=V^-2,
det Cov(G)=V^4 det Cov(U).                              (1)
```

When the segment diameter tends to zero, the rows of U tend uniformly, including arbitrarily unequal node spacings, to

```text
J = (F_u, F_uu, F_uuu/2!, F_uuuu/3!,
     F_v, F_uv, F_uuv/2!, F_uuuv/3!)(z).                (2)
```

One elementary justification of uniformity is the integral representation of an order-j divided difference as an average of the jth derivative divided by j!, over the simplex spanned by its nodes. It follows by repeated use of the fundamental theorem of calculus; the positive averaging measure has fixed mass 1/j!. Thus the difference from the coincident-node derivative is at most the derivative's modulus of continuity on the segment. Finite Fourier derivatives have that uniform modulus in mean square, uniformly in frame.

The covariance of J is positive definite for every frame and K>=2. To see this directly from the source covariance formula, a zero-variance linear combination gives a frequency polynomial that is a linear combination of

```text
a, a², a³, a^4, c, ac, a²c, a³c,
a=n.u, c=n.v,                                          (3)
```

with nonzero omega, i and factorial factors. Its degree in each original frequency coordinate is at most four. Positivity of q makes it vanish on the full grid with at least five coordinates in each direction. The elementary polynomial root argument in each coordinate makes the polynomial identically zero; the invertible frame change makes the eight coefficients vanish. No arbitrary-jet theorem is imported.

Continuity and compactness of the frame group give a positive covariance floor for J. The uniform row convergence therefore yields constants c,C>0 for all sufficiently short segments and every configuration of four distinct nodes:

```text
c I <= Cov(U) <= C I.
```

Equation (1) gives the exact singular order, uniformly in all node ratios,

```text
p_G(0) = (2pi)^-4 V^-2 / sqrt(det Cov(U)) = Theta(V^-2). (4)
```

For the two-pair scale,

```text
(1-rho)^4 r_1 r_2 eta^4 <= V
  <= (1+rho)^4 r_1 r_2 eta^4.
```

Hence the raw gradient density grows like `r_1^-2 r_2^-2 eta^-8`. This is an actual covariance calculation for the finite field.

## 4. Four pinned gradient zeros force precisely the matching Hessian vanishing

Set `g(s)=F_u(z+s u)` and `h(s)=F_v(z+s u)`. On the G=0 fiber, both functions vanish at all four x_i.

For any C4 scalar function a with these four zeros,

```text
|a'(x_i)| <= D_i * sup |a^(4)| / 4!.                   (5)
```

Proof: for a point s different from the four nodes, subtract from a the degree-four polynomial `a(s) product_j(t-x_j)/product_j(s-x_j)`. The difference has five zeros. Four applications of Rolle's theorem give the Lagrange remainder bound `|a(s)|<=sup|a^(4)| product_j|s-x_j|/4!`. Divide by |s-x_i| and pass to x_i. This establishes (5) without a conditional genericity assumption.

Apply (5) to g and h:

```text
|F_uu(z_i)| <= D_i ||F_(u^5)||_infinity /24,
|F_uv(z_i)| <= D_i ||F_(u^4 v)||_infinity /24.           (6)
```

Take a fixed coordinate C5 norm and let M=1+C||F||C5 dominate all indicated directional derivatives and F_vv. With the segment radius small enough that D_i<=1,

```text
|det H_i|
 = |F_uu F_vv-F_uv²|(z_i)
 <= C M² D_i.
```

Every pairwise distance appears twice in `product_i D_i`, so

```text
product_i |det H_i| <= C M^8 product_i D_i
                       = C M^8 V².                    (7)
```

Under G=0 the standard coefficient vector has canonical law PZ, where P is an orthogonal projection and Z is standard Gaussian. It is centered, even though Cov(G) degenerates as sites approach. In particular `||PZ||<=||Z||`, and the fixed finite expansion gives uniform conditional moments of M of every finite order. Equations (4) and (7) prove:

**Restricted collinear Rice lemma.** For the fixed finite Gaussian law, all sufficiently short collinear four-site configurations satisfy

```text
p_G(0) E[product_i |det H_i| * B | G=0] <= C             (8)
```

for every Borel mark 0<=B<=1. This includes typed or actual-selected four-endpoint marks whenever defined; it does not assert any selector identity.

In the two-pair regime, (7) has the more informative form

```text
product_i |det H_i| <= C M^8 r_1² r_2² eta^8.
```

The individual fold-pair determinant factors are r_i²; the simultaneous four-pin geometry contributes the extra eta^8 that cancels the additional eta^-8 gradient-density singularity. Dropping that vanishing by bounding determinants merely by field norms would create a spurious divergent bound.

## 5. Positive cubic marks force a Gaussian large deviation, already at finite radii

There is a stronger deterministic constraint. The same Rolle remainder gives, for s between the first pair's endpoints,

```text
|F_u(z+s u)|
 <= ||F_(u^5)||_infinity
       (s-x_1)(x_2-s)(x_3-s)(x_4-s)/24.
```

Each cross-pair factor is at most `(1+rho)eta`. Integrate along the first pair and use `integral_0^r s(r-s) ds=r³/6`. Repeat for the second pair:

```text
|ell_i| <= ((1+rho)² eta² /144)
                     ||F_(u^5)||_infinity r_i³,

|k_i| <= ((1+rho)² eta² /144)
                     ||F_(u^5)||_infinity.             (9)
```

This bound does not need the maximum/saddle type or an H0 pairing. It only uses the four longitudinal gradient zeros. Reversing one ordered pair changes the sign of its gap, not the absolute bound.

If B includes either `k_i>=k_->0`, then (9) forces

```text
||F||C5 >= c k_- eta^-2.                                (10)
```

The canonical zero-gradient conditional coefficient law PZ has a uniform Gaussian norm tail. Therefore

```text
E[M^8 1{||F||C5 >= c k_- eta^-2} | G=0]
   <= C exp(-c' k_-² eta^-4)
```

for eta sufficiently small relative to fixed k_-. Polynomial Gaussian-tail prefactors are absorbed by reducing c'. Combined with the exact cancellation in §4:

**Compact-positive-mark collinear bound.** Under the same fixed-law assumptions,

```text
p_G(0) E[product_i |det H_i|
          *1{k_1>=k_-,k_2>=k_-}*B | G=0]
 <= C exp(-c k_-²/eta^4).                               (11)
```

One can add `0<ell_i<=Tt`, upper mark bounds, Hessian type and actual selected-bar marks to the left. None worsens the upper bound. Equation (11) is stronger than a power bound on this geometric stratum and explains why keeping k away from zero is not an innocuous change of quantifiers near a double contact.

It does **not** cover k_i of order eta² or smaller: those are precisely the marks that can coexist with a field of ordinary C5 size as the centers collide. Any all-mark theorem must estimate that boundary instead of applying (11) with a disappearing lower bound and keeping its constants unchanged.

## 6. Exact limiting-contact covariance action and its eta^-4 coercivity

Now take two exact limiting contacts at z and z+eta u, with common longitudinal direction, using the source six-row vector at each center:

```text
U_i=(F,F_u,F_uu,F_uuu,F_v,F_uv)(z_i),
v_i=(b_i,0,0,12k_i,0,0).                               (12)
```

The first contact direction is u. Reversing a contact orientation changes the sign of its longitudinal third-derivative target in this common frame; the one-endpoint bound below uses its absolute value.

Let a(s)=F_u(z+s u). Define the following **linear functional of the contact observation rows**:

```text
L_1 = a''(0)
      -6[a(eta)-a(0)]/eta²
      +[4a'(0)+2a'(eta)]/eta.

L_1 = integral_0^eta
         s(eta-s)²/eta² * a^(4)(s) ds.                  (13)
```

The identity follows by writing the Taylor formulas with integral remainder for a(eta) and a'(eta) through their respective cubic/quadratic terms and subtracting. The kernel is nonnegative and has integral eta²/12. At a pair of contact targets, the pinned a and a' values vanish, so

```text
L_1(v)=12k_1.
```

By stationarity and Cauchy–Schwarz, with the actual fifth-derivative variance

```text
m_10(u)=Var(F_(u^5)(z))
      =sum_n q_n (omega n.u)^10,
```

we have

```text
Var(L_1) <= m_10(u) eta^4 /144.                         (14)
```

For nonsingular joint contact covariance Sigma, the Gaussian quadratic action satisfies

```text
v^T Sigma^-1 v
 >= L_1(v)²/Var(L_1)
 >= 20736 k_1² /[m_10(u) eta^4].                        (15)
```

The reflected formula at the second endpoint gives the same bound with |k_2|. Thus a compact positive cubic mark costs an `eta^-4` Gaussian action, regardless of birth marks. The Gaussian density's exponential is bounded by `exp[-10368 max(k_1²,k_2²)/(m_10(u)eta^4)]`. No conditional density of a pinned coordinate is declared positive.

For same-oriented positive contacts, summing the two reflected functionals gives the equivalent identity

```text
F_uuu(0)+F_uuu(eta)
 -2[F_uu(eta)-F_uu(0)]/eta
 = integral_0^eta s(eta-s)/eta * F_(u^5)(s) ds.
```

On the target this is 12(k_1+k_2), and it also yields the deterministic necessary condition `||F_(u^5)||infinity >=72(k_1+k_2)/eta²`. The individual-endpoint formula is safer when directions are reversed.

These formulas explain more than an unspecified loss of covariance floor: the degenerating Gaussian coordinate itself is identified, its variance is bounded explicitly, and the prescribed positive-contact target lies a distance of order eta^-2 in its standard-deviation units.

## 7. Why the twelve-dimensional density is unavailable at K2/K3

Restrict the actual field to a coordinate line y=y_0. The scalar function F(x,y_0) is a real trigonometric polynomial of degree K, dimension 2K+1. Its value and x-derivative at four sites produce eight longitudinal rows. The transverse derivative F_y(x,y_0) is another degree-K trigonometric polynomial, producing four value rows.

These two data channels can be varied independently in the full-square coefficient support: use x-only modes for the first and `sin(omega(y-y_0))` times an x trigonometric polynomial for the second. Thus for four distinct sufficiently close sites, the full value/gradient observation rank is

```text
rank O = min(8,2K+1)+min(4,2K+1)
       = 9 (K2), 11 (K3), 12 (K>=4).                   (16)
```

The same count holds for the two limiting contacts: longitudinal jets of orders 0–3 at two centers impose eight Hermite rows, while F_y,F_xy at those centers give four transverse Hermite rows. Standard polynomial/trigonometric root counting proves the ranks. In particular the twelve-dimensional Lebesgue density simply does not exist on this stratum at K2/K3. This is not a numerical small-eigenvalue issue.

There is an even sharper K2 support obstruction. Suppose two distinct aligned exact contacts had

```text
F_x=F_xx=0 at both centers,  F_xxx=12k_i !=0.
```

Then a=F_x restricted to that coordinate circle is a degree-two trigonometric polynomial with zero mean, and has two double zeros. A nonzero degree-two trigonometric polynomial has at most four zeros counting multiplicities. The two double zeros exhaust them, so its sign cannot change on the circle; zero mean then forces a identically zero. Consequently F_xxx is zero, contradicting either positive cubic target.

Equivalently, in a sufficiently short coordinate-aligned four-site gradient-pinned configuration at K2, the four longitudinal values of F_x are independent observations on the four-dimensional zero-mean derivative space. Their zero target forces F_x identically zero on that line. Hence F_xx=0 at all four pins and none can be a nondegenerate maximum. The typed four-endpoint kernel is zero there. This is compatible with the nonsingular **eight-gradient** covariance proved in §3; full gradient observation rank and conditional maximum admissibility are different questions.

The K3 height covariance remains singular but its positive two-contact target need not be excluded by this K2 zero-mean argument. A lower-dimensional affine-support treatment would have to be stated and checked; a twelve-dimensional Gaussian density cannot be used.

## 8. What a stronger finite carrier gives for the height-density route

For K>=4, the source-style frequency polynomial argument supplies positive covariance for the twelve limiting jet rows

```text
(F, F_u, ..., F_(u^7),
 F_v, F_uv, F_uuv, F_uuuv),                              (17)
```

with factorial factors as appropriate. The total degree is at most seven in each original frequency coordinate, below the nine-point grid available at K4. The framewise covariance floor is positive. This is a sufficient carrier condition for the following collinear normalization; it is not claimed necessary.

For four values and four longitudinal first derivatives, the confluent Hermite Vandermonde determinant is V^4. The four transverse gradient values have ordinary Vandermonde determinant V. Therefore a twelve-row divided-difference transformation has absolute determinant V^-5:

```text
det Cov(O)=V^10 det Cov(normalized O),
p_O(target)=V^-5 * bounded_prefactor
                       * exp(-Gaussian_action/2).       (18)
```

On a one-scale collision `x_i=epsilon a_i` with fixed distinct a_i this gives raw density prefactor `epsilon^-30`. In the two-pair regime it is `r_1^-5 r_2^-5 eta^-20`.

For the two limiting contact vectors (12), confluent multiplicities four and four in the longitudinal channel contribute eta^16; multiplicities two and two in the transverse channel contribute eta^4. Hence

```text
det Cov(U_1,U_2)=Theta(eta^40),
p_(U_1,U_2)(v)=O(eta^-20) exp(-action/2).                (19)
```

The largest normalized derivative order is seven. The divided-difference transformation has entries bounded by `C eta^-7`, so the normalized target is bounded by `C eta^-7 |v|`. Regression with its uniformly nonsingular normalized covariance bounds conditional coefficient means by `C eta^-7 (1+|v|)`; conditional centered covariance is still a projection. On compact b_i,k_i marks this gives, for A_i=F_vv at the two centers,

```text
E[A_1² A_2² | U=v] <= C eta^-28.
```

The literal single-contact planar weight from the source is

```text
W_i/r_i² -> 36 k_i² A_i² 1{A_i<0}.
```

For any bounded selection factors, the limiting joint compact-positive-mark contact integrand is consequently bounded by

```text
C eta^-48 exp(-c k_-²/eta^4),                           (20)
```

and is smaller than every power of eta. The exponent 48 is a sufficient crude polynomial bound, not a sharp claim; the exponential suppression is the useful fact. No equality of actual-bar selection or new process law is inferred.

Equation (20) concerns exact contact limits. Uniform passage from finite radii to those limits as eta shrinks is a separate quantifier problem. The gradient-pinned finite-radius bound (11) avoids that exchange on the restricted collinear geometry.

## 9. Diagnostic computations, not proof

Files:

- `coalescing_diagnostics.py`
- `COALESCING_DIAGNOSTICS.json`

The script uses Python standard-library Fraction for exact polynomial evaluation-matrix ranks and Decimal at 95 significant digits for covariance determinants and actions. Sin/cos use a truncated Taylor summation without certified intervals. No field is sampled, no seeds are used, and there is no numerical H0 or Poisson test.

The diagnostic law has uniform positive full-square spectral weights and omega=1 (L=2pi). It is a legitimate fixed finite law of the stated class, but not an identification with the L24 catalogue sampler. The exact rational rank checks give:

| K | Eight gradient-jet polynomial rank | Twelve height/contact limiting-jet polynomial rank |
| ---: | ---: | ---: |
| 2 | 8 | 9 |
| 3 | 8 | 11 |
| 4 | 8 | 12 |

For four sites 0,epsilon,2epsilon,3epsilon and K4 contact centers 0,epsilon, the decimal values are:

| epsilon | det Cov(G_K2)/V^4 | det Cov(O_K4)/V^10 | det Cov(U_K4)/eta^40 | eta^4 * contact action |
| ---: | ---: | ---: | ---: | ---: |
| .2 | 1.27595559 | 62.91004725 | 2591309.75579909 | 1135.96506151 |
| .1 | 1.54859762 | 119.72582867 | 2945034.77606788 | 1147.42709118 |
| .05 | 1.62517191 | 140.51761514 | 3040737.50095657 | 1150.31525609 |
| .025 | 1.64488465 | 146.25015545 | 3065143.34549533 | 1151.03871914 |

The action target is `k_1=k_2=1, b_1=0, b_2=eta³/5`, the height difference compatible with the local quartic derivative `a(s)=6 eta^-2 s²(s-eta)²`. This choice avoids gratuitously forcing a higher-order inconsistent birth difference. These observations are consistent with (1), (18), (19) and eta^-4 coercivity; they do not certify constants, asymptotic windows, orientation-uniform global bounds or theorem closure.

The first diagnostic invocation failed because Decimal treats `0**0` as InvalidOperation; monomial evaluation was corrected to form the integer power before Decimal conversion. The successful rerun and its outputs are retained locally. That arithmetic repair supplies no scientific evidence.

## 10. Scope that remains genuinely open

The restricted lemma removes a naive covariance-versus-weight divergence in one geometry and demonstrates a support obstruction to another route. It does not settle the physical full-dimensional integral:

- Four generic sites need not lie on a common straight line. Transverse offsets and independently oriented pair axes change which divided differences remain bounded. A uniform tubular/multiscale extension is needed before the collinear cancellation can control spatial measure.
- The regime k_i=O(eta²) is not suppressed by the compact-positive-mark bound. Its Jacobian, conditional transverse Hessian law, selection and mark integration must be calculated.
- Coalescing patterns where radii and center separation are comparable, endpoint permutations, shared endpoints in candidate factorial counts, and possible higher degeneracies must be partitioned explicitly. Actual distinct H0 bars have four distinct endpoints, but that fact does not furnish all covariance estimates.
- The K2/K3 singular twelve-row strata must be treated with a supported affine incidence formula or avoided by a strengthened sufficient carrier; assigning them a positive Lebesgue density is not an option.
- A local/fixed-law bound does not give uniform increasing-volume constants, mixing, a limiting coefficient as L varies, tightness or a spatial Poisson limit.

A concrete next calculation is therefore to perturb the collinear configuration transversely with declared ratios to r_i and eta, derive a joint observation normalization with a uniform covariance floor, and determine the smallest transverse scale at which the Vandermonde cancellation survives. That is a finite extension of the demonstrated mechanism. It preserves the existing process owner's unproved coalescing-center and expanding-domain scope.
