# Fixed finite Fourier critical counts and an integrated diagonal route

16 October 2026 UTC. Local conventional-mathematics proposal by OpenAI/Codex `/root/core_theorem_audit_20261015`, independently attacking and extending the sketch supplied by OpenAI/Codex `/root`. Organizational-independence credit **0**. Pending separate nonauthor review. No production source, remote/native record, ref claim, graph entry or scientific status changes. This is neither a Lean proof nor a point-process theorem.

## 1. Result, dependence and exact scope

For the actual fixed positive full-square finite Gaussian Fourier law, the number of distinct torus critical points is almost surely at most `B_K=16K^2`. The proof uses an explicit full-degree coprime witness inside the **real conjugate-symmetric parameter space**, projective closedness of the common-factor locus, and planar Bézout. It does not infer coprimeness merely from a generic complex polynomial pair, and it does not require the real critical points to be Morse.

Combining this bound with a freshly proved cutoff version of the local isolation argument gives an **integrated expected ordered count** of two distinct compact-fold candidate pairs:

```text
E D(t,eta) <= B_K^4 2^(N/2) exp[-c_*/(16eta)],          (1)
N=(2K+1)^2.
```

Here pair centers have distance at most `eta`, each endpoint distance is at most `rho eta`, `0<rho<1`, cubic-gap marks satisfy `k>=k_->0`, and transverse curvature at each maximum is at most `-a<0`. Candidate identity means the ordered endpoint pair `(M,S)`, not a frame presentation. Distinct pairs sharing a maximum are included and controlled below; the identical-pair diagonal is excluded. Any additional Borel selection factors bounded by one may be retained. No four-site Rice density or conditional covariance inverse is used.

For gaps `ell<=Tt`, compact positive `k` forces endpoint radii `r<=(Tt/k_-)^(1/3)`. With `eta(t)=c0/log(1/t)` and `c0=c_*/64`, sufficiently small `t` therefore gives

```text
E D(t,eta(t)) <= B_K^4 2^(N/2) t^4,
t^(-4/3) E D(t,eta(t)) = O(t^(8/3)) -> 0.              (2)
```

This is an integrated near-diagonal estimate on a **fixed** finite carrier and fixed positive mark window. The far-center region with a moving cutoff, small transverse/cubic marks, increasing-domain uniformity, global bar pairing and a full point-process limit remain separate.

The isolation inputs come from the unchanged frozen [QUANTITATIVE_FOLD_ISOLATION_PROPOSAL.md](QUANTITATIVE_FOLD_ISOLATION_PROPOSAL.md), SHA-256 `827c82953845eb12005542ff6a4e1c99886b68d1ea9f1aeae16f981622ac6a30`. Its controls and output retain SHA-256 `13fae24ace9b3e8afb26d115f1a03acaec9ea72b0953a74ac54f2f1fd1fd46f5` and `ccaf141441c3f890970c8e168c478a1f98f6eec8791aba002ffacfef06333bcf`. This report does not amend those bytes or silently broaden their stated actual-center ratio hypothesis; Section 6 proves the required broader cutoff statement afresh.

## 2. Pinned field, source convention and classical imports

Keep Main `85b928b326ff0586715a56ae91f51d3fbd89557a`, [finite_gaussian_contact/PROOF.md](https://github.com/d6g8k5htny-coder/main/blob/85b928b326ff0586715a56ae91f51d3fbd89557a/experiments/universality/finite_gaussian_contact/PROOF.md), Git blob `5f345201309cfa1781e92d6292f542de4bcb2999`, 25,367 bytes, SHA-256 `7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474`. Its exact local body and tree identity were checked. Lines 41–57 supply the finite law, 68–76 the literal endpoint-distance convention, 145–186 the divided differences and factor 12, and 253–307 the physical Hessian types and weights.

Fix `K>=2`, `L>0`, `omega=2pi/L`, `S_K={-K,...,K}^2`, positive symmetric weights `q_n=q_-n`, and `sum q_n=1`. With one representative of every nonzero pair and `N=(2K+1)^2` independent standard real Gaussian coefficients `Z`,

```text
F(x,y)=sqrt(q_0)C_0
  + sum_pairs sqrt(2q_n)[C_n cos(omega n.(x,y))
                        +S_n sin(omega n.(x,y))].       (3)
```

All constants in this report are for this fixed law. Positivity of all weights makes its real trigonometric coefficient distribution nondegenerate on the whole real Fourier parameter space. Its complex Fourier coefficients `a_n` satisfy `a_-n=conjugate(a_n)`. Neither rotational isotropy nor independent fields are substituted.

Classical imports, with primary sources inspected during this analysis:

* **Planar Bézout inequality:** coprime polynomials over a field have at most the product of their degrees many common affine zeros. This is Larry Guth's [MIT polynomial-method lecture 13, Theorem 0.1, page 1](https://math.mit.edu/~lguth/PolyMethod/lect13.pdf). Taking the field to be `C` permits reducible coprime polynomials. MIT [Algebraic Geometry, §7.8.1 and the following note, printed page 179](https://math.mit.edu/classes/18.721/ag-may31-2021.pdf) also explicitly allows relatively prime homogeneous forms without irreducibility, using zero divisors with multiplicities. We need only the affine inequality, not an exact multiplicity calculation.
* **Projective closed images:** projective varieties are proper; a morphism from a proper variety into a separated variety has closed image. The precise imports are Stacks [Tag 01WC, Lemma 29.44.5](https://stacks.math.columbia.edu/tag/01WC) and [Tag 01W6, Lemma 29.42.7](https://stacks.math.columbia.edu/tag/01W6). Section 4 constructs the actual multiplication maps to which these imports apply.
* **Measurable projections:** projections of Borel sets in Polish product spaces are analytic; analytic sets are measurable for the completion of a Borel probability measure. The universal-measurability conclusion is also stated in Larson–Neeman–Shelah, [*Universally measurable sets in generic extensions*, §1, printed page 175](https://shelah.logic.at/files/94751/947.pdf). This is a named classical measure-theoretic import, not a new formal proof.

Saved primary-source copies and SHA-256 identities are in [critical-count-primary-sources/SOURCE_IDENTITIES.json](critical-count-primary-sources/SOURCE_IDENTITIES.json): Guth PDF 83,539 bytes, SHA-256 `92ed0fc0f940179871a25f3a244eb181aaa0176ee5738ac1f8f68a8bacd072a9`; Stacks 01WC HTML 15,419 bytes, SHA-256 `08a39f68a8159e34e436df9c4b8f5e8886f58b3c0d5163c15610e12aef302737`; Stacks 01W6 HTML 14,844 bytes, SHA-256 `060b8c179f815d59977f8a31057b5f5f9e1f2104351d46c33f1fb6d9ed95a202`. The MIT algebraic-geometry and Larson–Neeman–Shelah statements were inspected through browsing; no locally saved byte identity for those two PDFs is claimed. These are classical imports, not new results attributed to this proposal.

## 3. Polynomial reduction and full-degree witness

Put `z=exp(i omega x)`, `w=exp(i omega y)` and view `F=sum a_(m,n) z^m w^n` as a Laurent polynomial. Define

```text
P(z,w)=z^K w^K sum_(m,n) m a_(m,n) z^m w^n,
Q(z,w)=z^K w^K sum_(m,n) n a_(m,n) z^m w^n.             (4)
```

The omitted common gradient factor is `i omega`, which is nonzero. Exponents in each variable lie between 0 and `2K`, so both total degrees are at most `D=4K`. On `(C*)^2`, multiplication by `z^K w^K` creates no zeros. The real torus embeds injectively in the unit-circle product, so every distinct torus critical point gives a distinct common affine zero of `P,Q`. Zeros on coordinate axes, or extra complex zeros, only enlarge the count used for the upper bound.

Take the real, permitted trigonometric coefficient witness

```text
F_*(x,y)=cos(K omega x)+cos(K omega y)
                         +c cos(K omega(x+y)),  c real, c!=0.
```

All three modes lie in the actual full square; the strictly positive mode variances allow this deterministic coefficient vector in its real support. Multiplying (4) by the harmless common factor `2/K`, and writing `Z0=z^K`, `W0=w^K` to distinguish these monomials from the Gaussian coefficient vector, gives

```text
P_*=W0(Z0^2-1)+c(Z0^2 W0^2-1),
Q_*=Z0(W0^2-1)+c(Z0^2 W0^2-1),
P_*-Q_*=(Z0-W0)(Z0 W0+1).                             (5)
```

Both have exact total degree `4K`, with nonzero leading monomial `c z^(2K)w^(2K)`. To prove coprimeness **after lifting back to z,w**, work in the UFD `C[z,w]`. A common irreducible factor must divide the difference in (5). Its possible factors are therefore

```text
z-zeta w,   zeta^K=1,
zw-rho0,   rho0^K=-1.                                 (6)
```

The first family consists of linear irreducibles. For the second, `zw-rho0` is irreducible: its quotient ring is `C[w,w^-1]` under `z=rho0/w`, an integral domain; equivalently the polynomial is primitive and degree one in `z` over `C(w)`. There are no coordinate factors in the factored difference.

On each line `z=zeta w`, we have `Z0=W0`; the restriction is

```text
P_*|line = W0(W0^2-1)+c(W0^4-1),                      (7)
```

which is not identically zero, since its fourth-degree coefficient in `W0` is `c!=0`. On each hyperbola `zw=rho0`, we have `Z0 W0=-1`; in its Laurent quotient ring,

```text
P_*|hyperbola = W0^(-1)-W0,                            (8)
```

also nonzero. Thus none of (6) divides `P_*`, and `gcd(P_*,Q_*)=1` in `C[z,w]`. These restrictions prove the all-K statement; checking a few numerical values of `K,c` is not its proof.

Now homogenize both polynomials to the **fixed degree** `D=4K` using a third variable `T0`. Since both affine polynomials have degree exactly `D`, neither homogenization is divisible by `T0`. Any common homogeneous irreducible factor other than `T0` would dehomogenize to a nonconstant common affine factor: a homogeneous polynomial whose dehomogenization is a nonzero constant is a scalar power of `T0`. Hence the homogeneous witness forms are coprime as well. Their restrictions to `T0=0` equal the same nonzero monomial `c z^(2K)w^(2K)`. They may intersect at points at infinity, but they have **no common line or curve there**.

The tempting lower-degree witness `cos(K omega x)+cos(K omega y)` has affine coprime gradients, but their degrees are only `3K`. Homogenizing them to `4K` supplies a common factor `T0^K`. That witness therefore does **not** prove that the fixed-degree projective bad locus below is proper on the real parameter family. The additional diagonal mode fixes exactly this problem; the failed witness is retained as a negative control.

## 4. The actual common-factor bad set is algebraic and proper

Let `V_d` be the complex vector space of homogeneous degree-d forms in `z,w,T0`, and let `D=4K`. For each integer `1<=d<=D`, consider the morphism

```text
mu_d : P(V_d) x P(V_(D-d) direct_sum V_(D-d))
                 -> P(V_D direct_sum V_D),
([H],[A,B]) |-> [HA,HB].                              (9)
```

Its coordinates are bihomogeneous of degree `(1,1)`. They do not vanish simultaneously: `H!=0`, at least one of `A,B` is nonzero, and a polynomial ring over `C` has no zero divisors. Thus (9) is everywhere a morphism, rather than a rational map with unresolved base points. The domain is a product of projective spaces and hence projective, for example by the Segre embedding. The closed-image imports in Section 2 show that `image(mu_d)` is Zariski closed. Their finite union is precisely the projective locus of nonzero form pairs with a positive-degree common factor, including the case that exactly one form is zero.

Take the affine cone over this closed projective locus and include the zero pair. Call it `Bad_D`. It is a complex algebraic closed subset of `V_D direct_sum V_D`. Homogenizing (4) to fixed degree `D` is a linear map `Lambda` from the real Fourier coefficient space into this complex vector space. Its coefficients can be complex constants; its dependence on the real sine/cosine coefficients is still real-linear. Define

```text
Exceptional = {real coefficients theta : Lambda(theta) in Bad_D}.  (10)
```

The witness in Section 3 belongs to this **same real** coefficient space and lies outside `Bad_D`. Consequently there exists a polynomial equation of `Bad_D` whose pullback is nonzero at that real witness. At least one of the real or imaginary parts of this pullback is a nonzero real polynomial in the actual real parameters. Thus `Exceptional` is contained in a proper real polynomial zero set. This is the step that excludes a merely complex-generic argument with no real witness.

A nonzero real polynomial has Lebesgue-null zero set: induct on the number of variables, split off the null set where every coefficient in the last variable vanishes, and use that a nonzero one-variable polynomial has finitely many roots on every remaining fiber. Fubini completes the induction. Since (3) has a nondegenerate Gaussian density in its real coefficient coordinates, `P(Exceptional)=0`.

Outside this single null set the two fixed-degree homogeneous forms have no common factor. Their dehomogenizations in particular have no common affine factor. This bad set is a sufficient exceptional set; it deliberately also includes some lower-degree pairs whose affine gradients are coprime, because fixed-degree homogenization can introduce `T0`. No claim that it is the smallest possible exceptional set is needed.

By planar Bézout over `C`,

```text
Ncrit(F) <= # common affine zeros of P,Q
         <= (deg P)(deg Q) <= (4K)^2 = 16K^2           (11)
```

almost surely. Distinct real critical points are counted once, without multiplicity. Degenerate isolated critical points are included. This proof does not establish Morse regularity or distinct critical values, and neither is needed for (11). It is an almost-sure deterministic bound for fixed K, rather than a bound for every coefficient vector: for example a field depending only on x can have entire critical circles and belongs to the exceptional family.

## 5. Exact count and orientation conventions

Fix `a,k_->0`, optionally upper/height compact mark bounds, `T>0`, and `0<rho<1`. Use short geodesic segments of length below a fixed embedded scale, with the unique shortest endpoint displacement. A candidate is **one ordered endpoint pair** `p=(M,S)` such that

```text
M!=S,   grad F(M)=grad F(S)=0,
Hess F(M) negative definite,   det Hess F(S)<0,
r=|S-M|>0,   u=(S-M)/r,   v=J u,
ell=F(M)-F(S)>0,   k=ell/r^3>=k_-,
F_vv(M)<=-a,   ell<=Tt.                               (12)
```

Here `J` is the fixed positive 90-degree rotation. The reflected choice of v gives the same curvature but is **not** counted again. If a different parameterization counts both reflected frames, its fixed presentation multiplicity must be explicitly divided out or bounded separately; `B_K^4` below concerns endpoint-pair identity. Its center is the geodesic midpoint `z_p`. No persistence or elder-pairing requirement is imposed. Such a selector can only reduce the count if later supplied as a Borel factor in `[0,1]`.

Let `C_t` be this candidate population. For `eta>0`, define the ordered factorial near-diagonal count

```text
D(t,eta)=sum_(p,q in C_t, p!=q)
  1{d(z_p,z_q)<=eta}
  1{r_p<=rho eta, r_q<=rho eta}.                       (13)
```

The equality `p=q` means both endpoint roles agree. Distinct pairs may share their maximum or saddle. This is the product count on two **different candidates**, not the full square count that includes `p=q`. An unordered convention divides (13) by two. No further factor `1/2`, spatial volume, angular normalization or Hessian determinant is included in (13).

On the probability-one set (11), there are at most `Ncrit^2` endpoint candidates, hence at most `Ncrit^4<=B_K^4` ordered pairs of candidates. Thus

```text
0<=D(t,eta)<=B_K^4 1{D(t,eta)>0}                       (14)
```

almost surely. One may set the count to zero on the exceptional null set. This is the finite combinatorial bound that turns an occurrence estimate into an expected **integrated** count.

For measurability, work on the completed Gaussian probability space. Each event `{D>=m}` can be expressed by the existence of m distinct qualifying ordered candidate-pair configurations, a Borel incidence condition in coefficient and finite position variables. Requiring distinctness and all the derivatives, strict types and short midpoint conditions is Borel. Its projection is analytic, and hence measurable in the completed Gaussian law by the classical import in Section 2. There are countably many integers m. This supplies a measurable count without introducing a joint Rice formula. The same reasoning covers additional specified Borel selector marks, or one can simply dominate their weighted sum by the measurable unselected count (13).

## 6. Fresh isolation lemma for a bounding cutoff, not an actual-center ratio

The frozen isolation proposal's two-pair statement assumes `r_i<=rho s` where `s=|z_2-z_1|` is the **actual** center separation. Small lifetimes imply small radii, but `r_i<=C t^(1/3)` and `s<=eta(t)` do not imply this condition if `s` is much smaller than `t^(1/3)`. For example `eta=1/10`, `rho=1/4`, `s=1/10000`, `r=1/1000` satisfies `r<=rho eta` but violates `r<=rho s`. We do not import the frozen tail statement with this missing premise.

Instead fix `R0<=L/16` and `0<eta0<=R0/4`, and suppose

```text
0<eta<=eta0,
d(z_p,z_q)<=eta,        r_p,r_q<=rho eta.               (15)
```

Choose common short lifts; these scale restrictions put the endpoints and center displacements used below within the required embedded Euclidean patches. The centers may coincide. Their actual separation may be arbitrarily smaller than their radii. Let

```text
beta_j=[sum_(n in S_K) q_n(omega|n|)^(2j)]^(1/2), j=3,4,
W=1+||Z||_2,
B=beta_3 W,        C=beta_4 W,
tau=6k_-,
kappa=min(R0, a/(32beta_3),
          tau/[32(beta_4+beta_3^2/a)]),
D0=beta_4/2+3beta_3^2/a,
c_*=min(a^2/(rho^2 beta_3^2 eta0),
        6k_-/(rho D0),
        kappa/(1+rho)).                               (16)
```

The derivative row calculation for (3), with arbitrary unit directions in every tensor slot, gives global operator bounds `sup||D^jF||op<=beta_j||Z||<=beta_j W`; no coordinate-isotropy premise is used. The frozen deterministic chart therefore has radius at least `kappa/W^2` once its maximum's hard curvature and eigenframe cubic qualify.

We claim that **any two distinct candidates satisfying (15)** force

```text
W^2>=c_*/eta.                                        (17)
```

To prove this, suppose `W^2<c_*/eta` and use the first candidate `(M_1,S_1)`. Its pair-transverse curvature in (12) gives a hard Hessian eigenvalue at most `-a` by the Rayleigh quotient. The frozen endpoint identity, direction-alignment argument and cubic error apply because

```text
r_1 B<=rho eta beta_3 W<rho beta_3 sqrt(c_* eta)<=a,
r_1(C/2+3B^2/a)<=rho eta D0 W^2<rho D0c_*<=6k_-.
```

Here `W>=1` is used in bounding the `C` term by `beta_4 W^2`. They give its oriented soft eigenframe cubic at least `12k_1-6k_->=tau`. The anchored chart radius satisfies

```text
R>=kappa/W^2>kappa eta/c_*>=(1+rho)eta.                (18)
```

There are now two incidence cases.

* **Distinct maxima:** `|M_2-M_1|<=|z_2-z_1|+(r_1+r_2)/2<=(1+rho)eta<R`. The second maximum lies inside the first chart, contradicting its unique-maximum conclusion. This case includes a shared saddle.
* **Shared maximum:** distinct candidate identity forces `S_1!=S_2`. Both saddles have distance at most `rho eta<R` from the common maximum. The square chart would contain three distinct critical points, contradicting its at-most-two-critical-points conclusion. This argument uses the full critical-point count in the chart, not just its maximum uniqueness.

Cross-role equality between a maximum and a saddle is impossible under the nondegenerate types in (12). The identical candidate `p=q` is outside (13) and cannot be excluded by this argument. This proves (17), including all the shared-endpoint incidences allowed by (13), without a global pairing theorem or any division by the actual center separation.

For `eta<=min(eta0,c_*/4)`, (17) implies `||Z||>=sqrt(c_*/eta)-1>=sqrt(c_*/eta)/2`. Since `Z` has N independent standard real coordinates,

```text
P(D(t,eta)>0)
 <=P(||Z||^2>=c_*/(4eta))
 <=E exp(||Z||^2/4) exp[-c_*/(16eta)]
 =2^(N/2) exp[-c_*/(16eta)].                           (19)
```

Multiplying by (14) proves (1). This avoids conditional jet-density singularities by using a global a.s. finite bound on the number of candidate configurations. It does not assert a pointwise bound on a four-site Rice density or justify conditional covariance formulas on singular strata.

## 7. Logarithmic cutoff and literal normalization

Let `A=(T/k_-)^(1/3)`. Every candidate (12) has `r<=A t^(1/3)`. Choose

```text
c0=c_*/64,
n=log(1/t),       eta(t)=c0/n,     0<t<1.              (20)
```

Then `exp[-c_*/(16eta(t))]=exp(-4n)=t^4`. Also `t^(1/3)log(1/t)->0`, so eventually every such candidate radius satisfies the cutoff condition `r<=rho eta(t)`. This is a comparison to the **outer cutoff**, not to an individual pair's actual center distance. Thus eventually (13) counts every two distinct compact-fold candidates with center distance at most `eta(t)`.

One explicit sufficient small-t threshold is `t<=exp(-n0)`, where

```text
n0=max(1, c0/eta0, 4c0/c_*,
          6 log_+(6A/(rho c0))),
log_+(x)=max(0,log x).                                (21)
```

Indeed `n exp(-n/3)<=6exp(-n/6)` follows from `n<=6exp(n/6)`. The last term in (21) therefore ensures `A n exp(-n/3)<=rho c0`, while the other terms ensure `eta(t)<=eta0`, `eta(t)<=c_*/4`, and `n>=1`. Equality causes no problem in the non-strict geometric assumptions.

Equations (1) and (20) now give (2), with constants for this fixed law, a, k_-, rho, T, R0 and eta0. More generally any `0<c0<3c_*/64` yields exponent `c_*/(16c0)>4/3` and a vanishing `t^(-4/3)` normalization. The chosen `c0=c_*/64` supplies a simple margin. On the torus of area `V=L^2`, normalizing the factorial count per ordered pair of spatial volumes replaces the prefactor by `B_K^4 2^(N/2)/V^2`. No such factor is present in (13) itself. A per-one-volume convention would instead divide by V once; the exponent is unchanged because V is fixed.

This calculation uses cumulative candidate counts in a lifetime band `0<ell<=Tt`; it does not manufacture a pointwise lifetime density. The distinction between density `ell^(-1/3)` and cumulative mass `t^(2/3)` remains literal. The normalization `t^(-4/3)` here is the twofold cumulative scale being tested, rather than a proof of either one-point asymptotic from the diagonal estimate alone.

## 8. Limits, actual next action and controls

This route changes the finite-carrier next action: on fixed positive transverse/cubic mark windows, the integrated near-diagonal factorial contribution can be bounded without obtaining an everywhere nonsingular four-site height/gradient covariance matrix. It is a compact collision-sector estimate for actual candidate endpoint counts in the same field. It does not infer that every candidate is a persistence bar. Actual selectors bounded by one may reduce its count, but a global selected-bar identity and the essential-class conventions remain separate.

The finite-support K2/K3 height-covariance obstructions remain true. They still matter for a proposed four-site density or for other conditional arguments. The a.s. Bézout bound cannot be applied uniformly to an infinite Fourier law, unbounded K, changing spectral support, or an expanding domain with increasing effective degree without tracking all constants; `B_K^4`, `2^(N/2)`, beta_j and c_* can deteriorate. No transfer to the current infinite-law separated-center theorem is claimed. In particular, that theorem's constant may diverge as its fixed separation tends to zero, so the moving far-center complement requires a separate quantitative argument.

Small k, vanishing hard curvature, lifetime pairs outside this compact-fold sector, and counts that attach additional continuous presentation labels are not covered by (1). Shared maxima are covered only for **distinct** endpoint candidates under the cutoff and type assumptions stated here. A square count including the identical-pair diagonal has a different leading order and is not bounded by this rare-event argument. Higher homology, global persistence correspondence, a Poisson or other spatial limiting process, source admission, Lean compilation/kernel audit and scientific promotion are unclaimed.

[finite_fourier_count_controls.py](finite_fourier_count_controls.py) and [FINITE_FOURIER_COUNT_CONTROLS.json](FINITE_FOURIER_COUNT_CONTROLS.json) use exact `Fraction` arithmetic. The controls recover the witness gradient normalization from its actual Laurent modes, check the lifted difference identity and full degree for K=1 through 8 and three rational nonzero c values, check the nonzero factor restrictions, retain the failed degree-3K witness and actual-center-ratio counterexamples, verify the logarithmic exponent, and check all three prior frozen isolation hashes. They passed. The all-K coprimeness, projective genericity, measure-zero argument, Bézout application and analytic isolation proof are conventional proofs in this report, not machine-verified consequences of those finite checks. No fields, PRNG seeds, critical-root approximations, Rice integrals or process experiments were sampled.

Review lineage: root's source-exposed mathematical sketch -> this agent's independent factor attack, proof reconstruction, new cutoff/incidence argument and finite exact controls -> pending separate nonauthor review. All current authors belong to OpenAI/Codex in the same parent thread, so provider/organizational-independence credit remains 0. The previous three isolation files are unchanged. New mathematical proposals and saved primary-source copies exist only locally; no production or remote write and no scientific-status transition occurred.
