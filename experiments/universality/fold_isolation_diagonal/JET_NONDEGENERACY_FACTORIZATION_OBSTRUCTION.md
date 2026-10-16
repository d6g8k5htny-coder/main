# Uniform smooth moments do not supply spatial jet rank or factorization

16 October 2026 UTC. Local mathematical proposal by OpenAI/Codex `/root/core_theorem_audit_20261015`, following root's targeted obstruction request and independently constructing the examples and estimates below. Earlier isolation/norm sketches are joint root/agent work; this document is a new obstruction/factorization analysis, not a first review of them. Organizational-independence credit **0**. Pending separate nonauthor review. No production/native/remote edit, source claim, owner transfer, Lean execution or scientific-status transition.

## 1. Main findings

Normalized spectral mass and a uniform twelfth physical moment supply upper smoothness bounds. They supply neither lower jet variance nor decorrelation. There are two exact positive full-square counterfamilies:

1. A dominant DC or long-wavelength component makes the single-contact covariance floor tend to zero, despite normalized mass and uniformly bounded physical moments.
2. More sharply, a repeated fixed finite Gaussian spectrum plus an arbitrarily small positive full-square fill has a **uniformly positive seven-row one-contact covariance floor**, while its two-contact covariance floor tends to zero and correlations at macroscopic torus separations tend to one.

The second example also gives an actual candidate-process obstruction, not just a failed covariance estimate. On t_N=N^-3 and L_N=2pi N, the compact-fold candidate process tends to the empty process in probability. A deterministic sufficiently small positive fill can at the same time preserve a finite nonzero expected compact-fold count, using the fixed-base source's literal marked kernel and finite-N continuity. Thus a universal nonzero spatial Poisson conclusion cannot follow from uniform smoothness plus one-contact rank, even with this nonzero first-moment scale. No assertion about global persistence bars is transferred from this endpoint-candidate example.

A possible replacement is quantitative physical covariance control: a uniform single-contact floor, an explicit two-contact floor on the moving middle band, and decay of cross-contact covariance on the far band. Sections 6–8 prove what these hypotheses buy for the actual determinant-weighted candidate kernel, with a precise middle/far ledger. They are finite covariance/derivative estimates to establish for a source, rather than certificates that assume a point-process conclusion. Section 9 identifies a concrete Gaussian truncation family that could satisfy them; it retains the estimates and review work still needed before source adoption or process closure.

## 2. Authoritative consumed source and exact observable

The repository pin is Main `85b928b326ff0586715a56ae91f51d3fbd89557a`. The complete retained source identities were rechecked against its immutable tree:

| Source | Git blob | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| [finite_gaussian_contact/PROOF.md](https://github.com/d6g8k5htny-coder/main/blob/85b928b326ff0586715a56ae91f51d3fbd89557a/experiments/universality/finite_gaussian_contact/PROOF.md) | `5f345201309cfa1781e92d6292f542de4bcb2999` | 25367 | `7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474` |
| [finite_candidate_lifetime/PROOF.md](https://github.com/d6g8k5htny-coder/main/blob/85b928b326ff0586715a56ae91f51d3fbd89557a/experiments/universality/finite_candidate_lifetime/PROOF.md) | `a27a84150b3e70bfd794672eeeed6915425a1da4` | 29190 | `ed31f562f0ffaaf6d4bd2ceb1ae3d234798e983a55d55c6983b128fce27133f5` |
| [iid_short_bar_process/PROOF.md](https://github.com/d6g8k5htny-coder/main/blob/85b928b326ff0586715a56ae91f51d3fbd89557a/experiments/universality/iid_short_bar_process/PROOF.md) | `cd7ef8b8a364cb394be48208f55332658e9fff0b` | 30935 | `320bc988c468bea4938bb777a2d05952c6ab8e0a384f33725b50de847aeec1eb` |

The finite source proves contact rank for each fixed positive square spectrum. Its constants are expressly not uniform over vanishing weights, unbounded K or varying L. The process source's §§1–5 prove an iid-copy limit on one fixed torus for its actual **infinite** periodized Gaussian law. Its §6 gives a separated-center sufficiently-small-radius twofold O_eta0(t^(4/3)) bound at fixed L and eta0, using the actual K7/K5 interpolation blocks. It explicitly does not prove a spatial increasing-domain limit or a moving-separation uniform bound. None of those source qualifications is removed here.

The candidate endpoint convention remains

```text
M=x-r u/2, S=x+r u/2, grad F(M)=grad F(S)=0,
ell=F(M)-F(S)=k r^3>0,    r=|S-M|,
H_M<0,    det H_S<0,    v=J u,
A_M=F_vv(M).
```

r is the endpoint distance. u points from maximum to saddle and ranges once over S^1; reflected v is not another endpoint candidate. The six conditioning rows U_r and physical determinant weights are the finite source's literal ones. This analysis counts ordered endpoint candidates, without an elder/global selector. The short-bar density ell^(-1/3), cumulative t^(2/3) scale and twofold t^(4/3) scale are distinguished throughout.

## 3. Upper moments do not imply even one-contact rank uniformly

Write the ideal law with physical frequencies xi_n=(2pi/L)n and normalized symmetric positive weights q_n. Uniform assumption U is

```text
sum q_n=1, q_n=q_-n>0, L>=Lmin,
sum q_n |xi_n|^12 <= M12 uniformly.                    (U)
```

Take L_N=2pi N, K_N=2N, N>=4, and let the fill have uniform weights on {-2N,...,2N}^2. Put mass 1-eps_N at n=0 and eps_N times this fill, with eps_N->0. Every square weight is positive, total mass is one, and |xi_n|<=2sqrt(2), so M12<=262144. Nevertheless

```text
Var(F_u)<=eps_N (2sqrt(2))^2=8eps_N
```

for every unit direction u. The seven-contact covariance has a diagonal Rayleigh quotient at most 8eps_N. It is positive for each N by the source, but has no common positive floor. Removing the dominant DC mode alone does not fix the issue: dominant mass on the four axial modes (+/-1,0),(0,+/-1) has physical frequency 1/N, so gradient variance is O(N^-2+eps_N) and again collapses. These are lower-variance failures within U, not failures of individual smoothness.

## 4. One-contact rank can stay uniform while distant contacts cohere

Fix the base field G on T_(2pi)^2 with **uniform positive weights qbar_m=1/25 on {-2,...,2}^2** and the actual real Gaussian coefficients. Let H_N be the normalized uniform full-square ideal field on T_(2pi N)^2 with K_N=2N, independent of G. Define

```text
F_N(x)=sqrt(1-eps_N) G(x mod 2pi)+sqrt(eps_N) H_N(x),
q^(N)_n=(1-eps_N) sum_m qbar_m 1{n=N m}
                                      +eps_N/(4N+1)^2. (1)
```

For every 0<eps_N<1 this is exactly a centered ideal real Gaussian Fourier law with independent sine/cosine coordinates and the displayed positive full-square weights: overlapping frequencies add independent Gaussian coefficients, and different mode pairs remain independent. It is not an iid replacement of spatial cells. The same G is repeated in all N^2 cells. Its variance is one and

```text
sum_n q^(N)_n |n/N|^12 <= (2sqrt(2))^12=262144.         (2)
```

For a frame (u,v), let

```text
J(x,u)=(F,F_u,F_uu,F_uuu,F_v,F_uv,F_vv)(x),
Gamma_G(u)=Cov(J_G(x,u)).
```

The fixed-base source theorem gives `m_G=min_(frames) lambda_min Gamma_G>0`. The physical frequencies of the repeated G are exactly m, not m/N. Therefore

```text
Gamma_N(u)=(1-eps_N)Gamma_G(u)+eps_N Gamma_fill,N(u),
lambda_min Gamma_N(u)>=(1-eps_N)m_G.                   (3)
```

For eps_N<=1/2, all single-contact floors are at least m_G/2, uniformly in N and frame. This stronger counterexample does not obtain its failure by losing the one-fold cubic variance.

Now take a displacement h_N=(2pi j_N,0), with 1<=j_N<N, for example j_N=floor(N/2). Its torus distance is comparable to L_N. The dominant field and all its physical derivatives agree exactly at x and x+h_N. The uniform fill has the exact geometric sum

```text
Cov(H_N(x),H_N(x+h_N))=1/(4N+1),
R_N(h_N)=1-eps_N+eps_N/(4N+1).                        (4)
```

Indeed frequencies -2N,...,2N consist of four complete residue cycles modulo N and one additional zero residue. At a nontrivial Nth root the cycles sum to zero and the remaining term is one. The transverse coordinate contributes its full factor 4N+1.

The unit coefficient vector selecting `(F(x)-F(x+h_N))/sqrt(2)` in the fourteen-row two-contact vector has variance

```text
1-R_N(h_N)=eps_N(1-1/(4N+1))<eps_N.                   (5)
```

Thus the two-contact least eigenvalue tends to zero even at macroscopic separations, while (3) remains uniformly positive. Value correlation tends to one, so even zero-order far covariance cannot factorize. At any fixed N sufficiently large, the full-square source may still make the named distinct-center rows positive; positivity at each N is consistent with (5). More generally any shared-frame jet functional has difference variance O(eps_N) at these resonances. Uniform physical smooth moments and one-contact rank cannot preclude coherent spectral atoms or approximate spatial periods.

Equations (3)–(5) are already a rigorous obstruction to the proposed uniform covariance estimates. They alone would not prove a non-Poisson limit: an empty Poisson limit can result from vanishing intensity, and covariance conditions can be sufficient without being necessary. The next section gives an actual candidate-process statement with its own proof.

## 5. Actual empty-in-probability candidate process, and a first-moment variant

### Explicit small positive fill

Set eps_N=N^-12 and t_N=N^-3 in (1). The fixed physical bandwidth gives a direct global C4 bound, without consuming the new uniform local-norm proposal under review. For a finite normalized field with |xi_n|<=2sqrt(2), each coordinate derivative of order at most four is bounded by 64 times its standard coefficient-vector norm. There are 15 such multi-indices. Using the sum-of-suprema C4 norm,

```text
||H_N||C4<=960 ||Z_N||_2,
E||sqrt(eps_N)H_N||C4^2
 <=960^2 eps_N(4N+1)^2=O(N^-10).                     (6)
```

Also `1-sqrt(1-eps_N)<=eps_N`, and G has finite C4 second moment. Thus `||F_N-G_repeated||C4->0` in probability, in the physical norm on the whole growing torus.

Almost surely G is Morse and has distinct critical values by the fixed-base finite source. It has a finite critical set and a positive minimum absolute height difference Delta(G) between a maximum and a saddle; if one of these types were absent, the no-candidate conclusion would be immediate. There is also a positive C4 stability radius delta(G), **uniform over all repetitions**: choose the finite critical neighborhoods and a gradient floor on one base torus, then repeat them. More explicitly, choose convex lifted balls about each base point p so that `||H_p^(-1)(Hess G(x)-H_p)||<1/4` there. Choose the perturbation C2 radius so that the corresponding bound for `Hess f-Hess G` is below 1/4. Then `H_p^(-1)grad f` differs from the identity linear map by a map of Lipschitz constant below 1/2, and hence is injective on that ball. The implicit branch supplies the root there and its type persists; no additional oscillatory roots can appear in the same ball. A gradient floor on the remaining compact region excludes every other root. The minimum of these finitely many base choices works in every cell for every N, however rapidly the perturbation oscillates. Shrink delta(G) further to make every perturbed critical height differ from its base height by less than Delta(G)/4.

Consequently, if the global perturbation norm is below delta(G), every positive maximum–saddle candidate gap on T_(2pi N)^2 is at least Delta(G)/2. Distinct copies of one base critical point have the same type; they cannot create a maximum–saddle pair with base gap zero. The repeated field has cross-cell height ties, but these do not invalidate the candidate argument or its Morse stability. We do not call it a distinct-critical-value global bar field.

For every fixed T>0, (6), delta(G)>0 almost surely and t_N->0 yield

```text
P(any candidate with 0<ell<=Tt_N) -> 0.                (7)
```

One can justify the random radius explicitly: first choose a deterministic delta for which P(delta(G)<=delta) is small, then use convergence in probability of the perturbation below that delta, and finally use Delta(G)>0. This is not conditioning Gaussian noise at a selected critical point. It is a deterministic stability implication on the same global field.

For a marked process put birth locations at M/L_N in the unit torus, lifetime at ell/t_N, and retain b,k,u and, if desired, A_M. On every compact bounded-lifetime mark window its mass tends to zero in probability by (7). Its Laplace functional against any nonnegative continuous test supported in such a window tends to one. Hence no nonzero Poisson random measure on that sector can be its limit. This is an actual candidate-process counterexample to a universal **nonzero** conclusion from U plus the one-contact floor. It is compatible with the zero-intensity Poisson process. No statement about actual global H0 bars is inferred.

### Optional strengthening: nonzero mean with the same empty limit

This paragraph uses a proved fixed-base contact kernel, not assumed coefficient positivity. On the base torus count candidates with r<r0<pi/4 and

```text
b in [-1,1],  k in [1,2],  A_M in [-2,-1],
ell in [t/2,t].
```

Call this count C_G(t). The actual source's §§5,8–10 give the literal Borel Rice kernel and contact convergence. Restricting its compact b,k integral and the conditional A scalar gives a direct specialization:

```text
E C_G(t) ~ c_W(1-2^(-2/3)) t^(2/3),
c_W=6 V0 integral_(b in [-1,1], k in [1,2], u in S^1)
          k^(-2/3) pi_0(v_0) z_0^W(b,k,u) db dk d sigma,
z_0^W=36 k^2 E[A_0^2 1{-2<=A_0<=-1}|U_0=v_0],
V0=(2pi)^2,    v_0=(b,0,0,12k,0,0).                  (8)
```

The seven-row contact covariance is positive, and A_0 given U_0 has positive variance. Therefore its conditional Gaussian density gives z_0^W>0 at every such target, pi_0>0, and c_W is finite and strictly positive. Compact domination and zero Gaussian mass on A-window boundaries give this restricted limit by the same source DCT. This is a specialization derived here from its exact proved kernel, not a claim that the source already stated this particular window. The coefficient 6 comes from integrating the literal density coefficient 4 in lifetime; u has unnormalized arclength 2pi and no extra 1/2.

For t_N=N^-3 and N large, k>=1 makes every radius at most N^-1<r0. Every short base candidate has exactly N^2 translated endpoint lifts on the large torus, and no other lift with short radius. Thus the exact repeated-field count is

```text
C_dom,N(t_N)=N^2 C_G(t_N),
E C_dom,N(t_N)->c_W(1-2^(-2/3))>0,
C_dom,N(t_N)->0 in probability.                       (9)
```

The first equality is N^2 for **short endpoint pairs**. An unrestricted count of all spatial endpoint combinations would have different cross-cell multiplicity and is not substituted here.

For each fixed N and t_N, the count is continuous under sufficiently small C4 perturbations away from critical/mark-boundary events. Those events have probability zero under G: lifetime/birth boundaries use the source's continuous height densities, k boundaries use their positive r change, and A boundaries on the relevant small-r band use the seven-row covariance and its conditional scalar density. The finite critical-branch description makes this pointwise count continuity precise. Global copied height ties within one type cause no max–saddle boundary at ell=0 because base max/saddle heights are distinct.

It follows, for fixed N as eps decreases to zero in the coupling (1), that `P(C_eps,N(t_N)!=C_dom,N(t_N))->0`. Choose a deterministic positive eps_N<=N^-12 so small that this probability is at most N^-10. Such a sequence exists by the just-proved fixed-N continuity; it may, for example, be chosen from dyadic eps values after their eventual probability bound. No explicit numerical eps values or rate of this continuity are claimed.

The accepted a.s. count bound for K=2N gives at most `(16(2N)^2)^2=4096N^4` candidates. The dominant short count is at most 4096N^2. Therefore

```text
E|C_eps_N,N-C_dom,N|
 <=(4096N^4+4096N^2)N^-10 <=8192N^-6 ->0.             (10)
```

This positive full-square counterfamily satisfies U and (3), has an empty-in-probability compact candidate process, and retains the finite nonzero mean in (9). It exhibits failure of uniform integrability, so the first moment does not identify a nonzero Poisson law. This strengthened version is a deterministic existence construction of a coefficient sequence, not the unproved assertion that the explicit eps_N=N^-12 already transfers the mean. Equations (6)–(7) hold for that explicit sequence regardless.

## 6. A useful quantitative Gaussian factorization lemma

Here is a replacement estimate that does follow from stated covariance information. Suppose the two seven-contact vectors have covariance

```text
Sigma=[[Gamma_1,E],[E^T,Gamma_2]],
m0 I<=Gamma_i<=M0 I,
rho_cross=||Gamma_1^(-1/2) E Gamma_2^(-1/2)||op<=1/2.   (11)
```

Let D=diag(Gamma_1,Gamma_2), B=Sigma-D, and Sigma_s=D+sB. Relative quadratic-form comparison gives `(1-rho_cross)D<=Sigma_s<=(1+rho_cross)D`. The centered Gaussian density satisfies

```text
d/ds p_(Sigma_s)(y)
 = (1/2)p_(Sigma_s)(y)
       [y^T Sigma_s^(-1) B Sigma_s^(-1)y
                           -tr(Sigma_s^(-1)B)].
```

All dimensions here are fixed at fourteen. Since ||B||<=M0 rho_cross, integrating this identity gives constants depending only on m0,M0 such that

```text
|p_Sigma(y_1,y_2)-p_Gamma1(y_1)p_Gamma2(y_2)|
 <= C rho_cross (1+|y|^2) exp(-c|y|^2).                (12)
```

This is a proved density interpolation estimate, not a claim that small scalar covariance alone controls an arbitrary nonlinear/global event.

To apply it, let U be the six source contact rows and A=F_vv, with J=(U,A). At a compact target v_0=(b,0,0,12k,0,0), the normalized contact determinant weight is `V0=36k^2 A^2 1{A<0}`. Let w_i be bounded local marks (for example a Lipschitz function of b,k,u,A, supported in fixed compact b,k and negative-A windows). Define the one-fold contact amplitude

```text
K_i^0=12*36 k_i^2 integral_(A_i<0)
                  A_i^2 w_i p_(J_i)(v_i,A_i) dA_i.
```

The corresponding twofold amplitude is

```text
K_12^0=12^2*36^2 k_1^2 k_2^2
    integral_(A_1,A_2<0) A_1^2 A_2^2 w_1 w_2
                  p_(J_1,J_2)(v_1,A_1,v_2,A_2) dA_1 dA_2.
```

Integrating (12) against the displayed polynomial weights yields, uniformly on the compact b,k marks,

```text
|K_12^0-K_1^0 K_2^0|<=C rho_cross.                    (13)
```

The type and determinant factors occur exactly once. This remains true for bounded Borel **local** marks in the contact variables; the finite-r convergence rate below uses Lipschitz marks. A full-field elder selector is not a local jet mark and cannot be factorized by (12). It needs a separate weighted localization/selection estimate.

For a stationary physical covariance R(h), every entry of E is a coordinate derivative of R of total order at most six, expanded in the two frames. If

```text
d(Rsep)=sup_(dist(h,0)>=Rsep, |alpha|<=6)
                                      |partial^alpha R(h)|,
```

the coefficient l1 bounds in two dimensions give `||E||op<=56 d(Rsep)`: each entry has at most a factor `(sqrt(2))^6=8`, and the 7-by-7 operator norm is at most seven times its maximum entry. Therefore `rho_cross<=56d(Rsep)/m0`. A quantitative physical covariance derivative bound is a concrete sufficient far input. Counterexample (4) violates even its order-zero requirement.

## 7. Finite-r kernels: the actual row coupling and a conservative bound

One must also align the contact amplitude with the finite-radius observable. For two center/frame configurations, write U_r=(U_r1,U_r2), dimension twelve, with the source's literal divided differences and targets. Assume on the middle band that the **joint seven-row contact covariance** has least eigenvalue at least 0<m<=1. Uniform U gives a fixed upper bound on all row L2 norms and the estimates

```text
||L_(U_r)-L_(U_0)||HS<=C(r_1+r_2),
||Cov(U_r)-Cov(U_0)||op<=C(r_1+r_2).                   (14)
```

Here L denotes the row map from the actual finite standard coefficient vector. Integral representations of the rows use derivatives through order three; comparing them to the center uses order four. Their L2 bounds follow from U, without a coefficient-dimension factor. Thus joint Cov(U_r)>=m I/2 if r_1+r_2<=c m.

The physical normalized endpoint Hessians need more than the seven conditioned rows. Use the following **unconditioned regular row extensions**, with the first pair written from M to S:

```text
alpha_M#=[F_uu(M)-(F_u(S)-F_u(M))/r]/r,
alpha_S#=[F_uu(S)-(F_u(S)-F_u(M))/r]/r,
beta_M# =[F_uv(M)-(F_v(S)-F_v(M))/r]/r,
beta_S# =[F_uv(S)-(F_v(S)-F_v(M))/r]/r,
A_M=F_vv(M), A_S=F_vv(S).
```

For example `alpha_M#=-integral_0^1(1-s)F_uuu(M+rsu)ds`, and alpha_S# uses +s; beta uses F_uuv. These are uniformly bounded L2 rows and have O(r) L2 errors from their contact limits. Under the exact gradient pins they equal the physical alpha=F_uu/r and beta=F_uv/r. This step avoids asserting bounded unconditional covariance of the singular rows F_uu/r or inventing a full augmented-jet rank.

Canonical Gaussian conditioning at each r can be coupled using the same independent standard coefficient vector Z':

```text
c_r=L_r^T (L_rL_r^T)^(-1) v_r,
P_r=I-L_r^T(L_rL_r^T)^(-1)L_r,
conditional row vector = L_E,r c_r + L_E,r P_r Z'.
```

The compact target has v_r-v_0=O(r), where r=max(r_i). Inverse perturbation and (14) give `||c_r-c_0||<=C r m^-2`, `||P_r-P_0||op<=C r m^-2`. Because the E row dimension is fixed and its Hilbert–Schmidt norm is bounded, the conditional E-row difference has every fixed Lp norm bounded by C_p r m^-2. Its moments are bounded by C_p m^-1. These statements use finite Gaussian vectors only and do not require the conditional E covariance to be invertible.

The congruence-scaled physical Hessian is

```text
Htilde_i=[[alpha_i,sqrt(r) beta_i],
          [sqrt(r) beta_i,A_i]],
det Htilde_i=det H_i/r.
```

Under the coupling it differs in Lp from diag(-6k,A_0) at M and diag(6k,A_0) at S by at most C_p sqrt(r)m^-2. The maximum weight is the product of negative eigenvalue parts, and the saddle weight is (-det)_+. These are continuous, locally Lipschitz functions with quadratic growth. Their product at two endpoints has degree four; the product for two folds has degree eight and local Lipschitz growth degree seven. For bounded Lipschitz marks in A, the mark difference separately contributes O(r m^-2) times a degree-eight moment. Consequently, on the specified compact targets,

```text
E|V_r1 V_r2 w_r1 w_r2 - V_01 V_02 w_01 w_02|
 <=C[sqrt(r)m^-9+r m^-10] <=C sqrt(r)m^-10.
```

The twelve-dimensional pin density is at most C m^-6. Its perturbation at compact targets is at most C r m^-8 by the same differentiated-density identity as Section 6. Combining these bounds gives conservative, unoptimized estimates

```text
|K_12^r|<=C m^-14,
|K_12^r-K_12^0|<=C sqrt(r)m^-16,                      (15)
K_12^r=12^2 p_(U_r1,U_r2)(v_r1,v_r2)
                    E[V_r1 V_r2 w_r1 w_r2|U_r=v_r].
```

For (15), moments of the degree-eight weight are bounded by C m^-8. The density error contributes r m^-16; the unmarked weight error contributes sqrt(r)m^-15; the additional Lipschitz mark error contributes r m^-16. Each is bounded by C sqrt(r)m^-16 when r,m<=1. No inverse physical Hessian, inverse endpoint normalizer or covariance of every Hessian entry was used. Hard additional A-window indicators require a boundary anti-concentration argument instead of a Lipschitz rate; continuous compact test marks avoid that extra step. Their upper bound still follows by dropping the mark.

## 8. What quantitative middle/far hypotheses actually suffice

For a varying source law let V=L^2, r_t=(Tt/k_-)^(1/3), and choose cutoffs

```text
r_t << eta_t << R_t << L,
V t^(2/3)=O(1).
```

The last condition is a stipulated total rare-intensity scale, not a one-point theorem. In addition to U and fixed positive a,k_-, the following are explicit covariance estimates to prove:

* A uniform one-contact bound m0 I<=Gamma_0(u)<=M0 I, and convergence of these seven-row covariances to a specified limiting covariance if an intensity is to be identified.
* On eta_t<=dist(x_1,x_2)<=R_t, a joint contact floor `m_t>0`, with `r_t<=c m_t` and `V R_t^2 t^(4/3) m_t^-14->0`.
* On dist(x_1,x_2)>=R_t, cross derivative decay d_t(R_t)->0 through order six, small enough eventually that 56d_t/m0<=1/2.

For the middle domain, finite-r rank in (14) implies rank of the original twelve value/gradient observations by the actual invertible per-pair transforms. The source's coefficient-space area formula extends in the same finite dimension to four distinct critical endpoints: choose an invertible eight-gradient minor, differentiate to get the four physical Hessian blocks, and partition the position domain. This gives the genuine four-point Borel identity with the determinant product once. Endpoints cannot overlap on the middle/far domains when r_t<<eta_t; shared incidences lie in the near region.

Each pair has exactly the source ledger

```text
(r dr d sigma)(r^3 db dk)(12r^-5 pin_density)(r^2 V_r)
                    =r dr db dk d sigma (12 pin_density V_r).
```

For two pairs the ledger is its product. The lifetime integrals give O(t^(4/3)) on compact k. The middle spatial volume is at most C V R_t^2. Equation (15) therefore proves

```text
middle contribution <=C V R_t^2 t^(4/3)m_t^-14=o(1).   (16)
```

For the far domain, (13) and (15) now have a constant covariance floor, so

```text
|K_12^r-K_1^r K_2^r|
                <=C[d_t(R_t)+sqrt(r_t)].
```

Its integrated error is at most

```text
C V^2 t^(4/3)[d_t(R_t)+t^(1/6)]=o(1)                 (17)
```

under the displayed rare-intensity scale. The product kernel on the omitted middle spatial region also has integral O(V R_t^2 t^(4/3)) and vanishes if R_t^2/V->0. The near diagonal can be handled by the separate local-norm/count proposal if its hypotheses and review permit reuse; it is not supplied by (16)–(17).

For illustration, if `m_t>=c eta_t^nu` for some fixed finite nu, `eta_t` is of logarithmic order, `R_t^2=O(log(1/t))`, and V t^(2/3)=O(1), then (16) is O(t^(2/3) times a fixed power of log(1/t)) and vanishes. Also r_t/m_t->0. A finite exponent is enough; one need not assume a separation-independent floor at eta_t->0.

These are sufficient twofold **candidate** estimates on continuous compact local marks. They do not prove convergence of all factorial moments or a Laplace functional. In particular the iid-source weighted-cluster theorem is not a spatial mixing theorem. Actual global H0 selectors need uniform weighted local-selection/remote-exclusion error estimates before the local kernels can replace them; finite-dimensional jet decorrelation cannot factorize a global selector by itself.

## 9. A concrete source-specific Gaussian route, with residual work visible

The iid source's actual covariance is the infinite periodization of exp(-|h|^2/2), with physical Fourier weights proportional to exp(-|xi_n|^2/2). A natural finite family that respects this identity is the exact normalized truncation

```text
q_(K,L),n = exp(-|(2pi/L)n|^2/2)
                 /sum_(|m|infinity<=K) exp(-|(2pi/L)m|^2/2),
|n|infinity<=K,
Lambda=(2pi/L)K.
```

This is a specific candidate source family, not the arbitrary positive spectra of U. For L large and Lambda>=4, elementary Gaussian grid-sum comparison gives a uniform physical moment bound and a derivative tail through order six,

```text
delta_6(K,L)
 =sum_(|n|infinity>K) q_infinite,L,n(1+|xi_n|^6)
 <=C exp(-Lambda^2/64).                               (18)
```

A route to (18) is explicit: the infinite normalizer is bounded below by c omega^-2 using frequency cells in a fixed physical square; compare each tail grid point to its omega-cell, omega<=1, use `|xi_n|^2>=|x|^2/2-1`, and integrate outside radius Lambda-1. A polynomial Gaussian tail is bounded by C exp(-Lambda^2/64). Normalizing the truncation changes every covariance derivative of order <=6 by at most C delta_6, once the omitted mass is <=1/2. The same comparison at moment twelve supplies U. Constants in this paragraph are existence bounds from elementary Gaussian integrals, not evaluated numerical enclosures.

For the infinite periodization, differentiating the Gaussian images gives

```text
sup_(|alpha|<=6) |partial^alpha R_infinite,L(h)|
                      <=C exp[-dist_T(h,0)^2/8],
```

uniformly for L bounded below away from zero (and in particular for L large). Absorb the finite Hermite polynomials into exp(-|x|^2/4), factor out exp(-dist_T^2/8), and bound the remaining separated lattice Gaussian sum. Thus the truncated far input has the concrete candidate bound

```text
d_t(R)<=C[exp(-R^2/8)+delta_6(K,L)].                  (19)
```

For a middle lower bound, the Euclidean Gaussian covariance has strictly positive two-contact covariance at distinct centers and every frame. Its positive spectral density makes a vanishing variance imply that an exponential-polynomial multiplier vanishes identically. Its inverse Fourier distribution is a sum of finite jets supported at the two different centers; their disjoint supports force each local jet functional to vanish, and the seven contact rows at one point are independent. This is a finite named-jet argument, not an arbitrary rank claim for a finite support.

There is a possible qualitative-to-power improvement for this one fixed analytic kernel. Parameterize the separation as s e, s real near zero, and the frames by angle charts. The determinant of its fourteen-row covariance is real analytic. It is positive for s!=0 and zero at s=0, because fourteen third-order rows at one point lie in a ten-dimensional full third jet. A compact real-analytic distance inequality then gives `det>=c s^nu` for some finite nu, uniformly in e and frames on a fixed small s band. Dividing by a uniform bound on the other thirteen eigenvalues gives a least-eigenvalue lower bound of the same power. Compactness handles the remaining bounded separations; Gaussian derivative decay handles arbitrarily large separations.

The named classical import for that last implication is the compact real-analytic distance inequality, inspected as Theorem 3.1 in Teissier's [*Some resonances of Lojasiewicz inequalities*](https://webusers.imj-prg.fr/~bernard.teissier/documents/Lojasiewicz_2012.pdf), which explicitly attributes it to Lojasiewicz. The consumed read is the theorem statement, not the complete paper or a fresh read/proof of the cited original 1959 work. A saved copy and exact identity are in [jet-factorization-primary-sources/SOURCE_IDENTITY.json](jet-factorization-primary-sources/SOURCE_IDENTITY.json). On the compact angular charts the analytic zero set is s=0, so its distance is comparable to |s|. The import supplies some finite exponent, not an evaluated exponent or effective constant.

To transfer that floor to a finite truncation at eta_t, one needs

```text
delta_6(K,L)+C exp(-c L^2)=o(eta_t^nu).                (20)
```

On a fixed bounded separation band the periodization differs from the Euclidean covariance by the displayed image error; on the remaining large-separation band (19) gives a constant floor after choosing a fixed sufficiently large distance. Weyl's inequality then transfers the middle lower bound. Choosing Lambda^2 comparable to log(1/t), eta_t of logarithmic order, and L of order t^(-1/3) makes (18), (20), the r_t/m_t requirement and (16)–(17) compatible. K is then of order L sqrt(log(1/t)), so the logarithmic near cutoff also remains compatible with r_t<<eta_t.

This is a concrete replacement route linked to the actual Gaussian physical covariance. It has not been incorporated into either repository or independently reviewed. Before treating it as a discharged source theorem, review the exact compact analytic determinant application, grid-sum/image constants, finite-r coupling/kernel bounds (14)–(15), and all-Borel four-point area-formula application. An evaluated nu is unnecessary for the logarithmic existence regime, but a numerical window would require more. The project's existing finite K2/K3 carrier cannot be relabeled as this changing truncation, and its singular enlarged-height strata remain negative evidence.

Even a successful compact-candidate second-factorial calculation does not supply the full point-process theorem: higher configurations or an appropriate dependency/Laplace method, one-point marked convergence/tight mark tails, actual selected-bar localization and complements, and global topology remain work. No region owner is transferred by this proposed route.

## 10. Controls and status

[jet_factorization_obstruction_controls.py](jet_factorization_obstruction_controls.py) and [JET_FACTORIZATION_OBSTRUCTION_CONTROLS.json](JET_FACTORIZATION_OBSTRUCTION_CONTROLS.json) use exact Fraction/integer arithmetic. They check normalized positive full-square mixtures, the uniform physical moment bound, the single-jet covariance mixture identity, the exact resonant geometric sum/two-value eigenvalue, an axial seven-row base LDL with positive pivots, perturbation/count-cap exponents, the literal ordered fold Jacobian and conservative middle/far powers. The controls also check consumed frozen-file hashes. No floating eigenvalues, field draws, seeds, approximate roots, Rice integrals or Poisson simulations were used.

Finite tests do not prove all-frame positivity, random Morse stability, the diagonal count-continuity construction, the Gaussian interpolation bound, row-coupling estimates or the analytic power-floor route. Those are the conventional proofs/imports above and need the new separate review. The exact counterfamily and the explicit empty-in-probability result are stronger than a failed estimate; the optional mean-preserving version carefully records its nonnumerical diagonal choice. None is an actual-global-bar counterexample.

All earlier frozen proposals, controls and reviews are unchanged. The uniform local-norm proposal remains under its separate review and is not being reviewed again here. This local report has no source admission, production/native/remote write, graph transition, scientific promotion or provider-independent approval effect. Actual author/principal is OpenAI/Codex, organizational-independence credit 0.
