# RN5 — Corrected near-region moment bound and local spatial certificates

2026-09-17 · ROUND5-20260917-b9c2 · OpenAI / Codex · author-side candidate.

The older near-region envelope uses the wrong determinant moment. This release proves the defect, implements the corrected inequality for the actual SIDE24 field, and supplies 65 point-law certificates and 10 genuine spatial-box certificates. The complete near-annulus integral, the all-small-r result, and independent acceptance remain open.

The existing RN3 far-region proof uses the correct second moment and is outside this defect's affected scope. Its prior author-side result is retained, with its review gate still open.

## 1. Exact defect and correction

Write A=det H_M, B=det H_S, C=det H_y under the nine-pin conditional law. Dropping the maximum/saddle indicators and applying Hölder with exponents 4,4,2 gives

\[
E[|ABC|1_{M\text{ max}}1_{S\text{ saddle}}1_{y\text{ saddle}}]
\le (EA^4\,EB^4)^{1/4}(EC^2)^{1/2}.\tag{1}
\]

The pinned `d3_perc.py` function `envelope_v` instead returns `(EA^4 EB^4)^(1/4) sqrt(EC^4)` and describes it as Cauchy–Schwarz twice. Its `det_moment(mu,S,k)` is the kth determinant moment, so this is a substantive power error. `window_cap_env` and the `d3_amend_v2.py` annulus/remote assembly consume that function. The exact CL-RNU-003 progress report repeats the same expression in Section 3.

There is no general inequality allowing the second moment in (1) to be replaced by the fourth moment with the same square-root exponent. For small determinants the replacement can decrease the bound.

### Exact typed, nondegenerate Gaussian counterexample

Take all nine symmetric-Hessian coordinates mutually independent with variance 10^-6. Let their (xx,yy,xy) means be (-1,-1,0), (1,-1,0), and (1,-1/4,0). With probability at least 91/100, every coordinate differs from its mean by at most 1/100, by the union bound and Chebyshev. On this event the first Hessian is negative definite and the other two are saddles. Their determinant magnitudes are at least 98/100, 9801/10000, and 2376/10000. Thus the typed expectation is at least

\[
\frac{91}{100}\frac{98}{100}\frac{9801}{10000}\frac{2376}{10000}
=0.207675035568.
\]

The old envelope is approximately 0.06250406249. The verifier proves **old envelope < 63/1000 < 207/1000 < typed expectation** using exact rational comparisons of fourth powers. The complete rational moment and separation values are in `NEAR_MOMENT_REPAIR_CERTIFICATE.json`.

This disproves the claimed universal Gaussian inequality. It does not establish that the actual SIDE24 typed expectation exceeds any particular old numerical table entry. The affected certification route is invalid regardless of whether some individual entries happen to overestimate the true expectation.

## 2. Bound source and scientific scope

The field, six fixed pins, Hessian order, and mark interval match the pinned RN3 object:

\[
K(x)=k(x_1)k(x_2),\quad
k(s)=\frac{\sum_{j\in\mathbb Z}e^{-(s+24j)^2/2}}{\sum_{j\in\mathbb Z}e^{-(24j)^2/2}},
\quad r=1/20,\ b=6/5,
\]

\[
M=(-r/2,0),\ S=(r/2,0),\quad
c=(b,0,0,b-r^3/6,0,0),\quad v\in[b-r^3/6,b].
\]

Fixed jets have order (f,fx,fy); Hessians have order (fxx,fyy,fxy). All calculations condition additionally on (f(y),fx(y),fy(y))=(v,0,0). The typed pair normalizer is bounded below by the imported H3 floor Z_lo=0.0077592917375327855. Its source hash and exact decimal are checked before calculation. This release imports that floor and does not re-prove H3.

`closure_round2/rn_field.py` is pinned to SHA-256 `d9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8`. It supplies Arb-enclosed image sums, infinite image tails, spectral derivative moments, normalization, and fixed pin conditioning. The five source dependencies are checked by full hash. The scientific carrier is Drive `1vSI-evINWskhXVyiZ0slt-rLT74sPpXH`, archive SHA-256 `a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b`.

## 3. Whole-mark envelope at an exact spatial point

The conditional nine-Hessian mean is affine in t=v-(b-r^3/12), and the conditional covariance does not depend on v. Wick recurrence therefore gives exact polynomials: degree at most 8 for EA^4 and EB^4, and at most 4 for EC^2. Arb encloses their coefficients.

For p(t)=sum c_j t^j and |t|<=h=r^3/12, use

\[
\sup p(t)\le c_0^{\rm hi}+\sum_{j\ge1}|c_j|^{\rm hi}h^j.\tag{2}
\]

All factors are bounded on the complete mark interval before taking roots and multiplying. Multiply by the exact conditional gradient density and the enclosed conditional height-window probability, then divide by Z_lo. The error-function implementation switches to complementary tails to avoid catastrophic subtraction in tiny tails. No three-station spread heuristic or numerical quadrature is used for the mark bound.

All 65 locations on the historical 13-radius, 5-angle D4 net pass at 384 bits. At d=0.1, theta=45 degrees, sqrt(EC^2/EC^4) at the central mark is about 289.00; at d=5 it is about 0.1155. The change is not a uniform rescaling.

For planning only, the same coarse polar trapezoid applied to the new whole-mark station caps gives approximately **2.34195 r^3**, versus **17.67237 r^3** for the wrong-power version with the new mark caps. These are **diagnostic sums, not bounds on the spatial integral**. The older published 17.6804 used a different mark-cap heuristic. None of these numbers closes Piece 2.

## 4. Spatial certificate: centered Taylor jets and separate marginal whitening

Let F be the Gaussian field after the six pins. At the center z of a Cartesian box, collect the pair Hessians and all derivatives of F through order N+2, N=6. Their mean and covariance are computed from the exact kernel before any spatial interval substitution.

Expand the moving value/gradient and third Hessian in delta=y-z. For every derivative a of order at most 2, the centered Taylor remainder has the L2 bound

\[
\|R_a\|_2\le\sum_{|\alpha|=N+1}
\frac{h^{N+1}}{\alpha!}
\sqrt{\operatorname{Var}(\partial^{a+\alpha}f)}.\tag{3}
\]

Here |delta_i|<=h. Formula (3) follows by applying the one-variable integral Taylor remainder to F(z+t delta) and Minkowski. Conditional centered derivative variance is bounded by the unconditional stationary variance, since Gaussian conditioning subtracts a positive semidefinite matrix. The mean remainder is bounded by the same expression times sqrt(c^T G_6^-1 c), by Gaussian regression and Cauchy–Schwarz. These derivative moments include the spectral tails.

At the box center, put D=Cov(Y), A=Cov(H,Y)D^-1, and T_Y=chol(D)^-1. Use **separate** Cholesky factors L_M,L_S,L_y of the three conditional marginal Hessian covariances; let L be their block diagonal matrix. Transform to

\[
Z_Y=T_Y Y,\qquad Z_H=L^{-1}(H-A Y).
\]

Each diagonal three-dimensional covariance block is the identity at the center; Cov(Z_H,Z_Y)=0. The three Hessian blocks may be mutually correlated. Equation (1) needs only their marginal moments, so no assumption of independence or well-conditioned joint nine-dimensional covariance is imposed.

The program assembles every covariance polynomial coefficient by summing all equal monomials **before** substituting the spatial interval. This retains cancellations. If P_i is the centered Taylor polynomial and R_i its remainder, the covariance error is bounded by

\[
|\operatorname{Cov}(P_i,R_j)+\operatorname{Cov}(R_i,P_j)+\operatorname{Cov}(R_i,R_j)|
\le\sigma_{P_i}\epsilon_j+\sigma_{P_j}\epsilon_i+\epsilon_i\epsilon_j.\tag{4}
\]

All terms in (4) have certified upper bounds from (3) and the polynomial covariance. For the resulting enclosed blocks G,B,V, condition Z_H on Z_Y=T_Y(v,0,0): covariance V-BG^-1 B^T and mean mu_H+BG^-1(T_Y(v,0,0)-mu_Y). Undo L and A, form the marginal determinant polynomials, and apply (1)-(2). Positive interval Cholesky pivots are required for G and each conditional Hessian marginal.

The height integral is bounded by its length times the supremum of the full three-jet Gaussian density. A lower bound for its quadratic form is clipped at zero, a valid conservative step; det(T_Y) supplies the change-of-variables factor. This completes a bound for every spatial point and every mark in each accepted box. Failed larger boxes are recorded and refined, never accepted.

### Local results

Each square has center (d cos theta,d sin theta) and halfwidth h. Values below are strict upward roundings of the certified spine/r^3 upper bounds.

| d | theta (degrees) | h | Upper bound |
|---|---:|---:|---:|
| 1 | 0 | 0.005 | 0.007504612 |
| 1 | 45 | 0.005 | 0.039947969 |
| 1 | 90 | 0.005 | 0.005653819 |
| 1 | 135 | 0.005 | 0.056481516 |
| 1 | 180 | 0.005 | 0.005863701 |
| 0.1 | 45 | 0.00001 | 0.396522869 |
| 0.1 | 90 | 0.00001 | 0.271032862 |
| 0.1 | 0 | 0.0000001 | 6.264516 x 10^-8848 |
| 0.5 | 45 | 0.002 | 0.052419488 |
| 3 | 45 | 0.01 | 0.031039628 |

At d=1, theta=45 degrees, the bound/center ratio is about 1.60470. CL-RNU-003 reported about 2.8 x 10^19 at nominal halfwidth 0.005 under the older method. The boxes and corrected integrands differ, so this is a feasibility comparison, not a proof of formal numerical dominance. Near the axis at d=0.1, substantial refinement and a loose relative density estimate remain; the absolute bound is nevertheless extremely small. Ten boxes do not cover the annulus.

## 5. RN3 / CL-RNU-003 reconciliation

| Dimension | RN3 exact proof | CL-RNU-003 exact progress report | Disposition |
|---|---|---|---|
| Fixed rung / endpoint positions | r=0.05, endpoints +/-0.025 | Same explicit rung and positions | Agrees at declared level |
| Full field, mark, and normalizer binding | Formula, six pins, complete mark window, H3 hash | Refers to inherited exact-kernel engine, D4 net, normalized frame | Full executable binding still requires the named CL code/archive |
| Comparison | One joint Gaussian comparison of all nine Hessian coordinates | Product of pair/y factors plus a cross correction and density/window ratios | Distinct majorants |
| Far correction | Uniform upper correction <0.12058 | Peak diagnostic about 0.67728; partial certified cells | Values need not agree or conflict |
| Coverage | 13,604 boxes, full declared far region, author-side | 1,170 closed cells and 561 pending at this report's timestamp | No transfer of completeness |
| Third determinant moment | Exact typed second moment in RN3 Section 3 | Wrong fourth-moment square root in near Section 3 | Near defect does not invalidate RN3's formula |
| Acceptance | External review open | Proposed progress, no canonical impact | No organizational credit added |

RN3 raw proof: 12,956 bytes, SHA-256 `0c9446b7cb49e6e1e45e43f84bb72ce680f022882e225fdff61c8e6edede1373`, Drive `1f0EBsq6UIZ0jDCABT251TTNEWYCVA0FO`.

CL-RNU-003 raw report: 7,407 bytes, SHA-256 `59b8f002f1b1e81e7829416baf16ef1d84b1326d4b3b4604b833e5a55a9d6d53`, Drive `1aCa-QG9CSrNUB9SUFKISifghSf-41fRy`.

The named `CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip`, `rnu_t4.py`, and `rnu_spine.py` were not returned by the bounded exact-name lookup or broader bundle-name lookup. **CANNOT_VERIFY: the current CL executable and checkpoint identity**, pending those exact bytes and hashes. The report-level defect is directly verifiable without them. A later CL progress state must be inspected before making a current coverage claim.

## 6. Disposition and next exact action

1. Exclude the old `envelope_v` certification claim, its annulus/remote consumers, and CL-RNU-003's Section 3 integrand-certification/forecast claims. Keep frozen bytes and unaffected statements intact. `SCOPED_HOLDS.json` records exact objects, affected scopes, and restoration conditions.
2. Use the corrected implementation for successor work. A qualified nonauthor reviewer should reconstruct (1)-(4), replay the point/box certificates, and audit image/spectral remainders and inherited H3 scope.
3. Build a complete nonoverlapping spatial cover of 0.1<=|y|<=5, retaining boundary area bounds and every rejected cell. Sum area times the corrected cell supremum, verify that no cells remain pending, then reassemble the remote budget. Treat the near-axis refinement cost explicitly; the present ten boxes are not a coverage certificate.
4. Preserve the fixed-r and fixed-axis scope. All-small-r extension and theorem-specific external predicates require their own evidence.

Validation: 66 consistency checks pass under Python `-O`, including exact source extraction, rational counterexample/margins, three independently assembled raw nine-pin Gaussian laws, and 40 box-corner comparisons. Corner comparisons check implementation consistency; equations (1)-(4) justify the uniform boxes. The code and outputs are author-side, fully exposed, and earn **zero organizational-independence credit**. This session continues the OpenAI RN3 author lineage and corrects its own lineage's LM004 review report.

## 7. External reconnaissance and reuse

Fresh primary-source reconnaissance on 2026-09-17 checked Gaussian conditioning with derivative observations and stable Taylor residual representations. [Rasmussen and Williams, GPML Appendix A](https://gaussianprocess.org/gpml/chapters/RWA.pdf) supplies the standard conditional Gaussian formulas and Cholesky framework reused here. [Karvonen et al., A probabilistic Taylor expansion with Gaussian processes](https://arxiv.org/abs/2102.00877) provides related Taylor/derivative-observation context. No theorem from that paper is imported as a certificate for this field. The L2 remainder and covariance bounds above are derived explicitly for this implementation; no novelty claim is made.

Replay commands and exact payload identities are in the delivery bundle's `README.md` and `DELIVERY_MANIFEST.json`.
