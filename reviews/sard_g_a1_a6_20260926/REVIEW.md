# PR122 successor: bounded A1/A6 mathematical review

Review date: 2026-09-26. Repository: [d6g8k5htny-coder/main, PR122](https://github.com/d6g8k5htny-coder/main/pull/122).

**Disposition: A1 AMEND; A6 AMEND.** The successor excludes the previously identified hidden critical point, but its fixed gradient-exclusion threshold still admits equality. Consequently its chart set need not be open. Below is an analytic counterexample at an actual Morse saddle connection, followed by the conditional slicing argument and the precise amendment it requires. This is a counterexample to the stated chart-openness assertion, not to the desired almost-sure theorem.

| Interface | Verdict on the exact source below | Reason |
|---|---|---|
| A1: open countable charts and connection coverage | **AMEND** | The condition `min_K |grad f| >= eta` is not open. Unique-critical-point and strict spectral-gap conditions do not remove its equality boundary. |
| A6: chart-validity intervals and conditional Gaussian slicing | **AMEND** | The assertion that the validity set on a line is open invokes the false A1 premise. With genuinely open charts and the stated A2–A5 dependencies, the Borel/Tonelli argument below is valid. |

No new verdict is issued on A2–A5, finite-jet nondegeneracy, parent C103, or any downstream theorem. The full-Gaussian C103 corollary remains on HOLD under the source's own review rule. No scientific disposition or author source has been changed.

## Exact object and review exposure

- Reviewed PR head: `6a7f1f6cfb49703b7b758343d65b3b65abe6b11b`.
- Reviewed Git tree: `f2deec3727a73ab803b13f314d1b62f9f9f404b4`.
- [Complete proof at that head](https://github.com/d6g8k5htny-coder/main/blob/6a7f1f6cfb49703b7b758343d65b3b65abe6b11b/research/q0/sard_g/SARD_G_EXECUTION_REPAIR_20260925.md): `research/q0/sard_g/SARD_G_EXECUTION_REPAIR_20260925.md`.
- Proof blob: `a0e9f7d8960f785281fea5aef1bc8b011cbbb055`; 14,672 bytes; SHA-256 `6b04ad32e55c3e0c4f489e45f7687b0f96ec9e1ad25d01b2de5df50a6ab333d0`.
- Predicate-repair author commit: `b36222248e6909a2cae19639dbd2896818444e00`. The engineering merge above retains these proof bytes.
- Predecessor reviewed in [comment 5841682597](https://github.com/d6g8k5htny-coder/main/pull/122#issuecomment-5841682597): commit `d49be07074bd0696cc034757379771eae3c852de`, proof blob `3ae92848d606cfbc22c44a2281610822e799f88e`, 9,952 bytes, SHA-256 `2f02495e272bc26054311cea03b66ee1de0ca6079dd097deb3084794a9514032`.

Reviewer: OpenAI / ChatGPT (Codex agent), source-exposed technical reviewer. The source's author line is OpenAI / ChatGPT. This is a same-provider review and claims **zero organizational-independence credit**. It is not blind review. The predecessor review identifies its reviewer as Grok 4.7 / Cursor; that reviewer then authored the predicate successor, so its repair receipt is not fresh acceptance of its own successor. No child agents were used for this review.

I read the complete successor proof, `SOURCE_MAP.json`, `RECONNAISSANCE.md`, the complete original review comment, the complete parent working manuscript, the complete KIMI-AUD-022 review, and the complete negative-control script. I independently checked the source bytes and that Sections 4, 5, and 6 are byte-identical to the predecessor. These comparisons verify identity, not their mathematics.

Additional exact source identities:

| Source | Git blob | Bytes | SHA-256 |
|---|---|---:|---|
| Parent Gaussian-transversality working source | `a0f4aa6ab6d0e21e7a24b4a6e1aa4fdebc40efdc` | 19,242 | `4a77f3b7ed390c0ebff3c580baa4d8a107bf3cf2b892c5927e5ffe005ad1c1fe` |
| KIMI-AUD-022 | `e968385a3bf4e8318a9ab6cb4f57818828f01b18` | 9,527 | `8efcd937502b67ab515d41435eace639000c287ce5e6773105953319f48429b8` |
| `SOURCE_MAP.json` | `78a2a4550bbd83071ba7089a2da2c0d6d37dbda3` | 1,354 | `934ae22905f9ac58a3854053202ebe61b79fc53805fdae964749cb9130f49b40` |
| `RECONNAISSANCE.md` | `63f5aa4c0ebfa27ba3b20fad11b5914ba974cd2e` | 1,426 | `90623b128f0de91cc29818bc12392b354a51995e2f4c1bf67a94dd8112db9330` |
| `chart_predicate_negative_control.py` | `2892248934bc0ac9ba3463faa51917b6b9e085ac` | 9,443 | `a2a45454954a4235fe214101783ac62970c891fe3f9b96300d98b980b70f8f7e` |

The source map's paths expose the parent and specialist review as repository files; no private Drive access is required to inspect these sources. The source map pins finite-jet input blob `1bea9e62ef1ec621f51953dd3f5d5410fba616af`; that input was not re-proved in this review.

## A1: the remaining nonopen boundary

Section 3 condition 1 requires that the gradient norm be **at least** the rational threshold `eta` on the compact exclusion set. Its following paragraph explicitly writes `|grad f| >= eta` and calls that condition open in the C1 topology. That assertion is false. A strictly positive threshold is not the same as strict slack above a fixed threshold. The Hessian condition already says its spectral gap **exceeds** `delta`; that separate strict condition is not the defect.

For any fixed nonempty compact set K, write

$$m_K(f)=\min_{x\in K}|\nabla f(x)|.$$

The functional is continuous, indeed

$$|m_K(f)-m_K(g)|\leq\|\nabla f-\nabla g\|_\infty.$$

Thus `m_K(f) > eta` is open, whereas the admitted equality case in `m_K(f) >= eta` can be a boundary point. The following example verifies this inside the other chart conditions, at a genuine saddle connection; it is not merely a counterexample to an isolated scalar inequality.

### Explicit torus example

On the unit torus let

$$F(x,y)=\sin(2\pi x)\bigl(2-\cos(2\pi y)\bigr).$$

Then

$$F_x=2\pi\cos(2\pi x)\bigl(2-\cos(2\pi y)\bigr),\qquad
F_y=2\pi\sin(2\pi x)\sin(2\pi y).$$

Because `2-cos(2 pi y)` is positive, its only critical points have

$$x\in\{1/4,3/4\},\qquad y\in\{0,1/2\}.$$

Writing `a=(2 pi)^2`, the Hessians at

$$p=(3/4,0),\qquad q=(1/4,0)$$

are respectively `diag(a,-a)` and `diag(-a,a)`. Both are index-one saddles for the ascending gradient flow. The other two critical points are a minimum and a maximum; the four critical values are `-3,-1,1,3`. In particular F is Morse and has distinct critical values, so it satisfies the generic conditions stated in Section 2.

The line `y=0` is invariant. In cover coordinates on the arc

$$3/4<x<5/4,$$

the flow obeys `dx/dt=2 pi cos(2 pi x)>0`. The orbit approaches p backward in time and q, represented as `(5/4,0)`, forward in time. This is a saddle-to-saddle connection.

Choose square coordinate boxes centered at p and this lift of q, with rational halfwidths

$$r_+=1/20,\qquad r_-=3/100,\qquad r_N=1/100.$$

They give disjoint outer boxes, nested inner boxes, and continuation neighborhoods with the required compact containments. Each closed outer box contains exactly its designated saddle. Set

$$K=\overline{B_p^+\setminus N_p}\ \cup\ \overline{B_q^+\setminus N_q},
\qquad m=\min_K|\nabla F|.$$

K is compact and contains no critical point, so `m>0`, and its minimum is attained. Define

$$f=F/m,\qquad \eta=1.$$

The threshold is now exactly rational, and `min_K |grad f|=1`. Both endpoint exclusion conditions hold as currently written; at least one is attained with equality. The Hessian gaps at p and q equal `a/m`; choose any rational `0<delta<a/m`.

The remaining chart data can have strict slack. Take local sections through the horizontal branch at `x=77/100` and `x=123/100`, inside the corresponding inner boxes, and choose the other members of the branch-label section pairs on the opposite local branches. Choose the central transverse section through `x=1`, with a sufficiently small rational vertical extent. The tracked compact horizontal arcs are regular, their vertical-section crossings are transverse and interior, and each reaches the central section for the first time without an earlier crossing. A narrow rational polygonal tube can contain these arcs with positive clearance. Their speed has positive minimum, their crossing angle is bounded away from tangency, and their finite travel times admit rational strict lower/upper margins as appropriate. Thus these data define a chart chi in which f satisfies all four current membership conditions and `D_chi(f)=0`.

For `0<epsilon<1`, put

$$f_\epsilon=(1-\epsilon)f.$$

Then

$$\|f_\epsilon-f\|_{C^2}=\epsilon\|f\|_{C^2}\longrightarrow0.$$

Positive scalar multiplication preserves all critical-point locations and types and every oriented gradient trajectory; it merely reparametrizes time. For sufficiently small epsilon, the chosen strict Hessian, speed, angle, and travel-time margins still hold. Branch hits, tube containment, and first-crossing locations are unchanged. But

$$\min_K|\nabla f_\epsilon|=1-\epsilon<\eta.$$

Therefore `f` belongs to the current `U_chi` and arbitrarily close `f_epsilon` do not. This proves that `U_chi` is not open in C2, even at a connection in the stated generic set. The example is a smooth finite trigonometric polynomial; it does not rely on a hidden degenerate critical point.

### Required local amendment and why coverage survives

Require the **strict** exclusion inequality

$$\min_{\overline{B_p^+\setminus N_p}}|\nabla f|>\eta$$

and the analogous condition at q, consistently in the definition and its uses in Sections 3, 7, and 8. This is a recommendation for the author, not an edit made by this reviewer.

With this definition, the displayed Lipschitz estimate gives a genuine neighborhood preserving the exclusion bound. For the critical point, the map `(x,f) -> grad f(x)` is C1 on local coordinates times the C2 Banach space, and its derivative in x is the invertible Hessian. The implicit-function theorem produces a unique continuing zero in a small ball compactly inside N. On the compact remainder of the closed outer box outside that ball, the original gradient has positive minimum, since the original zero is unique. A sufficiently small C1 perturbation creates no zero there. The continuing zero stays in N, while the index and the strict gap `>delta` persist by continuity of the Hessian and its eigenvalues. This supplies the missing openness argument for condition 1.

There is no need to choose a globally continuous signed eigenvector. In two dimensions, the positive and negative eigenspaces of an index-one Hessian are distinct one-dimensional spaces; their local continuation is enough. The declared local-section pairs determine which branch is tracked. This addresses sign selection locally and leaves the actual C1 invariant-manifold and hitting-map theorem as the existing A2 dependency.

For any genuine nondegenerate connection, sufficiently small endpoint outer boxes isolate their critical points. Rational boxes and rational continuation neighborhoods can be chosen with compact containment around those points even when the points themselves have irrational coordinates. The compact exclusion gradient minimum is positive; choose a rational eta **strictly smaller** than that minimum, and a rational delta strictly below the Hessian gap. Accordingly the stricter predicate does not lose the claimed connection coverage.

On a compact regular orbit segment, the geometric data used for coverage need robust interior section hits, positive tube-boundary clearance, and absence of an earlier section hit on the compact prefix outside a small transverse-crossing neighborhood. These are the geometric hypotheses on which the previous A2 first-crossing argument depends. Strict speed/angle/time bounds do not replace these geometric conditions. At the connection one can choose a narrow flow neighborhood and a small interior section, then rational approximations preserving their positive margins. No new A2 acceptance is asserted here: subsequent use of A2 must retain these strict geometric conditions, rather than read “inside” or a segment endpoint as allowing a boundary hit.

Finite tuples of rational boxes, finite rational polygon data, rational thresholds, and finite branch labels form a countable collection. Taking the union over all such charts requires no measurable first-chart selection. Countability and coverage therefore survive the local strict-threshold amendment.

## A6: conditional argument, including measurability

The current Section 7 begins by declaring the validity set open because `U_chi` is open. The counterexample above invalidates that premise. The correct verdict on the present bytes is consequently AMEND, even though the intended slicing method is sound under the repaired premise.

For precision, here is the complete conditional argument. Assume a genuinely open `U_chi`, the C1 map `D_chi` supplied by A2, the nonvanishing at charted connections supplied by A3–A4, and the countable directions and independent decomposition supplied by A5. Define

$$E_{\chi,j}=\{f\in U_\chi:D_\chi(f)=0,\ dD_\chi(f)[h_j]\ne0\}.$$

This is a Borel subset of X: the chart is open, and both displayed functions are continuous on it. For fixed j, let `nu_j` be the law of its residual `g_j` and `gamma` standard Gaussian measure on the scalar coordinate. The A5 input gives joint law `nu_j x gamma`. The map

$$T_j:X\times\mathbb R\longrightarrow X,\qquad T_j(g,\xi)=g+\xi h_j$$

is continuous. Hence `T_j^{-1}(E_chi,j)` is a Borel product-space event, providing the measurability needed for Tonelli/Fubini without any enumeration of zeros or interval components depending measurably on g.

For each fixed residual g, its validity set

$$I_{g,\chi}=\{\xi:g+\xi h_j\in U_\chi\}$$

is open in the real line and is a countable disjoint union of open intervals, since each component contains a distinct rational. On that open set,

$$F_g(\xi)=D_\chi(g+\xi h_j),\qquad
F_g'(\xi)=dD_\chi(g+\xi h_j)[h_j].$$

Each zero counted in `T_j^{-1}(E_chi,j)` has nonzero derivative and is isolated among the zeros by the one-dimensional inverse-function theorem. The set of such regular zeros is countable: give each an isolating interval from the countable rational basis. Accumulation at excluded boundary parameters or at nonregular zeros does not alter this conclusion. Thus each scalar section of this Borel event has standard Gaussian measure zero. Tonelli gives

$$\mu(E_{\chi,j})=
\int_X\int_{\mathbb R}{\bf1}_{E_{\chi,j}}(g+\xi h_j)
\,d\gamma(\xi)\,d\nu_j(g)=0.$$

The countable union over `(chi,j)` is a Borel null set containing `Connection intersect Omega_gen`. If the uncompleted connection event has not separately been shown Borel, the exact conclusion at this stage is outer measure zero, hence measurability and measure zero in the completed Gaussian probability space. This is sufficient for the almost-sure assertion after adding the assumed null complement of `Omega_gen`; it does not require an analytic-set projection or a measurable choice of chart.

The argument never needs every field on the shift line to be Morse. A degeneracy elsewhere on the torus that does not invalidate the selected local chart is harmless. Accordingly Section 8's language about degenerate parameters lying outside the interval should be read as **chart-invalidating** degeneracies, not a claim that every global degeneracy prevents membership in that chart.

## Dependency boundaries and validation limits

- A2–A5 retain only the predecessor review's dispositions. Sections 4–6 were independently checked byte-identical; their analytic results are dependencies here, not newly accepted results.
- An out-of-scope wording issue in Section 6 should not be read as an additional assumption of norm separability of `X*`. Its phrase “countable dense family ... in X*” is stronger than what its reason proves. The slicing argument needs a countable family of functionals whose **covariance images are dense in H**, using separability of H. That is the interpretation explicitly used by the prior A5 review and by the conditional argument above. This observation does not reopen or extend A5's mathematical verdict.
- I executed the unmodified negative-control script using `python -B -S research/q0/sard_g/chart_predicate_negative_control.py`; it exited 0 and printed `negative control passed`. It checks the predecessor's hidden-critical-point example and two finite endpoint-box cases, using sampled gradient minima. It does not check equality at the fixed exclusion threshold, arbitrary C2 perturbations, or the countable Gaussian measure argument. Its pass does not resolve the analytic counterexample above.
- The official [Cambridge page for Stroock, Chapter 8, “Gaussian Measures on a Banach Space”](https://www.cambridge.org/core/books/abs/probability-theory-an-analytic-view/gaussian-measures-on-a-banach-space/7728BE474E26B4DE50E936668EFC4C26) was inspected for bibliographic context. The full chapter was not accessible there and no theorem-level verification of its contents is claimed. This review does not newly validate the invariant-manifold references behind A2. The counterexample and the Borel/countability calculation are displayed above rather than delegated to a citation.
- The finite-jet/Bulinskaya/Kac–Rice input, C103 regional estimates, endpoint distribution calculation, positive-Fourier RKHS injection, A5 Gaussian decomposition, Theorem B, RN/JETMOD, and all 3D claims were not re-proved. Hosted engineering checks are not evidence of their mathematical truth.

The reviewed worktree remained clean at the exact head/tree above. Only the scoped pickup comment was published by this reviewer; this full verdict is prepared for separate review and public publication by the coordinating agent. No author proof, source registry, claim status, or branch was modified.
