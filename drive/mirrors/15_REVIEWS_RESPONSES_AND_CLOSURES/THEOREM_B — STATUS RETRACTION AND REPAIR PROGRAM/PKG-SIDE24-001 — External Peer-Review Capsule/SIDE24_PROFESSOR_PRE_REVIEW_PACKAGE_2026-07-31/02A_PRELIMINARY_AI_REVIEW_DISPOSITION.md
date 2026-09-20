---
title: "Disposition of preliminary AI reviews and revision record"
author: "Dylan Roy"
date: "Side-24 pre-peer-review revision: 31 July 2026"
---

# 1. Purpose

Four preliminary AI-generated reports or contributions were supplied before this packet was sent for independent human review. They were treated as adversarial issue lists, not as expert opinions or evidence that the candidate theorem is correct. This document records which criticisms were accepted, which were only partly correct, which were rejected, and what changed in response.

The central outcome is deliberately conservative:

> No counterexample to the local cubic-fold calculation or the resulting \(\ell^{-1/3}\) all-typed contact law was identified. The packet nevertheless did overstate the closure of the global elder-selection and dynamical steps. The revised manuscript is therefore a proof-oriented theorem candidate, not a claimed completed theorem.

# 2. Changes made in the revised packet

The following changes are incorporated in the controlling review manuscript and appendix.

1. **Status corrected.** “Closed proof chain,” “we prove,” and “this completes the proof” language was removed. Candidate Theorem 2.1 now has an explicit verification-status paragraph.
2. **Analytic sample paths clarified.** The complex Fourier series is controlled by the nonnegative random sums \(S_R\). Tonelli, a countable intersection over integral strip widths, and the Weierstrass M-test give the asserted entire periodic extension.
3. **Candidate Borel elder construction inserted.** Lemma 1.2a constructs superlevel connectivity through a countable basis and characterizes the elder event by countably many connectivity tests immediately above and below the saddle level. It is submitted for human checking.
4. **Absolute-continuity proof revised.** Proposition 1.2 now argues for absolute continuity of the selected measure by domination by the all-typed two-point Kac--Rice measure on separated configuration sets. The Borel mark remains available for the marked formula.
5. **Pointwise density convention fixed.** The manuscript specifies the near-plus-far Kac--Rice representative of the almost-everywhere Radon--Nikodym derivative and repeats that this is a per-volume first-moment intensity, not a probability density for a sampled bar.
6. **Corrected-pin determinant exposed.** The raw-to-corrected derivative matrix is displayed in block-triangular form. The tangential correction is an off-diagonal shear; the exact determinant is \(r^{-3}r^{-3}=r^{-6}\).
7. **Palm law and normalizer defined.** The unselected determinant-weighted Palm law, \(Z_r\), and \(p_r\) are defined in one place. The limiting Gaussian cone argument gives local positivity of \(Z_r/r^2\).
8. **Exact near-density formula inserted.** Lemma 3.3 contains the test-function identity, the pointwise lifetime representative, and the rescaled form. The pair polar factor and gap Jacobian combine to \(r^2(r^3/6)=r^5/6\).
9. **Uniform domination inserted.** Lemma 3.5a isolates the Gaussian-regression majorant and explicitly checks integrability of \(\kappa^{-2/3}\) at zero.
10. **Genuine far pairs separated from remote witnesses.** Lemma 3.5b states and argues that actual elder pairs with separation at least \(\rho\) have a lifetime density bounded uniformly near zero, so they are \(o(\ell^{-1/3})\). This new argument is submitted for review.
11. **Selection step demoted to an open proposition.** Proposition 3.4 states exactly what is still needed. Lemmas 3.4a--3.4c now supply the component-based event inclusion, determinant-weighted dynamics transfer, and fixed-distance witness estimate. The collar and singular-near triple-contact bounds remain open.
12. **Terminology and conventions corrected.** The three-dimensional flat comparison coefficient is called “Euclidean (nonperiodized),” not “planar.” The \(2\times2\) GOE variance convention is explicit.
13. **Fourier equality corrected.** The review appendix now includes \((2\pi)^{3/2}/(24^3Z_{24})\) in both the periodized Fourier series and the variance identity.
14. **Appendix provenance corrected.** Because this appendix contains review edits, it is no longer described as byte-exact. Baseline source hashes remain visible.
15. **Computational transcript repaired.** The coefficient-map script prints the relative Hessian bound and the \(5\times10^{-185}\) remainder actually used. The two nonstandard success markers are normalized, optimized Python execution is rejected, and clean-environment logs are bundled.
16. **Component-based failure exhaustion proved.** Lemma 3.4a replaces the informal branch-based case split by a deterministic merge-tree argument. It includes adjacency failure explicitly, distinguishes earlier death, same-component attachment, and a younger other component, and partitions the canonical witness by fixed collar/near/far regions.
17. **Euclidean cone moment derived in prose.** Section 3.7 now evaluates \(D_2=29/6-\sqrt6\) by the independent trace/Rayleigh decomposition of the GOE-plus-scalar law. The final constant no longer depends on a script-only assertion.
18. **Deterministic dynamics consolidated.** Appendix Module J now reproduces the inward tapered-tube flux, monotonicity, and convergence proof in the current energy-adapted module. No superseded file is needed to audit the deterministic capture/escape theorem.
19. **Module E hypotheses restored.** The sixth-jet and local \(C^8\) hypotheses omitted from the earlier extraction are now present. The separate \(10^{40}\) blown-up-chart derivative envelope remains explicitly unproved rather than being treated as script-certified.
20. **False sharp \(C^8\) number corrected.** The inherited claim \(3.1\times10^{-109}\) is contradicted by the single term \(\operatorname{He}_8(23)e^{-264.5}=9.9921\ldots\times10^{-105}\). A corrected shell bound \(\|K_{24}-K_0\|_{C^8(B(0,1))}<10^{-103}\) still proves the required \(10^{-100}\) hypothesis. The transfer ledger and expected output now record this correction.
21. **Probabilistic dynamics transfer written out.** Manuscript Lemma 3.4b combines the exact endpoint \(r^2\) weight, a transverse-eigenvalue coarea estimate, conditional Gaussian tails, Borell--TIS, and the Palm normalizer to bound the matrix-theorem exceptional set by \(O(r^3)\). It also states precisely what the inward and outward branches imply for the component cases in Lemma 3.4a.
22. **Fixed-distance witness bound promoted to a lemma.** Manuscript Lemma 3.4c uses full Fourier support, compact separated nondegeneracy, the exact endpoint soft factor, and the \(\kappa r^3/6\) value window to obtain the required \(O(r^3)\) Palm bound.
23. **First-image polynomial derived in prose.** Manuscript equations (3.31a)--(3.31f) define \(a,m_4,\chi,\Omega\), compute the six-image moment variations, and apply the exact logarithmic differential to obtain \(P_3(L)\). The periodization script is now an independent expansion check rather than the sole displayed derivation.
24. **Module K notation closed.** The isotropic spectral moments \(a,m_4,\chi,\beta,\Omega\) and the normalized matrix \(M_n=G_n+\sqrt{2/3}ZI_n\) are now defined before the generic recombination, including the GOE variance convention.

# 3. Issues accepted as genuine and still open

These are the load-bearing obligations on which a human reviewer should concentrate.

## 3.1 Remaining elder-selection probability bounds

The desired estimate is

\[
1-p_r(t,b,\kappa)=O(r^3).
\]

Lemma 3.4a proves the deterministic component-based exhaustion. Lemma 3.4b bounds adjacency failure, a live same-component attachment, and the younger-other-component case outside an \(O(r^3)\) Palm event. Lemma 3.4c controls fixed-distance witnesses. The unresolved quantitative work is now confined to the collar and singular-near earlier-death witnesses. The topological classification and the other Palm reductions are no longer being left as an informal case split, but all newly written lemmas still require human checking.

## 3.2 Triple-contact bounds

The symbolic ledger exposes the limiting collision factor, but a proof still has to connect the symbolic chart to every geometric regime, preserve the vanishing order under the side-24 perturbation, control all angular singularities, and carry out the regional integrals uniformly.

## 3.3 Probabilistic transfer of deterministic capture and escape: supplied for review

The energy-adapted scaling removes a hidden dependence on an upper eigenvalue bound, the rational margins check out, and Module J contains the inward and outward deterministic proofs in one current text. Manuscript Lemma 3.4b now derives, in the present notation, the normalized conditioned \(C^1\)-approximation reduction and the determinant-weighted Palm probabilities of the shallow-eigenvalue, large finite-jet, and fifth-derivative exceptional events. This is a new proof block submitted for independent checking, not an externally validated closure verdict.

## 3.4 Quantitative structural transfer

Appendix Module E now includes the previously omitted sixth-jet and local \(C^8\) hypotheses, corrects the false inherited sharp \(C^8\) number, and proves contact-face positivity directly from full Fourier support and compactness. It still invokes a \(10^{40}\) derivative envelope for normalized blown-up chart coefficients without deriving it; the packaged stability script explicitly labels that number as assumed. Contact-face positivity does not replace the missing joint finite-\(r\) blow-up argument. That argument, or a proved normalized-coefficient envelope on every boundary face, is required before the side-24 selection transfer can be called complete.

## 3.5 Literature and independent coefficient checking

The manuscript now reproduces both the GOE/cone scalar and the nearest-image first-variation polynomial analytically. Their conventions and derivative identities still require independent checking. Novelty also requires a systematic MathSciNet, zbMATH, and citation-network search; a targeted web search is not conclusive.

# 4. Criticisms accepted only in part

| Preliminary criticism | Disposition | Reason and revision |
|---|---|---|
| Genericity proof lacks all detail | Partly accepted | Uniform eigenfloors and derivative-distribution independence were already present. The strip convergence argument was made fully countable and explicit. |
| Marked Kac--Rice is unjustified | Accepted as a presentation gap | A Borel mark was inserted. Proposition 1.2 also has an unmarked domination proof, so its absolute-continuity conclusion does not depend on a sophisticated marked theorem. |
| Palm normalizer may degenerate | Partly accepted | The revised text defines \(Z_r\), gives the contact cone positivity argument, and states the local uniform two-sided bound. Independent checking is still invited. |
| The near pushforward is only verbal | Accepted | The complete test-function and density formulas are now Lemma 3.3. |
| \(\kappa\downarrow0\) domination is asserted | Accepted | Lemma 3.5a now separates the all-mark bound from the integrable \(\kappa^{-2/3}\) weight. |
| Genuine distant elder pairs are not bounded | Accepted | Lemma 3.5b is now a separate off-diagonal result. It is distinct from a remote third-point witness conditioned on a near pair. |
| GOE normalization is ambiguous | Accepted | The diagonal and off-diagonal variances are stated explicitly. |
| Side-24 transcript is inconsistent | Accepted | The source, expected output, runner, environment record, and execution logs are synchronized in this revision. |
| The technical appendix is submission-ready | Rejected | It is now expressly labeled an audit appendix with local equation numbers and open modules. |

# 5. Criticisms not adopted

Several preliminary comments would have added false requirements or changed a correct formula.

## 5.1 No Kolmogorov three-series or Fernique theorem is needed

For each integral strip width \(R\), the relevant random majorant is nonnegative and has finite expectation. Tonelli gives finiteness almost surely. A countable intersection over \(R\in\mathbb N\) and the M-test then give local uniform convergence on every compact strip. This closes the step without a three-series theorem.

## 5.2 Absolute continuity does not require a uniform near-diagonal count

If each restricted selected measure \(\mu_n\) is absolutely continuous and \(\mu_n\uparrow\mu\), then every Lebesgue-null set has \(\mu(A)=\lim_n\mu_n(A)=0\). A uniform bound as the spatial cutoff tends to zero is not required for Proposition 1.2. A separate far-pair argument is needed for the asymptotic and has been inserted for review for that reason.

## 5.3 The global maximum creates no finite bar requiring a density estimate

The global maximum corresponds to the essential \(H_0\) class and is excluded in the definition of the finite lifetime measure. No separate \(O(|A|)\) estimate for a “global-maximum bar” is needed.

## 5.4 The far-pair argument does not use the short logarithm chart

The midpoint/directed-sphere chart is used only for \(d(M,S)<\rho<12\). Lemma 3.5b works directly on the compact set \(\{d(x,y)\ge\rho\}\), so cut-locus ambiguity does not enter.

## 5.5 The near formula must contain \(r^5/6\)

One preliminary formula omitted the value-gap Jacobian. The retained identity has \(r^2\) from displacement polar coordinates and \(r^3/6\) from \(h=b-\kappa r^3/6\), hence \(r^5/6\) before the Kac--Rice factor.

## 5.6 The negative-definite cone need not have probability tending to one

For positivity and normalization, the limiting cone event needs positive Gaussian probability and a boundary of Gaussian measure zero. Probability tending to one is neither expected nor required.

## 5.7 Birth-height collapse does not require independence of \(q_t\) and \(f\)

Equation (3.20) is the law of total expectation under disintegration, justified by Tonelli for the nonnegative integrand. Correlation between the transverse Hessian and the field value is integrated out.

## 5.8 Script success is not mathematical proof

Every pass marker certifies only the assertions encoded in that program under the recorded environment. It cannot establish the applicability of Kac--Rice, topological exhaustiveness, conditional-flow estimates, or correctness of the encoded geometric model.

# 6. Literature claims in the preliminary reports

The first report attached the same eleven references to many unrelated paragraphs. Those citations should not be treated as claim-level verification.

- Chazal and Divol's paper concerns densities of expected persistence diagrams for broad point-cloud filtrations such as Čech and Rips constructions. It is useful background, but its abstract does not directly establish the elder-paired smooth Gaussian Morse-field density used here: <https://arxiv.org/abs/1802.10457>.
- The Bargmann--Fock percolation paper concerns the planar excursion-set threshold. It supplies context for the field, not the present short-lifetime selected persistence asymptotic: <https://arxiv.org/abs/1711.05012>.
- Gass and Stecconi prove finiteness of moments for numbers of critical points under suitable hypotheses. That is relevant to Kac--Rice integrability, not a proof of this persistence law: <https://arxiv.org/abs/2305.17586>.

No report supplied a source proving the same fixed-side, three-dimensional, elder-selected near-diagonal asymptotic. That is evidence that the question may be novel, not proof of novelty.

# 7. Claim-by-claim review crosswalk

| Interface raised by the preliminary reports | Revised location | Current status |
|---|---|---|
| Entire analytic sample path | Manuscript Proposition 1.1 | Clarified; submitted for review |
| Morse and distinct critical values | Manuscript Proposition 1.1 | Argument present; specialist citation check requested |
| Borel elder mark | Manuscript Lemma 1.2a | Newly inserted |
| Absolute continuity | Manuscript Proposition 1.2 | Repaired by all-typed domination |
| Pointwise density representative | After Proposition 1.2 | Explicitly fixed |
| Ordered pair and no \(1/2\) | Manuscript Section 3.1; Appendix L | Present |
| Corrected-pin \(r^{-6}\) determinant | Manuscript Section 3.2; Appendix A | Block structure inserted |
| Palm normalizer \(Z_r\asymp r^2\) | Manuscript Section 3.2 | Newly isolated |
| Exact near pushforward | Manuscript Lemma 3.3 | Newly inserted |
| Uniform all-mark majorant | Manuscript Lemma 3.5a | Newly inserted |
| Elder-failure topological exhaustion | Manuscript Lemma 3.4a | Proved in the manuscript; human check requested |
| Elder-selection \(1-p_r=O(r^3)\) | Manuscript Proposition 3.4 | Open at the regional Palm bounds |
| Genuine off-diagonal pairs | Manuscript Lemma 3.5b | Newly inserted |
| Triple-contact envelope | Appendix B, E, I; symbolic annex | Incomplete |
| Capture/escape | Appendix J; Manuscript Lemma 3.4b | Deterministic proof and probabilistic transfer present; independent checking requested |
| GOE convention and coefficient | Manuscript Sections 3.6--3.7 | Convention and full analytic calculation present |
| Side-24 transfer and remainder | Manuscript Section 3.8; Appendix D, E, G, H | First variation derived and arithmetic corroborated; Module E finite-\(r\) envelope open |
| Fourier prefactor | Appendix F | Corrected |
| Reproducible scripts | Reproduction guide and execution evidence | Clean run recorded; arithmetic only |

# 8. Recommended interpretation by the professor

The strongest current result is the local analytic architecture: corrected colliding pins, two soft determinants, cubic height gap, exact lifetime pushforward, and the resulting all-typed \(\ell^{-1/3}\) scale with an explicit candidate coefficient. The newly inserted measure-theoretic and off-diagonal arguments address several avoidable presentation gaps, subject to human verification.

The project should not yet be endorsed as a proved persistence theorem. The decisive question is whether the new component/dynamics reduction is valid and whether Proposition 3.4 can be completed with uniform finite-\(r\) collar and singular-near triple-contact bounds. A negative answer there would affect the selected persistence claim while leaving much of the all-typed critical-pair asymptotic intact.

The most useful human report would therefore identify the earliest failure, if any, in this order:

1. Borel elder mark and use of marked Kac--Rice;
2. component-based selection exhaustion and the regional Palm union bound;
3. triple-contact regional estimates;
4. side-24 quantitative chart transfer;
5. coefficient and literature checks.

# 9. Disposition of the DeepSeek proposed insertions

The later DeepSeek response supplied six blocks described as mathematical
solutions. Each was checked against the current manuscript, the exact Gaussian
conventions, and the packaged ledgers. They were not inserted verbatim.

## 9.1 Proposed elder-selection trichotomy: repaired, not accepted verbatim

The proposed statement incorrectly said that the regional witness events are
mutually exclusive. Several third critical points can occur simultaneously,
and a self-attachment can coexist with unrelated witnesses. It also defined
self-attachment too narrowly through both flow branches ending at \(M\), used
\(r^\alpha\) as a purported fixed-distance cutoff, and omitted adjacency
failure from the full event inclusion.

The useful core was retained in corrected form as Lemma 3.4a. The current
lemma is component based, does not assume Morse--Smale flow, includes the case
where the other component has a younger representative maximum, uses a fixed
distance \(\delta_0\), and states a union inclusion rather than false mutual
exclusivity of witness events.

## 9.2 Proposed off-diagonal lemma: accepted but already present

The Gaussian-regression proof of
\(\sup_{0<\ell\le\ell_0}\nu_\rho^{\mathrm{far}}(\ell)<\infty\) is correct. It is
already Manuscript Lemma 3.5b, with the endpoint type and elder marks and the
distinction between genuine far elder pairs and remote witnesses made explicit.
No duplicate lemma was added.

## 9.3 Proposed GOE integration: rejected; final constant independently derived

The response gives the correct target value \(29/6-\sqrt6\), but its displayed
integration is false. For \(M=ac>0\), integrating the off-diagonal
\(N(0,1)\) entry requires
\[
\frac{2}{\sqrt{2\pi}}\int_0^{\sqrt M}
(M-y^2)^2e^{-y^2/2}\,dy,
\]
not an integral to infinity, and the truncated Gaussian integral does not
collapse to the polynomial claimed in the response. The response also failed
to impose the trace condition correctly.

Section 3.7 now contains a different exact proof: the trace coordinate is
\(N(0,5/3)\), the traceless radius is independent Rayleigh, and direct
integration gives \(D_2=29/6-\sqrt6\).

## 9.4 Proposed triple-contact covariance derivation: partly correct, incomplete

The final Euclidean vanishing powers are correct, but the response incorrectly
states that the collar transverse covariance is \(\rho^2I_2\). The exact block
is
\[
\rho^2I_2+vv^\top,
\qquad \det(\rho^2I_2+vv^\top)=2\rho^4.
\]
Together with the axial variance this gives
\(\rho^6(4X^2+\rho^2)\). The singular-near determinant
\(\rho^{10}(3c^2+\rho^2)/24\) on the unit sphere is correct and is already
displayed in Appendix Module B.

These contact covariance identities do not establish the triple-Hessian
numerator envelope, the complete angular integrations, finite-\(r\) uniformity,
or the side-24 transfer. Proposition 3.4 therefore remains open.

## 9.5 Proposed finite-jet measurability proof: rejected

Global superlevel connectivity and elder pairing are not determined by a fixed
finite collection of jets at the critical points, and the random field is not
a fixed finite-dimensional semialgebraic family. Critical points can be labeled
continuously only locally in function space. The proposed reduction to finitely
many polynomial inequalities is therefore invalid.

The countable-basis construction in Lemma 1.2a is retained. It expresses
superlevel connectivity by finite chains drawn from a countable basis and then
uses countably many rational level tests, which is the appropriate Borel
argument.

## 9.6 Proposed energy-adapted dynamics: repaired substantially; global closure claim rejected

The cone and strip fractions agree with the current energy ledger. The response
nevertheless contains the literal error \(P(M)=1/4\); for the stated
\(p_0\), \(P(M)=0\). The valid comparison is
\[
P(5/4,Y)>65/256>1/4>0=P(M).
\]
Moreover, the proposed paragraph did not supply the inward tapered-tube proof
or the conditioned fifth-derivative and shallow-eigenvalue probability bounds.

Appendix Module J now reproduces the inward flux and convergence proof and
combines it with the energy-adapted outward argument. Manuscript Lemma 3.4b
then supplies a separate determinant-weighted Palm argument for the shallow
eigenvalue, finite-jet, and fifth-derivative exceptional sets and states the
resulting component consequences. That proof is not present in the DeepSeek
block and must be checked independently. Even if it is correct, it does not
close the collar and singular-near triple-contact estimates required by the
full elder-selection proposition.

**Net effect.** DeepSeek’s response prompted the formal component-based failure
lemma, consolidation of the deterministic dynamics, a new Palm transfer
argument, and the full GOE derivation. Its own proposed proofs were not
accepted without correction, and its claim that all six blocks would make the
proof peer-review ready is not supported.
