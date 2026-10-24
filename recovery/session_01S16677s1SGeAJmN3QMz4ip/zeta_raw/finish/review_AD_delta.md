# PR #373 A/D successor review: scoped PASS

Dylan Roy — delegated AI review. Actual performer: OpenAI dot / Codex `/root/review_zeta_fold_mathematics`, the same nonauthor, source-exposed reader as [5486843735](https://github.com/d6g8k5htny-coder/main/pull/373#pullrequestreview-5486843735). This is a bounded changed-input source review, not blind reconstruction. Organizational-independence credit 0; Dylan's personal reading PENDING; scientific effect NONE.

**Verdict: PASS for the A/D amendment scope at `0969b3271b6fc39af57ac7a639e8e0a52ab92f3a`. ZF-AD-01–05 are addressed; no required A/D AMEND remains.** The earlier AMEND remains the correct historical verdict for `70f3d993`. This does not supply slice B, numerical reproduction slice C, hosted/integration verification, readiness, approval to merge or theorem-wide acceptance.

## Identity and bounded comparison

- Personal pickup: [6119775263](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6119775263); author handoff: [6119395474](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6119395474).
- Exact fix commit: `479d24504d3bc5fd3f519938b2093b218ddb5f7c`, whose parent is the reviewed `70f3d99302c994182b99098f6328e623667b599e`. The native comparison confirms only 11 files under `experiments/zeta_fold_exploration/` changed.
- Successor `0969b327` has parents `479d2450` and main `940e4855487bb85d32dde2d60f963eba4b49d686`. The comparison from the fix to the successor changes only `docs/RESEARCH_INDEX.md` and `governance/OP-WORKFLOW-20260930.md`: no packet change. Those base changes are not being mathematically reviewed here.
- Consumed blobs: README `b367aecd6a4559274657b79fc97df0b53332bd7e`; CLAIMS `c1d4931d918837ff7641eb7584d148d498c92aa9`; exact tests `fbfa9cca567a9754ca1e7eba630c86745c65f6be`. Manifest blob is `a3fd707f41a2e85ed68778bbb687b86e5ab1f697`; the manifest executable was not run.

## Finding-by-finding disposition

1. **ZF-AD-01 addressed.** README §4 item 3 and §6.3 now require a positive C1,1 weight, controlled fold asymmetry and uniform hypotheses for ensemble transfer. The general C1 claim is weakened to the justified `1+o(r)` local remainder; the exact `1+|x|^(1+eta)` counterexample is preserved. The weighted coefficient is correct: ell_w/(w(z0)ell) = 1 + [(12/5)(B/A)w'/w + (1/5)w''/w]tau + O(tau²) for the stated smooth fold setting. The extra linear-weight contribution from asymmetric H is included rather than asserted to vanish.

2. **ZF-AD-02 addressed.** In-brief, bottom-line table, §5.3 readings, §7, CLAIMS 5.4, PR body and amended plot labels now identify the specified Hardy-Z finite-window statistic. The unsupported “decisive” and total-error claims have been removed from the current conclusions. Bootstrap intervals remain explicitly sampling-only; controls are expectations under hypotheses, not certified total-error calibrators; fixed-N CUE's full bar law remains a target. Pair correlation, nearest gaps and persistence pairs are distinguished. The dependent-mark r⁴ example correctly shows why a finite marginal negative moment is not sufficient. No limiting exponent or xi/de Bruijn–Newman numerical conclusion is inferred.

3. **ZF-AD-03 addressed.** Proposition 1(c) now supplies an analytic domain containing a compact interval, a base point in that interval, a field and the superlevel convention `(d,b]`. The reflection H̃(t,z)=−H(t,2z0−z), F̃(t,z)=F(t,2z0−z) preserves the superlevel barcode for A<0 and correctly maps (A,B,C) to (−A,B,−C). The local elder argument and all three previously checked kappa coefficients are retained. The text separates the small-tau local bar from arbitrary close pairs and t=0 pairing; the headline opening law is asymptotic, not the false exact r²=8tau identity.

4. **ZF-AD-04 addressed.** README §1.2, CLAIMS 3.4 and the two velocity/check docstrings separate the unconditional simple-root identity from the all-real sum. RH at t=0 is distinguished from simplicity; multiplicities of other real zeros are retained. The truncated real-zero comparison is explicitly a consistency/truncation check and is not evidence that off-axis contributions vanish.

5. **ZF-AD-05 addressed in the main target.** Conjecture Z now fixes H(T)=sqrt(T), defines a per-bar amplitude even for nonadjacent pairs, identifies the median normalization and dropped endpoint/essential bars, and expressly distinguishes the new normalized-bar measure from the existing measurements. Existence of the limit over T is itself conjectural, followed by a separate ell→0 limit. The conclusion is cumulative scaling, not a density theorem; the atomic example correctly demonstrates the distinction. T-CUE now explicitly requires global selection/other-stratum control before identifying the full coefficient at beta=1.

6. **Positive-Lambda informational clarification preserved correctly.** CLAIMS 6.2 and README §§3,7 use the existing global positive-time bound to force a finite multiple real zero if Lambda>0. They do not infer a nondegenerate double zero, a finite collision at Lambda=0 or progress toward RH. The compactness/continuation argument matches the prior review. The statement that no compact bound is available at zero is read as the limitation of this estimate, which diverges as t→0.

## Exact-test semantics and independent algebra

The delta adds three exact tests. Static inspection verifies their intended coverage:

- The new weighted-series helper integrates wH between the existing formal root series, divides by the unweighted integral and w(z0), and tests the tau coefficient and odd-term cancellation for three folds and three polynomial weights. Negative A is included.
- The C1 counterexample substitutes a=u², x=v², integrates the resulting polynomial and checks both the `(4/15)a^(3/2)` excess and its ratio to r², `1/(15u)`.
- The dependent-mark test checks E[r^−1]=3/2 for density 3r² and the CDF scaling under ell=r⁴, yielding density exponent −1/4.

I independently used a small reviewer-written symbolic calculation, without importing project code, to recover the weighted tau coefficient, the general counterexample ratio `3a^(1+eta)/((2+eta)(4+eta))`, its eta=1/2 specialization and the −1/4 exponent. Those checks returned the stated expressions. They establish the algebra only. The finite test cases are not a proof of every analytic remainder or the ensemble hypotheses.

**No packet test, numerical pipeline, quadrature or retrieved project code was executed in this review.** The author-reported 20/14 passing tests are not my execution evidence. The changed numerical scripts contain explanatory/label changes only in the inspected diff; the SVG/output changes are corresponding text labels. This preserves recorded numerical inputs/results but does not reproduce them.

## Nonblocking clarity note and remaining exclusions

- README §6.2's optional coefficient paragraph still writes `mu((0,ell]) ~ c(T)ell at finite T`. As the corrected main statement itself explains, a finite empirical measure cannot have a positive ell→0 asymptotic below its shortest bar. Read that optional paragraph as a finite-tail fit on a declared range; writing `approximately` with the range would avoid ambiguity. It is expressly outside Conjecture Z and does not undo the corrected iterated limit.
- The Sturm/zero-number and boundary-flux explanation in README §5.2 was already present at `70f3d993`; the changed `dbn/count_check.py` docstring now mirrors it. This review does not newly certify its boundary hypotheses, first-collision completeness, recorded zero counts or numerical root enclosures. Those numerical/certification questions remain outside A/D and are not supplied by green repository CI.
- Citation-wide dispositions stay with slice B. Slice C remains unperformed here. Source custody stays with the author; this review takes no branch or integration custody.
