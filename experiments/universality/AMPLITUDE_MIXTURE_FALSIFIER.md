# Covariance-matched amplitude falsifier — executable engineering pilot

**Scientific effect: NONE.** This is a deterministic, standard-library-only *negative control* for a bad mathematical/architectural inference. It neither implements the random field's persistence algorithm nor modifies mathematical status, acceptance, graph nodes, theorem proofs, required checks, or external accounts.

## The mathematical source is pre-existing

- The original amplitude-mixture boundary appears in [Math- smooth Gaussian short-bar admission, section 11](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/gaussian_short_bar_theory_20261017/smooth_gaussian_manifold_admission/PROOF.md). Exact Math- commit: 9f3fb6918b28e6378c034ebc19e76d432f5391cb; Git blob d61fd11a826856baecb4cf4cd8a76ad5c25350dd.
- An existing [main fold-structural source, section 9](https://github.com/d6g8k5htny-coder/main/blob/be5a458ecb493bc54dd228377c6ce2000f9f21ce/experiments/universality/fold_structural_universality/PROOF.md) gives the explicit small-amplitude count lower-bound argument. Source commit be5a458ecb493bc54dd228377c6ce2000f9f21ce; Git blob 483d779cfdfdcea0bd8d0ec47c9fdaeff485fd4b.
- The broader analytic amplitude crossover and whole-barcode cluster research belongs to [Draft main PR #334](https://github.com/d6g8k5htny-coder/main/pull/334). This tool is one narrower, exact *metamorphic control*; no mathematical authorship, review, or integration is transferred from that draft.

The program declares the Math- Git source identity in its JSON output, but does not read or authenticate external Git content **at runtime**. A separately pinned source review is still necessary. No source text is executed, fetched from a URL, or silently treated as a theorem acceptance record.

## One exact family (not a new theorem)

Let H be a centered Gaussian field and U an independent uniform random variable on the open interval (0,1). Let k be an integer at least 2. Define the positive amplitude

\[
R = \sqrt{2k+1}\, U^k,\qquad F=RH.
\]

The program's moment identities, all exact, are

\[
\mathbb E[R^2] = 1,\qquad
\mathbb E[R^4] = \frac{(2k+1)^2}{4k+1} > 1.
\]

Independence yields identical covariance functions for H and F. But the fourth moment of a normalized Gaussian marginal changes from 3 to three times \(\mathbb E[R^4]\). Positive amplitude preserves each realization's critical-point and elder-pairing topology, while finite persistence lifetimes are multiplied by R.

For 0<s<sqrt(2k+1), the CDF is \(\Pr(R\le s)=(s/\sqrt{2k+1})^{1/k}\). Thus its lower-tail exponent is 1/k, at most 1/2, not the short-bar cumulative exponent 2/3 for the sourced Gaussian scenario. The negative moment \(\mathbb E[R^{-2/3}]\) diverges for k>=2.

An **external, unverified** premise must supply an interval 0<a<b and a finite positive lower bound p with \(\mathbb E[N_H([a,b])]\ge p\). This program does not certify that premise for any real H or run any barcode computation. With symbolic thresholds \(t_m=\sqrt{2k+1}\,b/m^k\) (integers m>=2),

\[
\mathbb E[N_F((0,t_m])]\ge p/m,\qquad
\frac{p/m}{(t_m/(\sqrt{2k+1}\,b))^{2/3}}
=p\,m^{2k/3-1}\longrightarrow\infty.
\]

Using the *dimensionless* threshold retains exact fractions; the fixed scale factor does not affect divergence against t^(2/3). This rejects a covariance-only transfer. It does **not** prove a complete alternate non-Gaussian density law.

## Runtime and negative-control contract

The CLI accepts one inert, bounded JSON specification or prints a synthetic example; no floating-point evidence is admitted. There are no third-party dependencies and no runtime remote source access or writes. Run from the repository root:

~~~sh
python -B -S experiments/universality/amplitude_mixture_falsifier.py --example
python -B -S experiments/universality/amplitude_mixture_falsifier.py --spec /path/to/spec.json
python -B -S experiments/universality/amplitude_mixture_falsifier.py --spec /path/to/spec.json --require-covariance-transfer
python -B -S -m unittest discover -s tests -p 'test_amplitude_mixture_falsifier.py' -v
python -B -O -S -m unittest discover -s tests -p 'test_amplitude_mixture_falsifier.py' -v
~~~

Synthetic example (not a measured bar intensity):

~~~json
{"schema_version":1,"power":2,"bar_interval":{"lower":"1/2","upper":"3/2"},"base_expected_bar_count_lower_bound":"1/4","grid":[2,3,5,9]}
~~~

A normal run returns one deterministic ASCII JSON line and exits 0. It reports exact rational moments, a conditional short-bar witness grid, an explicitly *unverified* source identity and external premise, a positive exponent gap, and the text REFUSED_BY_COUNTEREXAMPLE for covariance-only 2/3 transfer.

In the optional --require-covariance-transfer mode the same JSON is returned but exit status is 1: the attempt to infer a universal short-bar exponent from covariance alone is vetoed. Bad schema, unknown fields, duplicate JSON keys, unsupported numeric forms, invalid rationals, nonfinite values, oversized inputs, malformed grids and invalid intervals return exit 2 and an opaque error; caller input is not echoed.

The unit suite tests normal and optimized Python, deterministic output across interpreter modes and file/stdin routes, exact moment identities, sorted rational witness grids, malformed/floated/oversized input, caller-data nonleakage, the explicit inference veto, and a constant-amplitude negative mutation. Those test results are **engineering outcomes**, not standalone mathematical proof or a claim that field/barcode algorithms were exercised.

## Architectural incorporation boundary

The machine-readable witness can serve as a *falsification interlock* in a future source-aware dependency or hypothesis checker: if a proposed inference uses equal covariance as sufficient evidence for the Gaussian 2/3 short-bar law, the bounded counterexample requires rejecting that inference. It cannot infer the truth of unrelated claims from the witness and cannot automatically create a graph relation, source-review grade, or scientific status.

The actual contact, selection, common envelope, near/far population and other hypotheses of a particular field require their own reviewed proofs, exact source identities, and applicable execution/hosted controls. A green unit test cannot discharge them. The sourced amplitude-crossover theorem remains owned by its original authors and reviewers.

[Owner mandate](https://github.com/d6g8k5htny-coder/main/blob/b665307520948308a53d190d7341bf301df3b3a2/governance/OWNER_MANDATE_20261015.md) · [Delegation](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OP-AUTONOMY-20260923-v2.1.md).
