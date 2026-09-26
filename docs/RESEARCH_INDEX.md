# Research guide

[Home](../README.md) · [Reproduce a result](REPRODUCE.md) · [Workspace and branches](WORKSPACE.md) · [Open work](#open-work)

This is a reading map, not a theorem-acceptance register. Read the statement, hypotheses, proof, and latest source-bound review before using a result. Merging a candidate or passing its tests is not independent mathematical acceptance.

## Start with the task

| Task | Destination |
|---|---|
| Read or extend mathematics | The topic sections below and [Math-](https://github.com/d6g8k5htny-coder/Math-) |
| Run a calculation | [Reproduction guide](REPRODUCE.md) |
| Review or claim work | [Open work](#open-work) and [downstream-first queue #86](https://github.com/d6g8k5htny-coder/main/issues/86) |
| Locate exact source bytes | [Catalog](https://github.com/d6g8k5htny-coder/meta-framework/blob/main/registry.json) and [query tool](https://github.com/d6g8k5htny-coder/query-) |
| Work on the larger numerical tree | [Hardening branch](https://github.com/d6g8k5htny-coder/main/tree/chatgpt/drive-github-hardening-20260919), not this default-home checkout |

## Gaussian persistence

| Reading order | Source | Scope and review |
|---|---|---|
| 1. Geometry | [Marked-cylinder proof](https://drive.google.com/file/d/1BnPods7Lf-ECdD34noQihZcEfcqpy7R5/view) | Deterministic sufficient criterion; not itself a Gaussian probability estimate |
| 2. Conditional field and leading law | [Matrix-cap and lifetime proof](https://drive.google.com/file/d/1foDgiDi4XIKOfbrZEE8dWKV_BkZU8LIb/view) | Exact periodized model; [review #63](https://github.com/d6g8k5htny-coder/main/issues/63) |
| 3. Coefficient | [SIDE24 dimensions 2 and 3](https://github.com/d6g8k5htny-coder/Math-/blob/main/coefficients/side24_v1/PROOF.md) | Evaluates the parent formula, without accepting that parent; [review #65](https://github.com/d6g8k5htny-coder/main/issues/65) |
| 4. Quantitative error | [Bounded unrestricted remainder](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md) | Author-side O(1) remainder; [review #67](https://github.com/d6g8k5htny-coder/main/issues/67) |

Keep the dimension, full determinant normalizer, mark restrictions and finite-versus-essential-bar convention attached to each result. Numerical constants and a useful lifetime range for the remainder are not supplied by a qualitative O(1) statement. Drive sources may require access.

## RN counting

Start with the [probability-to-count interface](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md). It supplies exact implications and counterexamples and identifies the three-determinant integral; it does not evaluate that integral. A remote critical point may exist despite correct local pairing.

Next read the [fixed-remote height-window estimate](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/remote_window_20260924/PROOF.md), published in [Math- PR #6](https://github.com/d6g8k5htny-coder/Math-/pull/6), with [review #76](https://github.com/d6g8k5htny-coder/main/issues/76). This bounds the actual three-determinant numerator by O(r^5) and the expected count by O(r^3) for points in the between-pin height window on a region a FIXED positive distance from the coalescing pins. It identifies the positive contact mean kernel and fixed-separation higher-witness counts. It does not cover a spatial exclusion shrinking with r, nearly coincident witnesses, all remote heights, or a numerical 24-jet certificate.

[Run the remote-window checks](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/remote_window_20260924/README.md) from the Math repository; exact catalog key `rn-fixed-remote-window` locates the published source. The proof rederives the needed full-pin count estimates without consuming global elder selection. Its author-side disposition and the intermediate-annulus boundary remain explicit.

A newer fixed-annulus successor now has a source-bound nonauthor technical review at its exact limited scope: [fixed d=2 scaled-annulus height-window theorem](https://github.com/d6g8k5htny-coder/Math-/blob/760340e921ac4ceda296b8118da936f1133e956e/frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md) with [review record](https://github.com/d6g8k5htny-coder/Math-/blob/760340e921ac4ceda296b8118da936f1133e956e/reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md). It covers fixed L, a fixed scaled annulus in d=2, compact positive gap marks, all frames, and the between-pin height window. It does **not** close pin neighborhoods, intermediate physical scales, witness collisions, kappa→0, d>=3, global RN/JETMOD, or elder selection.

An additive intermediate-scale note for one extra critical point at physical distance `s` with `r ≪ s ≪ 1` is [PROOF.md](../notes/intermediate_scale_20260926/PROOF.md). It records the majorant `C k r^3 |x|^{-6}`, sharpened to `C k r^3 |x|^{-4}` on the cone `|x·e_1| ≤ |x|/2`, with `C` independent of the location, the scale, and the frame. On that cone the conditional covariance determinant of `(∇f, f)` is of order `|x|^{12}`, so a spectral floor independent of `|x|` fails. The note is a derivation with algebraic jet guards. Pin neighborhoods, shrinking witness separations, a numerical constant, and a distance-independent multiple of `k r^3` stay open. The intended Math- path `frontiers/intermediate_scale_20260926/` is unpublished here: push by `cursor[bot]` was refused with HTTP 403.

The [legacy RN status](https://github.com/d6g8k5htny-coder/main/blob/chatgpt/drive-github-hardening-20260919/docs/math_status/STATUS_RN_UNIF.md) and [execution guide](https://github.com/d6g8k5htny-coder/main/blob/chatgpt/drive-github-hardening-20260919/docs/RESEARCH_EXECUTION.md) describe a separate numerical route. Its missing historical carriers and 24-jet obligations are not discharged by a newer probability proof, and do not erase newer author-side candidates.

## P15 combinatorics

| Reading order | Source | Distinction |
|---|---|---|
| Local-to-global interface | [Original P15-B](https://drive.google.com/file/d/19D-eHQAIXMGGy2ThZUfZ0GGjIKWm5C2j/view) | Actual restrictions, complete crossing witnesses, and compatible local budgets |
| Palette optimization | [Matroid specialization #59](https://github.com/d6g8k5htny-coder/main/issues/59) | Requires its structural hypotheses |
| Actual-coordinate family | [Realized covers](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md) | Specified cover threshold 816; whole ground set needs 818 |
| Failed extension | [Price-boundary counterexample](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md) | Same-palette unrestricted transformed-price extension without extra hypotheses is false |
| Restricted positive successor | [Transformed-price budget](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/price_budget_20260924/PROOF.md) | Demands at least 2, probabilities at most 1/4; [Math- PR #4](https://github.com/d6g8k5htny-coder/Math-/pull/4) |
| Full probability-range successor | [Sharp full-price budget](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/full_price_20260924/PROOF.md) | All independent probabilities, demands at least 2, same realized family and palette; [review #74](https://github.com/d6g8k5htny-coder/main/issues/74) |

The full-range successor removes the probability ceiling, not the demand or realized-family hypotheses. Its sharp uniform factor is 1/[3-log(3e-2)]<6/7. The earlier 16/27 factor remains better on its smaller domain, and demand-one counterexamples remain valid. An inconclusive sufficient-budget test is not an impossibility proof. Newer entries belong in this map only after their actual publication is verified.

## Open work

| Front | Concrete task | Discussion |
|---|---|---|
| Gaussian foundations | Review marked Kac–Rice, global elder selection and matrix regression | [#63](https://github.com/d6g8k5htny-coder/main/issues/63) |
| Quantitative Gaussian work | Evaluate usable error constants and lifetime cutoff | [#67](https://github.com/d6g8k5htny-coder/main/issues/67) |
| Coefficient | Review cone truncation, normalized images and outward arithmetic | [#65](https://github.com/d6g8k5htny-coder/main/issues/65) |
| RN / 24-jet | Bound the actual count-weighted integral on its complete declared region | [#67](https://github.com/d6g8k5htny-coder/main/issues/67) |
| RN fixed-remote successor | Review the extra-pin count proof, evaluate its kernel, and control the remaining shrinking-exclusion/collision regions | [#76](https://github.com/d6g8k5htny-coder/main/issues/76) |
| P15 | Review full-range hazard transfer and sharpness; extend beyond the realized-family hypotheses | [#74](https://github.com/d6g8k5htny-coder/main/issues/74), [#67](https://github.com/d6g8k5htny-coder/main/issues/67) |

Check current claims before writing and release completed work. Offered reviews are not accepted tasks or evidence of active agents. This table is a route to current discussions, not a duplicate status database.

## Repository map

| Repository | Responsibility |
|---|---|
| [main](https://github.com/d6g8k5htny-coder/main) | Home, campaign, reviews and integration |
| [Math-](https://github.com/d6g8k5htny-coder/Math-) | Proofs, calculations, outputs and mathematical tests |
| [meta-framework](https://github.com/d6g8k5htny-coder/meta-framework) | Curated exact-source identities and routing |
| [query-](https://github.com/d6g8k5htny-coder/query-) | Read-only lookup and local byte verification |
| [google-drive](https://github.com/d6g8k5htny-coder/google-drive) | Selected public replicas, not an automatic whole-Drive backup |
| [trial](https://github.com/d6g8k5htny-coder/trial) | Engineering and integration tests |
| [governance-](https://github.com/d6g8k5htny-coder/governance-) | Working practices, not theorem acceptance |
| [sandbox](https://github.com/d6g8k5htny-coder/sandbox) | Private experiments, excluded from automatic public exports |

## Branch placement

Default-main navigation and the hardening navigation port have different destinations. Retargeting a main-based documentation patch to the separate research tree changes its entire comparison. Review exact files and base before integration; neither navigation update moves the underlying research stack. [Historical 2025 material](../history/2025/README.md) remains history.