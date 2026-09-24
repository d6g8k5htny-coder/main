# Research guide

[Home](../README.md) · [Run the checks](REPRODUCE.md) · [Workspace](WORKSPACE.md) · [Open work](#open-work)

This is a reading map, not a mathematical acceptance register. Snapshot: 24 September 2026. Follow each source's exact hypotheses and linked discussion for later reviews or corrections. Current work in `Math-` and the legacy hardening stack are both visible; a missing historical carrier is not evidence that newer candidates do not exist.

## Gaussian persistence

Read this sequence to follow the argument rather than starting with a numerical constant.

| Step | Read | What it supplies; what it does not |
|---|---|---|
| 1. Separating geometry | [Marked-cylinder proof](https://drive.google.com/file/d/1BnPods7Lf-ECdD34noQihZcEfcqpy7R5/view) | A deterministic global elder-pairing criterion; geometry alone is not a Gaussian probability estimate |
| 2. Actual conditioned Gaussian law | [Matrix-cap and lifetime proof](https://drive.google.com/file/d/1foDgiDi4XIKOfbrZEE8dWKV_BkZU8LIb/view) · [review #63](https://github.com/d6g8k5htny-coder/main/issues/63) | Compact-mark cubic failure bound and unrestricted leading lifetime density; parent proof is unreviewed |
| 3. Leading coefficient | [SIDE24 d2/d3 enclosure](https://github.com/d6g8k5htny-coder/Math-/blob/main/coefficients/side24_v1/PROOF.md) · [review #65](https://github.com/d6g8k5htny-coder/main/issues/65) | Evaluates the exact parent expression; numerical precision is not parent acceptance |
| 4. Quantitative remainder | [Bounded unrestricted remainder](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md) · [review #67](https://github.com/d6g8k5htny-coder/main/issues/67) | Candidate O(1) absolute remainder; no numerical remainder constant or usable lifetime cutoff yet |

The main claims concern a specified normalized periodized Gaussian model on each fixed torus. The birth/gap/orientation scope, dimension, normalizer, and essential-versus-finite-bar convention must stay attached to the statement. Drive access may be required for the older standalone parents; the newer proof files above open directly on GitHub.

## RN counting

Start with the [probability-to-count interface](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md). It explains why a cubic event probability does not by itself give a cubic expected count and writes the actual three-determinant integral. It does not evaluate that integral.

Then consult the [legacy RN status](https://github.com/d6g8k5htny-coder/main/blob/chatgpt/drive-github-hardening-20260919/docs/math_status/STATUS_RN_UNIF.md) and [legacy execution guide](https://github.com/d6g8k5htny-coder/main/blob/chatgpt/drive-github-hardening-20260919/docs/RESEARCH_EXECUTION.md) for the specific numerical route. Its absent source objects, cell coverage and 24-jet requirements are not discharged by the newer cap proof. Do not apply the cap's local support inclusion to an arbitrary remote region.

## P15 combinatorics

| Read in order | Source | Important distinction |
|---|---|---|
| Local-cover interface | [Original P15-B](https://drive.google.com/file/d/19D-eHQAIXMGGy2ThZUfZ0GGjIKWm5C2j/view) | Actual local restrictions, complete crossing witnesses, and budgets at the same prices |
| Palette optimization | [Matroid specialization #59](https://github.com/d6g8k5htny-coder/main/issues/59) | Structural hypotheses are needed; not an arbitrary-support formula |
| Realized family | [Original-coordinate covers](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md) | 816 is the specified cover threshold; the whole ground set needs 818 |
| Failed extension | [Two-coordinate price boundary](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md) | The unchanged-palette extension to all transformed prices without extra hypotheses is false |
| Valid restricted successor | [Transformed-price budget](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/price_budget_20260924/PROOF.md) | Every demand at least 2 and every probability at most 1/4 restore the full transformed-price range at the same palette |

The last result does not retract the counterexample: it imposes additional hypotheses. Its checker can certify sufficient local budgets outside the simple uniform range, but a non-certified result is not a proof of impossibility. Use [review #67](https://github.com/d6g8k5htny-coder/main/issues/67) and the mathematical source PR for review.

## Open work

| Front | Next substantive target | Existing discussion |
|---|---|---|
| Gaussian proof review | Challenge the marked Kac–Rice/elder interface, matrix regression and all-mark cutoff estimates | [#63](https://github.com/d6g8k5htny-coder/main/issues/63), [#67](https://github.com/d6g8k5htny-coder/main/issues/67) |
| Quantitative Gaussian estimates | Evaluate a useful remainder constant and lifetime range; do not infer a second coefficient from boundedness | [#67](https://github.com/d6g8k5htny-coder/main/issues/67) |
| Coefficient audit | Independently check cone truncation, image/normalization comparison and outward arithmetic | [#65](https://github.com/d6g8k5htny-coder/main/issues/65) |
| RN and 24-jet route | Bound the actual count-weighted integral on its complete declared region and supply the missing numerical enclosures | [#61 RN tasks](https://github.com/d6g8k5htny-coder/main/issues/61), [#67](https://github.com/d6g8k5htny-coder/main/issues/67) |
| P15 scope and price range | Audit the realized family and restricted price theorem; investigate sharper conditions or additional palettes beyond the counterexample | [#59](https://github.com/d6g8k5htny-coder/main/issues/59), [#67](https://github.com/d6g8k5htny-coder/main/issues/67) |

These are existing work targets, not newly activated agents. Read current replies and claims before starting. An offered review is not an accepted assignment. [Campaign #61](https://github.com/d6g8k5htny-coder/main/issues/61) remains the collaboration entry and the hourly loop's handoff surface.

## Source lookup and history

The [exact-source catalog](https://github.com/d6g8k5htny-coder/meta-framework/blob/main/registry.json) and [query tool](https://github.com/d6g8k5htny-coder/query-) connect named keys to immutable commits and hashes. They track identity, not scientific acceptance or guaranteed currentness. The [workspace guide](WORKSPACE.md) separates default branches, legacy research branches, and Drive records. [2025 material](../history/2025/README.md) is history, not a current research result.
