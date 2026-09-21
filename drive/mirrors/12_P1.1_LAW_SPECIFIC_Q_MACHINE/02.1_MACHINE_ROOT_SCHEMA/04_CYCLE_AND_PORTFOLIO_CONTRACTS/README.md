# `02.1_MACHINE_ROOT_SCHEMA/04_CYCLE_AND_PORTFOLIO_CONTRACTS`

Drive folder id `1mOfOsmvexVggGc4SuIITpESqpkxCAdnk`. The 2026-09-17 inventory gives this
folder **9 items** — 8 `application/json` and 1 `text/markdown` — and all 9 are held
byte-exact.

## What is held, and at what exactness

| stored file | exactness | bytes |
|---|---|---:|
| `Q0_C096_CONDITIONAL_RESIDUAL_CONTRACT.md` | byte-exact | 1,359 |
| `q0_c091_canonical_contract.json` | byte-exact | 3,690 |
| `q0_c093_release_contract.json` | byte-exact | 2,472 |
| `q0_c096_conditional_contract.json` | byte-exact | 18,033 |
| `q0_c097_gamma_contract.json` | byte-exact | 29,334 |
| `q0_c098_conditional_contract_v2.json` | byte-exact | 34,201 |
| `q0_c098_gamma_maxcount_contract_v2.json` | byte-exact | 25,885 |
| `q0_c101_limit_contract.json` | byte-exact | 27,607 |
| `q0_c108_portfolio_contract.json` | byte-exact | 27,331 |

## The status banners, verbatim

Each contract carries its own status in its own metadata. Transcribed, in file order:

`q0_c091_canonical_contract.json` records `UPPER_SHAPE_CERTIFICATE` as `OPEN`,
`UPPER_UNCERTAINTY_CALIBRATION` as `OPEN`, `H4_NUMERICAL_PATH_CERTIFICATE` as `OPEN`,
`LOWER_FINITE_084` and `LOWER_ASYMPTOTIC_08501` as `BLOCKED`, and kills one node outright:

> six-pin multivariate Gaussian corridor counterexamples

Its `forbidden_promotions` list names, among five entries,

> 0.8501 finite lower through r=.05

> 0.99 or 1.01 continuous upper before H4-PATH and U-SHAPE

`q0_c093_release_contract.json` records `status` as `CLOSED` for release
`Q0-C093-CLOSEOUT`, with

> external_review_is_separate

true and

> theorem_changed_by_c093

false.

`q0_c096_conditional_contract.json` records its status as

> PROVEN-MODULO explicit residual conditions

and the companion `Q0_C096_CONDITIONAL_RESIDUAL_CONTRACT.md` states the verdict in prose:

> The currently supportable selection theorem is conditional:

> This contract does not declare the three conditions proved. It proves the implication and prevents the conditions from disappearing through an intermediate `Proven-Modulo` node.

> The absolute station ceiling \(e^{-92}\) is not one of these conditions; it is evidence motivating a Gamma anti-concentration theorem, not a uniform cubic coefficient.

`q0_c097_gamma_contract.json` records

> PROVEN-MODULO seven explicit obligations

with `no_decimal_promotion` true.

`q0_c098_conditional_contract_v2.json` records

> PROVEN-MODULO three explicit residual conditions

and `q0_c098_gamma_maxcount_contract_v2.json` records

> PROVEN-MODULO collar and near count obligations

both with `exterior_Gamma_count` / `exterior_transfer` as `CLOSED-QUALITATIVELY` and
`no_decimal_promotion` true.

`q0_c101_limit_contract.json` records `status` as `CORE-CLOSED-INTERNAL-PROGRAM-GRADE`,
`R0_internal_status` as `PROGRAM-GRADE-CLOSED`, `R0_external_status` as
`SPECIALIST-REVIEW-PENDING`, and both numeric coefficients as `NOT-CLAIMED`.

`q0_c108_portfolio_contract.json` records `terminal_state` as `RELEASE-CLOSED` with
external dependencies

> independent SARD-G specialist acceptance

> real LLM endpoint, trace, latent/logit, token-accounting, and benchmark capabilities

and

> future_numerical_sharpening

set to

> new freeze required

## What this does not establish

These nine files are the sources' own contracts about their own claims, and every word
above is theirs. **Nothing here promotes, closes or discharges anything.** In particular:

* A contract recording itself as `CORE-CLOSED-INTERNAL-PROGRAM-GRADE` or `RELEASE-CLOSED`
  is recording an internal, program-grade disposition of its own author's line. The same
  files record the external status as pending specialist review, and this repository has
  no basis to treat the internal word as more than a transcription.
* `PROVEN-MODULO` is not proven. The named residual conditions are open in the sources
  themselves, and mirroring the contract discharges none of them.
* No numerical coefficient is adopted here. Two of these files say in their own metadata
  that no decimal may be promoted; this repository agrees by doing nothing with them.
* None of these contracts was checked, re-derived, or run against a kernel. Their internal
  `valid`, `promotable` and `reachable` fields are records of runs this repository did not
  perform.
