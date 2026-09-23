# `02.3_VALIDATION_RECEIPTS/05_CONTRACT_REPORTS_AND_VALIDATION_OUTPUTS`

Drive folder id `144yogjwKZY5OZ9AvHsVK8Gt3Vp1Ailqf`. The 2026-09-17 inventory gives this
folder **18 items** — 17 `application/json` and 1 `text/markdown` — and all 18 are held
byte-exact. It is the largest folder in the lane by item count.

## What is held, and at what exactness

| stored file | exactness | bytes |
|---|---|---:|
| `C095_BUNDLE_VERIFICATION.json` | byte-exact | 260 |
| `C095_VERIFIER_V5_REPORT.md` | byte-exact | 1,527 |
| `C096_BUNDLE_VERIFICATION.json` | byte-exact | 496 |
| `C101_BUNDLE_VERIFICATION.json` | byte-exact | 477 |
| `C108_PACKAGE_VERIFICATION.json` | byte-exact | 348 |
| `Q0_C093_RELEASE_VERIFICATION.json` | byte-exact | 459 |
| `gate_kernel_v2_0_validation.json` | byte-exact | 367,531 |
| `gaussian_corridor_certificate_report.json` | byte-exact | 13,327 |
| `q0_c091_contract_check_report.json` | byte-exact | 747 |
| `q0_c091_gate_reduction_report.json` | byte-exact | 4,460 |
| `q0_c096_conditional_contract_report.json` | byte-exact | 30,671 |
| `q0_c097_gamma_contract_report.json` | byte-exact | 50,679 |
| `q0_c098_conditional_contract_v2_report.json` | byte-exact | 56,226 |
| `q0_c098_gamma_maxcount_contract_v2_report.json` | byte-exact | 42,106 |
| `q0_c101_limit_contract_report.json` | byte-exact | 45,063 |
| `q0_c108_portfolio_contract_report.json` | byte-exact | 45,401 |
| `q0_llm_verifier_v2_validation.json` | byte-exact | 589 |
| `q0_llm_verifier_v5_validation.json` | byte-exact | 106,547 |

## The status banners, verbatim

The four contract reports for C096, C097 and C098 each record `shell_valid` and
`conditional_promotable` true with `unconditional_promotable` **false**; the reports for
C101 and C108 record all three true. That difference is the source's, and it is the whole
substance of those files.

`C095_VERIFIER_V5_REPORT.md` opens

> # C095 Verifier v5 Validation Report

with 22 cases, 22 passed, 0 failed, and records among its key adjudications

> the current Q0 upper contract is blocked by `ASSEMBLY` and `UB_G_RESIDUAL_UNIFORM`;

> 4.3 fails and 4.35 passes the base UB-G arithmetic regression;

> the finite lower 0.8411 assembly fails after explicit losses;

> a Proven parent cannot silently depend on a Measured object.

`q0_c091_gate_reduction_report.json` records its gate status in the source's own words:

> closed by separate matrix-transfer certificate

> closed structurally; numerical path certificate open

with `G_U_SHAPE` and `G_BONF_MARKED_REPULSION` open and `H4_MARKOV_PRODUCT` killed. Its
uncertainty block records

> CLOSED: an upper theorem consumes +/-0.02 as +0.02 unless inputs are proved already one-sided.

beside

> OPEN: confidence level, multiplicity, sample size, and source of 0.02 not recovered.

`q0_c091_contract_check_report.json` records all eleven of its checks true and then states,
in the same object,

> "live_decimal_theorem_promoted": false

`gaussian_corridor_certificate_report.json` records its adjudication as

> REJECTED AS A GENERAL UPPER BOUND

for the naive Markov product, and immediately bounds the evidence:

> The numerical orthant values are diagnostics, not interval proofs. The invalid-product conclusion is robust by factors well above numerical CDF error in the listed counterexamples.

## What this does not establish

Every file here is an **output**: a record of a run performed elsewhere by the program that
wrote it. None was reproduced here, and no number in any of them was recomputed. A JSON
object with `"valid": true` inside it is that program's report about its own inputs; it is
not a certificate, and this repository does not treat it as one. High precision is not
certification: the corridor report says so about its own numbers in the line quoted above,
and the same applies to every float in this folder. `shell_valid` is explicitly not the same
as promotable, and the six contract reports say which is which.
