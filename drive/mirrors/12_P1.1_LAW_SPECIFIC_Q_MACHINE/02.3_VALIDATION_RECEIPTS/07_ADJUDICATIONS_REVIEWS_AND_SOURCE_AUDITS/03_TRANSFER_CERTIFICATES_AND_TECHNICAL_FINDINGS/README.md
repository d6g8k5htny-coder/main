# `…/07_ADJUDICATIONS_REVIEWS_AND_SOURCE_AUDITS/03_TRANSFER_CERTIFICATES_AND_TECHNICAL_FINDINGS`

Drive folder id `1OZvjw46u3se1A5nAX7XvIERtqRt_ezOK`. The 2026-09-17 inventory gives this
folder **6 items**, all `application/json`, and all 6 are held byte-exact.

## What is held, and at what exactness

| stored file | exactness | bytes |
|---|---|---:|
| `C095_SIX_PIN_COVARIANCE_FACTORIZATION.json` | byte-exact | 4,487 |
| `C097_COLLAR_GAMMA_BUDGET.json` | byte-exact | 2,687 |
| `C097_GAMMA_DENSITY_REDUCTION.json` | byte-exact | 4,032 |
| `C098_GAMMA_EXTERIOR_TRANSFER.json` | byte-exact | 3,355 |
| `C099_SCALED_FRAME_CERTIFICATE.json` | byte-exact | 3,547 |
| `periodized_bf_matrix_transfer_report.json` | byte-exact | 2,566 |

## The status banners, verbatim

`C095_SIX_PIN_COVARIANCE_FACTORIZATION.json` records its grade as `DERIVED-EXACT` and
separates the algebra from the numerics in its own gate-effect block:

> The algebraic proof is exact and exempt from agreement-based promotion. The full-6x6 numerical comparison is diagnostic.

`C097_GAMMA_DENSITY_REDUCTION.json` records grade `DERIVED-EXACT-REDUCTION`, a
seven-obligation set with statuses `OPEN` and `OPEN-PARTIAL`, and a `not_claimed` list:

> "The seven obligations are not discharged by this reduction.",

> "The station exponent 91.6 is not a uniform density theorem.",

> "No decimal upper coefficient is promoted."

`C097_COLLAR_GAMMA_BUDGET.json` records grade `DERIVED-EXACT` and a decision rule:

> "decision_rule": "Do not optimize toward 4.35 before M_Gamma and C_nd are certified. First certify the residual coefficients; then round the complete assembly upward."

with the warning

> "rounded_input_warning": "Using the displayed rounded base inputs (0.66+2.82)/0.80=4.35 leaves zero residual budget. Every positive Gamma or collar term then forces a coefficient strictly above 4.35."

`C098_GAMMA_EXTERIOR_TRANSFER.json` records grade `DERIVED-EXACT-EXISTENCE-THEOREM`, a
terminal status of `CLOSED-QUALITATIVELY` for the exterior cubic beside `OPEN` for the
explicit decimal, and a `not_claimed` list:

> "No explicit numerical C_ext^max is certified here.",

> "The near region and collar coefficients are not closed by this theorem.",

> "The 4.35 total coefficient remains blocked."

`C099_SCALED_FRAME_CERTIFICATE.json` records grade `DERIVED-EXACT-EXISTENCE`, an
`interval_numeric_floor` of `OPEN`, and

> "The result is an existence certificate, not a decimal certificate."

`periodized_bf_matrix_transfer_report.json` records both transfers as `PASS`, and notes of
the frame it used

> "The proof passes using the tiny raw 9-pin floor; a regularized frame would further enlarge margin."

## What this does not establish

One of these files has the word *certificate* in its name — `C099_SCALED_FRAME_CERTIFICATE.json`
— and none of the six grades contains it; two others use "certified" or "certify" in body
prose, and the folder's own title calls them transfer certificates. **A document
calling itself a certificate does not make it one**, and none of them is a certificate
issued by this repository: containment-guaranteed bounds here go through
`research/interval/` and none of this passes through it. The files themselves draw the line
this README would otherwise have to draw — existence certificate, not decimal certificate;
algebra exact, numerical comparison diagnostic; explicit coefficient not certified. Every
floating-point value in this folder is a non-certifying number produced elsewhere, and
nothing was recomputed, re-derived or run here.
