# GP-LB-STAT-003 — K3 Missing-Square-Root and AO48 Handoff Reconciliation v1.0

Date: 2026-08-05

Status: ACTIVE ADDITIVE STATUS DELTA — NONCONTROLLING — LOWER CAMPAIGN HOLD

Provenance: ChatGPT/Dylan relay of the Kimi K3 Phase-0 checkpoint; the Cauchy–Schwarz defect below was independently checked. Exact Kimi Phase-0 raw carriers have not been delivered to ChatGPT or landed in Drive. This carrier is not a substitute for those missing Kimi originals.

## Purpose and precedence

This additive delta reaffirms `GP-LB-STAT-002` and corrects a later successor-facing status conflict in the raw AO48 session-state handoff at Drive ID `1cj3voNER0zIH_jYb3wdlsBSj4_WekQUg`.

That AO48 handoff remains frozen. Only its claims that the old WP Cauchy–Schwarz quantity is a certified upper bound, that `I_cs ~= 1.30399e-5` is thereby certified, that the rung upper-bound table is valid, and that `5.5e-3 r^1.6` is a verified WP upper envelope are superseded here. Its other assertions require their own evidence and are not adjudicated by this delta.

Controlling prior status delta:
https://drive.google.com/file/d/1ecRS56ODLlfK3cFNvhPYx9OIMurG6j9L/view

Conflicting AO48 session-state handoff:
https://drive.google.com/file/d/1cj3voNER0zIH_jYb3wdlsBSj4_WekQUg/view

## Independently verified mathematical defect

Under the conditional law `G = {grad f_tilde(y) = 0}`, let

`D = det H_y`

and

`A = {D < 0, b - ell < f_tilde(y) < b}`.

The typed window-saddle intensity satisfies

`rho_WP(y;r) / p_grad(y) = E[|D| 1_A | G]`.

Cauchy–Schwarz gives

`E[|D| 1_A | G] <= sqrt(E[D^2 | G]) sqrt(P(A | G))`.

If `P(A | G) <= min(P_type, P_window)`, the valid generic bound is

`rho_WP <= p_grad sqrt(E[D^2 | G]) sqrt(min(P_type, P_window))`.

The prior implementation instead multiplied by `min(P_type, P_window)` without the required outer square root. Since `p <= sqrt(p)` for `0 <= p <= 1`, that implementation is generally smaller than the valid Cauchy–Schwarz upper bound and is not a certified upper bound.

The reported witness values are directionally consistent with this defect:

- invalid missing-square-root expression: approximately `1.31310e-4`;
- Cauchy–Schwarz-valid generic expression: approximately `2.30247e-2`;
- ratio: approximately `175.35`.

These values confirm the defect’s magnitude at the witness point; they do not certify the remaining covariance, probability, or integration inputs.

## Quarantined claims

Until independently re-proved by a valid argument, the following are not theorem-grade upper bounds:

1. the pointwise missing-square-root WP bound;
2. the integrated `I_cs ~= 1.30399e-5` claim;
3. the WP rung “upper-bound” table;
4. the fitted `5.5e-3 r^1.6` envelope;
5. the WP-upper-bound branch of the reliability ledger;
6. the corresponding WP rows in `KIMI-AUD-023 v1.1` and `KIMI-THM-023 v1.1`;
7. every fixed lower-theorem constant or `r0` that consumes that branch.

The defect establishes nonproof, not numerical falsity of every displayed value. A separate independent proof could in principle recover some bound.

## Numerical evidence firewall

The numerical typed-intensity estimate

`I_WP(0.025) ~= 4.7602e-6`

is approximately `1.43` times the old printed budget

`0.213 (0.025)^3 = 3.328125e-6`.

This remains strong diagnostic evidence only. Without a rigorous lower enclosure it does not by itself refute the old integrated budget. A theorem-grade adjudication still requires exact periodized covariance, stable conditioned Gram inversion, conditional-Hessian and determinant-type control, Gaussian tails, spatial and quadrature enclosures, pin and near-singular charts, and uniformity in `r`.

## Dependency-minimization gate

The exact event inclusion must be re-derived before imposing a WP target rate:

- If WP is an absolute additive loss from a Lambda-side mass of order `r^3`, WP needs a coefficient-controlled `O(r^3)` bound; `o(r^3)` safely suffices.
- If WP is a conditional exclusion fraction multiplying a Lambda-side `r^3` mass, a proved uniform constant strictly below one may suffice, and `o(1)` preserves the leading coefficient.
- Multiplying a marginal WP intensity by Lambda mass is invalid without a correct conditional disintegration, matching Palm law, normalization, multiplicity accounting, and the `eta_r` transfer.

The Drive corpus has not yet supplied the exact event-level formula needed to choose between these cases. Therefore neither “WP must be `O(r^3)`” nor “a constant WP bound suffices” is currently established.

## Three-track firewall

### Track A — SIDE24 three-dimensional upper theorem

`AO48-OPR-045` remains ratified at its stated three-dimensional compact-positive-mark scope. `RP-C` and `RP-S` remain CLOSED. This lower-campaign defect does not reopen Track A.

### Track B — q0/P0.1 control overlay

`LS-CTL-003-v1.3` remains controlling: `FOUNDATION=TRUE`, `APPLICATION=TRUE`, `INDEPENDENCE=FALSE`, `INTEGRITY=FALSE`, `ELIGIBLE_P01_V110=FALSE`, and P0.1 is `HOLD / NOT PROMOTED`. The B0 evidentiary condition is satisfied, but the B0 Boolean remains unchanged and fold-ready.

### Track C — two-dimensional LB-RATE lower campaign

Disposition: OPEN / HOLD. The quantitative WP channel is OPEN; `P-NMZ-gamma` is OPEN; the Lambda-side `KIMI-DER-027b` is INCOMPLETE; `KIMI-DER-027a` is partial at its exact scope; `KIMI-DER-027c` is exact-rung deterministic-mean evidence only; and final lower assembly is not established.

## K3 intake state

Kimi reports Phase 0 completion and claims seven raw files under its inaccessible runtime. Exact Drive-name searches found no new K3 copies of `CANONICAL_STATE.json`, `INPUT_MANIFEST.sha256`, `FILE_LEDGER.tsv`, `MISSING_INPUTS.md`, `TASK_DAG.md`, or `ERRATA_2026-08-05.md`. The only exact-name `00_READ_FIRST.md` result predates this campaign and is unrelated.

Two locally available PDFs match the reported transport identities but contain replacement-character damage and are not canonical sources:

- `00_READ_FIRST.pdf`: 41,778 B; SHA-256 `5e1c4ddc30f39d2788794be3b3119bb4aaff9c57017ae2761811d9413980152f`; five U+FFFD characters;
- `ERRATA_2026-08-05.pdf`: 35,955 B; SHA-256 `f7658bcbadb83b9fae4493a48d4e817f0f0473ae62d5fbb5c25f4016b34763ff`; twelve U+FFFD characters.

They remain unlanded transport previews. Their damaged extracted text must not be reconstructed into a purported exact Kimi body.

## Next intake gate

Do not mint a theorem successor. Await:

1. W2 frozen report, source, raw transcript, separate exit receipt, mutation record, and hash;
2. W3 frozen report, interval-enclosure source, raw transcript, separate exit receipt, mutation record, and hash;
3. W4 frozen report, independent-representation source, raw transcript, separate exit receipt, mutation record, and hash;
4. updated blocker matrix;
5. only after W2/W3/W4 freeze, W12 blind adjudication.

Exact Phase-0 raw files and corrected ASCII-safe, zero-U+FFFD PDFs remain requested.

## Formal-control consequence

This work changes no LS-CTL Boolean, eligibility predicate, theorem status, RP status, ratification status, or q0 package status. Any such change requires separate operator adjudication.

