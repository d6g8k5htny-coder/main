# LS-CTL-003-v1.3 — Foundation-gate successor overlay

Artifact ID: LS-CTL-003-v1.3
Date: 2026-08-03 America/Chicago (2026-08-04 UTC evidence round)
Prepared by: GP / OpenAI GPT-5.6
Class: CTL — fail-closed Boolean successor overlay
Authority: Dylan Roy operator tasking; AO48-AUD-056-v1.0 reconciliation; no theorem-promotion authority
Predecessor: LS-CTL-003-v1.2, raw Drive ID `1Viah0Ly0IoNrpf7slJxXxzTv_Uv_bTC7`; 4,875 bytes; SHA-256 `f09d5966e347aa444a138fd0337c8eaf29fdefc9d8259fa275e6a973d8814416`
Controlling theorem successor: GP-DER-118-v1.10
Whole carrier: 30,933 bytes; SHA-256 `c1d6e559274d7e87a3fa92f44cb2534a5e277a3b5941fd40db380599ddba0564`
Frozen theorem body: 29,293 bytes; SHA-256 `9b7901e112a4857e3ea59942858684f72fc09a2825b1efa859360dd8fa55f014`
Reconciliation: AO48-AUD-056-v1.0, Drive ID `1BBif8dKeih_wFQxe2nl-8bcdDkvwTSV5`
Byte-tier receipt: AO48-AUD-057-v1.0, Drive ID `1oqJKAypxT2a9foJoamIYDJJVZUrbu6qJ`

This successor preserves every v1.2 state except the two operator-directed transitions `E0` and `E2`. It records the resulting foundation block exactly. It does not promote P0.1, infer a B0 transition, alter a frozen proof body, or turn the separate C2 advisory into a closure.

## Boolean delta

| Variable | v1.2 state | v1.3 state | Basis |
|---|---:|---:|---|
| E0 — measurable-event parent at current hash | FALSE/PENDING | **TRUE** | KIMI-AUD-020, APPROVE; exact report 5,475 B / SHA-256 `9daf84d0db98507ca7f766f60eb12351911368c42cab6129a9099533c307188b`; GP-DER-197 frozen body 13,797 B / SHA-256 `035d5a18018606bd8736dc5774a7adbed29de453b30a1780976e1e4f69285d4d`; AO48-AUD-056/057 reconcile and bind the byte tier |
| E2 — selector/globalization at current hash | FALSE/PENDING | **TRUE** | KIMI-AUD-021, APPROVE at stated scope; exact report 4,848 B / SHA-256 `1245fa9b3a211c53c9176dbbdb64352b629182b38c7cc51de653e6868b266548`; LCR-DER-061-v1.1 8,928 B / SHA-256 `8ec4386fc1214523c98a5ae84acf4f5f64a6601aa2c63728f4382701b82dc3b7`; selector dependency LCR-DER-057-v1.1 11,144 B / SHA-256 `62c23788fbed764240546b9f9aa84b40620a4e0725a39f2bec1fe7678fc74e72`; AO48-AUD-056/057 reconcile and bind the byte tier |
| Every other predicate | unchanged | unchanged | No automatic transfer from a report, an advisory, or an adjacent track |

## Re-evaluation

The controlling formula remains

`FOUNDATION = O0 ∧ O1 ∧ G0 ∧ Q0 ∧ T0 ∧ T1 ∧ N0 ∧ N1 ∧ E0 ∧ E1 ∧ E2`.

Every factor now evaluates TRUE. Therefore:

- `FOUNDATION = TRUE`.
- `APPLICATION = TRUE`, carried unchanged from v1.2 (`D0 = I0 = TRUE`).
- `INDEPENDENCE = FALSE`, because no B0, A0, or F0 transition is executed by this overlay.
- `INTEGRITY = FALSE`, unchanged.
- `ELIGIBLE_P01_V110 = FALSE`.
- P0.1 disposition remains `HOLD / NOT PROMOTED`.

## B0 boundary

The formal evidence condition for B0 is satisfied: one eligible provider family organizationally distinct from OpenAI and Anthropic returned separate affirmative verdicts for PZ0 (KIMI-AUD-012), I0 (KIMI-AUD-013), and D0 (KIMI-AUD-014). AO48-AUD-049, AO48-AUD-050, and AO48-AUD-056 reconcile that same-lineage bundle, and AO48-AUD-056 explicitly records “B0 condition satisfied.”

This v1.3 overlay does not execute that additional transition because the operator instruction identifies the v1.3 delta as `E0,E2 → TRUE` and asks for a B0 assessment. B0 is therefore `FOLD-READY / STATE UNCHANGED` here. A successor control action may set PZ0 and B0 TRUE after explicitly acknowledging the LS-WO-001/CLWO-P01-001 binding. F0 still requires A0 plus verified family distinction; it is not inferred from B0 alone.

## C2 boundary

Run 28 supplies a constructive C2 route, not an external closure. The route is recorded separately for the S2/LS authoring lane as `ROUTED / TYPED GENERIC CROSSWALK PENDING`. It changes no Boolean in this overlay and does not reopen AO48-OPR-045.

## Reopening rule

E0 or E2 reopens only upon an in-scope counterexample, failure of the corresponding audit’s acceptance conditions, identity drift in a controlling frozen carrier, or an operator disposition that explicitly supersedes this transition. Missing convenience mirrors alone do not reopen a byte-bound mathematical gate; a provenance conflict does.

END LS-CTL-003-v1.3
