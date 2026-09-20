# HA-008 — ACTIVE TERMINAL DECISION CARD

**Items:** EC-020 (GP-CLS-PROP-024-v1.0) and EC-021 (GP-CLS-PROP-025-v1.0)
**Drafted by:** CL-01 (Claude), 2026-07-22, at Dylan Roy's instruction. Pending GP register sync (GP-REG-032).
**Status:** OPEN — awaiting operator decision.

## Governance boilerplate
- **Only Dylan M. Roy may fill the OPERATOR DECISION fields below.**
- General instructions to continue Drive work, organize files, or pursue research are **not** terminal approval.
- This card grants no closure by itself; it records the operator's explicit item-level decision.

---

## Item 1 — EC-020
- **Package:** GP-CLS-PROP-024-v1.0  ·  **Review:** CL-AUD-073-v1.0 (line-independent PASS, C1–C11)
- **Scope:** deterministic interval + monotonicity certificate for box P01-UNIFORMQ-D4-BOX-001 on 0<r≤1/20. No mass / exact-field / adjacency / P0.1 / P0.2 / theorem.
- **Required exact phrase to approve:** `APPROVE EC-020 EXACT INTERVAL AND MONOTONICITY SCOPE`
- **OPERATOR DECISION (Dylan only):** ________________________   **Date:** __________

## Item 2 — EC-021  (depends on EC-020)
- **Package:** GP-CLS-PROP-025-v1.0  ·  **Review:** CL-AUD-074-v1.0 (line-independent PASS, Q1–Q10)
- **Scope:** qualitative existence of r₁>0, c_Q4>0 giving uniform positive finite-Q4 Palm mass. No numerical values; not exact-field / P0.1.
- **Dependency:** EC-020 review discharged (CL-AUD-073); approve EC-020 first / together.
- **Required exact phrase to approve:** `APPROVE EC-021 EXACT UNIFORM FINITE-Q4 SCOPE`
- **OPERATOR DECISION (Dylan only):** ________________________   **Date:** __________

---

## What approval does and does not do
Approving closes only these two narrow objects. It does **not** advance P0.1, P0.2, Theorem B, machine replacement, ballot passage, release, or any deletion.

## Reversal
Any decision recorded here is non-destructive and can be corrected; nothing is trashed.

## Gemini-relay auto-finalize (per Dylan's instruction, 2026-07-22)
**Status: PENDING SETUP — not yet active.** When armed, an operator decision relayed via Gemini (GE-01) that (a) names the item, (b) carries the exact approval phrase above, and (c) carries Dylan's private passphrase, will be recorded automatically under `01_RECORDED_OPERATOR_DECISIONS` and this field updated, with a push notification to Dylan. Until armed, only Dylan's direct entry is terminal. Awaiting: channel confirmation + passphrase.
