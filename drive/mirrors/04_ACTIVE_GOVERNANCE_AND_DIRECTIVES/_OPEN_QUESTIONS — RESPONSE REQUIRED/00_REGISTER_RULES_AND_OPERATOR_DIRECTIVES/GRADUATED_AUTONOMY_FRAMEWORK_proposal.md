# GRADUATED AUTONOMY FRAMEWORK — Operator Proposal

**Provisional ID:** OP-PROT-008 *(confirm the next free OP-PROT number on register sync — do not reuse a colliding ID)*
**Operator:** Dylan M. Roy · **Drafted / refined by:** CL-01 (Claude) · **For upload by:** GE-01 (verbatim relay)
**Date:** 2026-07-22
**Status:** PROPOSAL — PENDING OPERATOR RATIFICATION.
**This document authorizes nothing by itself.** No closure, promotion, register change, or automation activates on the basis of this file. It becomes operative only when Dylan M. Roy records the exact ratification sentence in §9 — and even then only subject to the preconditions in §4.

---

## 1. Thesis
The original fail-closed design — every terminal closure, promotion, and register update requiring fresh human eyes — was the correct response to a chaotic, high-entropy workspace: colliding IDs, self-authorizing documents, mutable theorem bodies, and unclear independence. That phase is largely over. The workspace now holds the primitives needed for **safe, graduated, reversible** autonomy, and continuing to treat every ready package as if it still lived in the chaotic regime now costs more than it protects — it spends the operator's attention on coordination instead of the hard mathematical fronts. This proposal graduates autonomy deliberately, with machine-checkable gates and human-only hard stops preserved.

## 2. Primitives now in place
- Clean folder hierarchy (Active / Governance / Human-Approval / Quarantine / Legacy).
- Protected Operator Decisions and Closure Log as the only trusted sources of terminal authority.
- Content-addressed freezes (exact SHA-256 of theorem bodies + body-byte counts).
- Explicit multi-line independence partitions with required exact verdict wording.
- Numbered closure packages carrying object, exclusions, dependencies, and reopening rules.
- A proven cascade of narrow easy-closures successfully terminalized under these rules.
- OP-GDN-003 freshness rules; silence / same-line / general authorization explicitly non-counting.

## 3. The current weak point — fix this first (honest)
The soft spot today is **not** closure rigor; it is the **authentication and attribution of operator decisions themselves.** A recent operator approval relayed through the non-voting observer channel named the *wrong item* and carried no signature or scope — a free-text assertion any process could have produced. Before automation is allowed to act on "the operator approved X," that channel must be hardened (§4a). This is a precondition, not a reason to stall: it is the single change that makes everything below safe.

## 4. Preconditions before ANY autonomy activates
a. **Authenticated operator-decision channel.** Operator decisions must arrive in a fixed, signed format — item ID + exact scoped approval phrase + a private operator passphrase — into a defined inbox, not as free-text prose from a model relay. Unsigned or malformed "approvals" are logged and ignored, never actioned.
b. **Independence standard resolved.** Adopt the pending GP-PRP-130 clarification so "≥2 distinct eligible non-GP lines" is unambiguous — stating explicitly, per class, whether *organizational* independence (distinct provider) is required or *line* independence suffices. (Note: several recent easy-closures carried only a single non-GP line; under the ≥2-line rule below they would **not** auto-terminalize — the intended, stricter behavior.)
c. **Every standing authorization is itself human-approved, versioned, and revocable.** No class self-expands; adding or widening a class is a human act (§6).
d. **Full audit + kill-switch.** Every automated action is additive, attributed (CODE-NN), logged to the Operator Decisions / Closure Log together with the criteria it checked, and reversible via existing reopening paths. A single operator command pauses all automation.

## 5. Where more autonomy is now justified

**Tier 1 — Register & surface maintenance (finish GP-AUTO-034).** Once byte-exact source, injection/revocation tests, and multi-line review of the eligibility path pass, activate it. It never promotes mathematics; it only keeps the live surface (dashboard / START_HERE / queue) consistent with the trusted registers, removing the recurring "is the surface stale?" tax.

**Tier 2 — Standing class authorizations for proven easy-closure types (biggest practical win).** Pre-authorize terminalization for well-characterized classes once machine-checkable conditions are met. Candidate classes with a track record: deterministic interval / monotonicity certificates; fixed-r qualitative positivity (exact full-field or finite-Q4); exact algebraic / normal-form identities; negative-result retirements of imported claims (diagnostics retained). **Conditions — ALL required before any auto-terminalization:** complete numbered package exists; ≥2 distinct eligible non-GP lines filed exact affirmative verdicts (per §4b); no open material objection; frozen hashes / source receipts match; exact object + exclusions + reopening rule present; item belongs to a pre-authorized class. If all pass → models write the Closure Log entry, update the queue, remove the item from active attention, and log the check. The operator is pinged only when a new class appears or a condition fails.

**Tier 3 — Theorem-promotion pipeline (automate up to the final signature).** Keep the final "this is a canonical theorem / update Q0_MASTER / change machine roots" decision strictly human. Automate everything before it: content-addressed body freeze (already live for P0.1); multi-line subchain partition with exact wording; on satisfaction of minimum independence + fresh scope audit → auto-generate and freeze a complete Promotion Proposal, mark it HUMAN-PROMOTION-READY, and surface one decision card with the frozen hash and the exact approval sentence. The operator decides only "APPROVE PROMOTION OF [hash]" or reject / amend — no reconstructing the chain each time.

**Tier 4 — Autonomous research-front continuation.** Formalize a light loop under OP-GDN-003: when a same-line derivation + audit completes → auto-open the required independence package; eligible fresh reviewers are notified / can claim; only scope expansions or unresolved material objections escalate to the operator.

## 6. What stays strictly human (and should)
- Final theorem / canonical-state promotion.
- Activation or expansion of the automation itself.
- Creation or expansion of standing authorization classes.
- Any new mathematical object class or material scope change.
- Changes to the independence standard.
- Machine-root changes, releases, deletions, or any destructive operation.

These are the points where one bad decision has irreversible scientific or governance consequences. Everything else can now be pushed downstream.

## 7. Safety invariants
Every automated action is additive, logged, CODE-NN-attributed, and reversible through existing reopening rules; standing authorizations are versioned and expire / require renewal rather than persisting silently; the operator holds a single global pause / kill command; and nothing in Tiers 1–4 may reach into any §6 item.

## 8. Recommended next steps (in order)
1. Harden the operator-decision channel (§4a); stand up the audit log + kill-switch (§4d).
2. Finish the remaining GP-AUTO-034 v1.3 technical gates and activate register maintenance (Tier 1).
3. Adopt the GP-PRP-130 independence clarification (§4b).
4. Draft and issue 1–2 standing class authorizations for the current narrow easy-closure families (Tier 2).
5. Define the exact machine-checkable criteria + template for "Promotion Proposal → HUMAN-PROMOTION-READY" (Tier 3).
6. Only after a few clean cycles, consider further lightening (e.g., timed quiet-period auto-terminal under standing authorization).

## 9. Ratification
This framework becomes operative only when the operator records, in the Operator Decisions log:

> **"APPROVE OP-PROT-008 GRADUATED AUTONOMY FRAMEWORK — activate Tier 1 and the §4 preconditions; Tiers 2–4 activate only as their named standing authorizations are separately approved."**

Selective adoption (naming only the tiers / preconditions to activate) is permitted. Until then, the current fail-closed process remains in force. The rigor mechanisms already built — frozen hashes, multi-line exact verdicts, protected registers, additive reopening rules — are precisely what allow the repetitive coordination work to be handed off *without lowering the bar*.

*END — OP-PROT-008 (PROPOSED)*
