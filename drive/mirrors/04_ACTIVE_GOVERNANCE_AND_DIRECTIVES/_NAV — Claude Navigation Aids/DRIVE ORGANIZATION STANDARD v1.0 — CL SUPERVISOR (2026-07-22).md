# DRIVE ORGANIZATION STANDARD — v1.0

**Author:** CL (Claude), acting as reorg supervisor at the instruction of Dylan Roy
**Date:** 2026-07-22
**Status:** Supervisor review + proposed standard. Items in §7 marked "FOR DYLAN" await your approval; everything else is a recommendation for the executing model (GP / ChatGPT) to carry out.
**Scope:** The shared research Drive (root `0AGEUF_sx7o_MUk9PVA`), 270 folders / ~964 files as of this date.

---

## 0. Bottom line

The reorganization ChatGPT (self-identifying as "GP") is running is **safe and well-architected.** Nothing is being deleted or trashed; the design is non-destructive by construction. The open issues are **navigation hygiene**, not data loss — and they are all fixable with moves/renames plus the small amount of new structure I have already added.

- **What's right:** zero deletions, one-way quarantine with a restore path, provenance preserved, human-approval gates, conservative classification. (§1)
- **What's wrong:** triage sprawl (10 parallel "UNKNOWN" trees), inconsistent numbering/prefixes, three date formats, a live typo, empty dead-ends, half-built research tracks. (§2)
- **The rules going forward:** one naming standard, one date format, reserved number ranges, one triage inbox, documented templates. (§3–§6)
- **Who does what:** a punch-list for ChatGPT (moves/renames), what I already created, and three decisions for you. (§7)

---

## 1. Safety audit — what is being done correctly

This is the part that matters most for "done correctly," and it checks out. Confirmed against the governing documents (`URGENT DIRECTIVE — Workspace File Management Protocol`, `AMENDMENT 1 — One-Way Quarantine and Restoration Request Protocol`, `GP-OPR-135`, `GP-OPR-136`):

1. **Zero deletion / zero trashing.** The `DELETE — QUARANTINED CANDIDATES` bucket is deliberately empty. No object has been removed from the Drive; nothing has been sent to Trash. Migration receipts assert files-deleted = 0.
2. **One-way quarantine, reversible.** Questionable items are *moved* into `DELETE` / `OUTDATED` / `CORRUPTED`, never destroyed. Each of those buckets carries a `TAKE OUT OF FOLDER — REQUESTS` folder, so any item can be requested back. Physical removal requires an explicit instruction from you or a documented multi-model restoration decision.
3. **Provenance preserved.** Drive IDs, names, contents, and revision history are retained through moves. No re-uploads that would break lineage.
4. **Human-in-the-loop.** `_HUMAN_APPROVAL — DECISIONS NEEDED` holds 8 open decision cards awaiting your APPROVED/REJECTED. Directives are treated as operator-issued, not model-invented.
5. **Conservative classification.** The protocol explicitly forbids condemning a file on age, small size, short name, or duplicate *appearance* alone.

**Verdict:** keep going. The foundation does not need to change.

---

## 2. Findings — punch-list (prioritized)

**P1 — Triage sprawl (highest priority).** There are **10 separate "UNKNOWN ACTION — PEER REVIEW" trees**, with inconsistent prefixes (`90_`, `05_`, `04_`, `06_`, `03_`, and unprefixed at root). This is the single biggest way for unresolved items to get lost — ten inboxes instead of one — and it is actively growing (four were created during the audit itself). Locations:
1. `/UNKNOWN ACTION — PEER REVIEW` (root)
2. `/01_ACTIVE_RESEARCH_PACKAGES/05_UNKNOWN_ACTION — PEER REVIEW`
3. `/00_ACTIVE_GOVERNANCE_AND_DIRECTIVES/_OPEN_QUESTIONS/06_UNKNOWN_ACTION — PEER REVIEW` (empty stub)
4. `/01_ACTIVE_RESEARCH_PACKAGES/00_AXIOMATIC_CORE_SPINE/90_UNKNOWN_ACTION — PEER REVIEW`
5. `…/06_THEMATIC_RESEARCH_TRACKS/T1/90_UNKNOWN_ACTION — PEER REVIEW`
6. `…/T2/90_UNKNOWN_ACTION — PEER REVIEW`
7. `…/T3/90_UNKNOWN_ACTION — PEER REVIEW`
8. `/02_LEGACY_Q0_ARCHIVE/02_LEGACY_MANUSCRIPTS_AND_PROOFS/04_UNKNOWN_ACTION — PEER REVIEW`
9. `/02_LEGACY_Q0_ARCHIVE/03_LEGACY_CODE_DATA_AND_CALIBRATION/04_UNKNOWN_ACTION — PEER REVIEW`
10. `/03_PERSONAL_AND_EARLIER_RESEARCH/03_UNKNOWN_ACTION — PEER REVIEW`

**P2 — Numbering collisions.** Inside `01_ACTIVE_RESEARCH_PACKAGES`, two folders share the `01_` prefix (`01_P0.2_ADJACENITY_TRANSIT_TREE` and `01_REVIEWS_RESPONSES_AND_CLOSURES`), and one sibling (`Research Carry-Forward Canon`) is unnumbered. `02_LEGACY_Q0_ARCHIVE` starts at `02_` with no `00_`/`01_`.

**P3 — Live typo.** `01_P0.2_ADJACENITY_TRANSIT_TREE` should be `ADJACENCY` (its own child `P0.2 — ADJACENCY-TO-ONE CUBIC RATE` spells it correctly). Fix before more is filed under it.

**P4 — Three date formats.** `2026 / 07 — JULY` (active trees) vs `2026-07` (disposition buckets) vs `2026-07 — July` (legacy containers). Pick one (see §3).

**P5 — Empty dead-ends.** Three `LEGACY CONTAINER — …(pre-amendment)` folders each nest an empty `2026-07 — July`. Redundant; quarantine them.

**P6 — Asymmetric research tracks.** T1/T2/T3 have full charter + scaffold; **T4 and T5 are bare** (T5 has one file, T4 effectively empty). Either build them out to the track template or mark them `PLACEHOLDER`.

**P7 — Template drift.** The review scaffolds differ across locations (`01_EXACT_DUPLICATES_CONFIRMED` vs `01_SAME_NAME_BYTE_VARIANCE — DO NOT MERGE` vs `01_EXACT_DUPLICATE_PROVENANCE`; `..._COMPARISON_LOGS` vs `..._CROSS_FOLDER_LOGS`). Reconcile to one canonical set (§6).

*Low priority / by-design:* the root mixes numbered content buckets with ALL-CAPS status buckets. That contrast is intentional (status buckets should look different) — keep it, but hold the line on the two styles so nothing in between creeps in.

---

## 3. Canonical naming standard

Adopt these rules for **all new** folders, and apply to existing ones opportunistically:

- **Content buckets & structural folders:** `NN_UPPER_SNAKE_CASE`, two-digit number, e.g. `00_CHARTER_AND_SCOPE`. Numbers define sort order; no gaps unless reserved on purpose.
- **Sub-tiers within a package:** `NN.N_UPPER_SNAKE`, e.g. `01.4_VALIDATION_CERTIFICATES`.
- **Reserved special numbers:** `90_` = triage/needs-decision; `99_` = archive/superseded-responses. Don't reuse these for content.
- **Research targets keep semantic IDs:** `T1 — …`, `P0.1 — …`, `G0 — …` (letter+number + em-dash + Title). These are identifiers, not sort keys — that's fine; keep them consistent.
- **Status / quarantine buckets (top-level only):** `ALL-CAPS — DESCRIPTIVE` with em-dash: `DELETE — QUARANTINED CANDIDATES`, `OUTDATED — SUPERSEDED PROVENANCE`, `CORRUPTED — FAILED OR UNREADABLE`, `RETIRED — INACTIVE BUT VALID`. Never numbered.
- **Pinned governance folders:** leading `_` , e.g. `_NAV`, `_OPEN_QUESTIONS`. (Note: a leading underscore does **not** reliably sort to the very top in Drive's name-sort; if you want guaranteed top placement use a `00_` numeric prefix instead. The `_` convention is fine as a visual marker — just don't rely on it for ordering.)
- **Dates — ONE format everywhere:** **`YYYY / MM_MONTHNAME`**, e.g. `2026 / 07_JULY`. Month is zero-padded first (so it sorts correctly), name second (so it reads). **Deprecate** `2026-07` and `2026-07 — July`.
- **Characters:** ASCII where practical. Avoid glyphs that collide visually — e.g. T3's `ℓ` vs `l`; pick one and use it everywhere for that object.

---

## 4. Target root taxonomy (future-proof)

Keep root **folders-only** (no loose files — currently true, hold it). Reserve number ranges so new domains never force a renumber:

| Prefix | Bucket | Purpose |
|---|---|---|
| `00_` | ACTIVE_GOVERNANCE_AND_DIRECTIVES | Rules, directives, approvals, navigation, coordination |
| `01_` | ACTIVE_RESEARCH_PACKAGES | Live research (theorems, problem targets, machines) |
| `02_` | LEGACY_Q0_ARCHIVE | The completed Q0 line, frozen |
| `03_` | PERSONAL_AND_EARLIER_RESEARCH | Personal corpus, imported text, earlier method work |
| `04_`–`09_` | **RESERVED** | Future active domains — add here, never renumber 00–03 |
| — | `DELETE — QUARANTINED CANDIDATES` | Consensus/authorized removal staging (empty by default) |
| — | `OUTDATED — SUPERSEDED PROVENANCE` | Older versions with a verified newer replacement |
| — | `CORRUPTED — FAILED OR UNREADABLE` | Broken/unreadable/failed objects |
| — | `RETIRED — INACTIVE BUT VALID` **(new)** | Whole workstreams no longer in use but still correct |

**The four "not-active" buckets are semantically distinct — keep them separate:**
- **DELETE** = intended for removal (needs authorization).
- **OUTDATED** = *wrong or superseded* — a better version exists.
- **CORRUPTED** = *broken* — can't be read/used.
- **RETIRED** = *fine, just finished* — no longer worked on, nothing wrong with it. This is the one you asked for ("folders no longer in use"), and nothing currently covers it (LEGACY is one specific archived line, not a general home).

---

## 5. Single triage inbox (replaces the 10 scattered trees)

**Rule:** there is exactly **one** place for "I can't confidently classify this yet." No more per-area `UNKNOWN` / `90_UNKNOWN` trees.

- Canonical location (created): `00_ACTIVE_GOVERNANCE_AND_DIRECTIVES / _TRIAGE_INBOX — DRIVE-WIDE`.
- Inside it, one master index doc logs each pending item with a pointer back to where it came from, so nothing loses its context.
- **Migration (ChatGPT):** move the *contents* of the 10 existing trees into this inbox (recording origin in the index), then quarantine the emptied scaffolds. Going forward, unresolved items go straight here.

If you'd rather keep items physically near their context, the fallback is one inbox per top-level bucket (max 4) plus this same central index — but a single inbox is cleaner and is what I've set up.

---

## 6. Canonical templates (stop the drift)

**Package template** (use verbatim for every P-target / research package):
`01_SOURCE_LINKS_AND_OBJECT_CARD` · `02_ACTIVE_DERIVATIONS` · `03_COMPUTATION_AND_TESTS` · `04_REVIEW_AND_FALSIFIERS` · `05_PROMOTION_PACKAGE` · `99_ARCHIVE_RESPONSES`. *(Already consistent across P0.1–P1.3 — just codifying it.)*

**Theorem-track template** (apply uniformly to T1–T5):
`00_CHARTER_AND_SCOPE` · `01_<CORE>_MANUSCRIPTS` · `02_SUPPORTING_CERTIFICATES` · `03_MACHINE_REPORTS_AND_ROOTS`. Triage no longer lives per-track (see §5). Build T4/T5 to this shape or label `PLACEHOLDER`.

**Review scaffold** (one canonical child set — reconcile the drift):
`00_REVIEW_INSTRUCTIONS_AND_ACTION_CARDS` · `01_DUPLICATE_OR_VARIANT_REVIEW` · `02_CONTENT_AND_PROVENANCE_REVIEW` · `03_CORRUPTION_OR_READABILITY_REVIEW` · `04_CROSS_DEPENDENCY_REVIEW` · `05_RETURNED_FROM_REVIEW — RESTORED`. Pick these names everywhere; drop the one-off variants.

---

## 7. Action list

### A. For ChatGPT / GP (requires move / rename authority — I can't do these)
1. **Rename** `01_P0.2_ADJACENITY_TRANSIT_TREE` → `…ADJACENCY…` (P3).
2. **Resolve the duplicate `01_`** in `01_ACTIVE_RESEARCH_PACKAGES`: renumber `01_REVIEWS_RESPONSES_AND_CLOSURES` (e.g. to the next free number) and give `Research Carry-Forward Canon` a number (P2).
3. **Consolidate** the 10 `UNKNOWN` trees into `_TRIAGE_INBOX — DRIVE-WIDE`; log origins in the index; quarantine the emptied scaffolds (P1).
4. **Standardize all date folders** to `YYYY / MM_MONTHNAME` (P4).
5. **Quarantine the empty `LEGACY CONTAINER` dead-ends** (P5).
6. **Reconcile review-scaffold child names** to the §6 canonical set (P7).
7. **Build out or mark `PLACEHOLDER`** on T4 and T5 (P6).

### B. Already done by CL (me) — additive, reversible
- Created `RETIRED — INACTIVE BUT VALID` at root (+ README).
- Created `_TRIAGE_INBOX — DRIVE-WIDE` under governance (+ README/index).
- Wrote this standard into `_NAV — Claude Navigation Aids` and delivered a copy to Dylan.
- Stood up a daily drift monitor (see §8).

### C. For Dylan — decisions (I proceeded on the recommended default; override anytime)
1. **RETIRED bucket** — I created it top-level. Keep it there, make it a per-area subfolder instead, or drop it? *(Default: keep.)*
2. **Single triage inbox** vs one-per-area — I built the single inbox. *(Default: single.)*
3. **Deprecate the two extra date formats** and standardize on `YYYY / MM_MONTHNAME`? *(Default: yes.)*

If any default is wrong, tell me and I'll adjust; ChatGPT can relocate/quarantine anything I made via the normal protocol.

---

## 8. Monitoring

A daily supervisor check runs against the Drive and reports **drift** (not a full re-audit): new loose files at root, any new `UNKNOWN` trees, new naming/date violations, asymmetric or empty new scaffolds, and — most important — any sign of a file being trashed or hard-deleted (the protocol is zero-deletion; that would be flagged immediately). It reads this standard each run so the rules stay the source of truth. Cadence and on/off are adjustable on request.

---

*Appendix (full 270-folder snapshot) is in the copy delivered to Dylan and can be regenerated on request.*
