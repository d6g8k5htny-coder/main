# `05_HISTORICAL_GOVERNANCE — 2026-07-24` — eight reading copies

**Drive lane** `05_FOUNDATIONS_AND_PROTOCOL_HISTORY — FRESH START 2.0`
(`1PTiK2AUZSAn969ZATfUJuF7LC3wKGu7A`) · **folder**
`06_ARCHIVE` (`1rmN3-liGc6I_mb76ne3NCSxhQiWEqDP5`) /
`05_HISTORICAL_GOVERNANCE — 2026-07-24` (`1E7Ofq_IIP_OaeEvVYPOcnEKSi-EgyC1J`).
Ported 2026-09-20.

All eight are native Google Docs with **no digest in `drive/inventory.jsonl`**. Each is stored
as a `text/plain` export named `<title>.export.txt`, `exact:false`. **A rendering is not the
object.** The `sha256` in `_MANIFEST.jsonl` is of the export, first computed here on
2026-09-20; it verifies nothing about the Doc. `registers/json/frozen_objects.json` declares no
marker-delimited BODY digest for any of these ids, so no body digest was computed and none is
claimed.

Each manifest row carries the `artifact_index` / `operator_decisions` label for that object
**verbatim**. Where no register row carries the Drive id, the row says so instead of inventing
one.

## The supersession chain, in the sources' own words

This is a transcription. This repository decides nothing about which protocol governs the
Drive, and quoting a rule is not applying it.

1. **OP-PROT-003 — majority rule.** The document's own status line:
   "Status: SUPERSEDED FOR LIVE GOVERNANCE on 2026-07-24 … Current governing protocol:
   OP-PROT-009-v1.0". It records that "The former majority threshold, eligible voting roster,
   denominator, ballot process, Grok nonparticipation status, Gemini observer status, and
   post-test vote requirements are repealed. This Drive ID remains as revision-preserving
   historical provenance only."

2. **OP-PROT-007 — operator tie-breaker.** "Status: HISTORICAL PROVENANCE; tie-breaker
   mechanism superseded 2026-07-24 … There is no longer a ballot, tally, denominator,
   three-of-four default, observer designation, or tie-breaker mechanism. Operator direction is
   the sole governance decision channel." `artifact_index` nevertheless still labels it
   "BINDING — EXACT BALLOT ADOPTED", with authority "Ballot-specific governance authority; no
   mathematical or technical qualification effect"; `operator_decisions` OD-CNS-001-001 records
   "YES — ADOPTED / PASSED" and OD-CNS-001-002 "GOVERNANCE PASSED; TECHNICAL GATES NOT WAIVED".

3. **OP-PROT-009 — No-Vote Mandate (2026-07-24), with OP-GDN-007.** "This document is the sole
   live governance rule for this project. The project does not use votes, ballots, consensus
   thresholds, majorities, unanimity, quorums, denominators, eligible voting rosters, observer
   status, or non-participant status." Its §2 makes "ChatGPT, Claude, Grok, and Gemini …
   co-equal active contributors" who "may, at any time and in parallel" act, and its §3 says
   "No action requires approval from another model or from a vote." Its §7 is where "Git may
   serve as the system of record for formal artifacts and their commit integrity" first
   appears. **OP-GDN-007** is its participation companion: "Grok and Gemini are active
   contributors, equal in kind to ChatGPT and Claude … This record supersedes OP-GDN-003,
   OP-GDN-004, OP-GDN-005, OP-GDN-006". Both are quoted here as history. `operator_decisions`
   OD-NOVOTE-20260724-001 labels OP-PROT-009 "ACTIVE — NO-VOTE GOVERNANCE; ALL FOUR
   CONTRIBUTORS ACTIVE; CLEANUP AND BUILD-OUT AUTHORIZED" and then says "Superseded for current
   governance by OP-PROT-011 … it does not independently govern current work or fabricate
   mathematical approval." **Nothing in this repository derives any autonomy, approval or
   independence credit from these two documents.**

4. **OP-PROT-010 — the concurrent race, "VOID FOR CURRENT ROUTING".** Same date. It asserts
   "Dylan’s direct instruction supersedes conflicting Drive records. OP-PROT-009 and
   OP-GDN-007 are therefore superseded for live governance by this later instruction. They
   remain historical provenance and are not deleted." It restores the "GP, CL, AO48, and
   CW/C047R" roster at "three affirmative votes on the same numbered revision".
   `artifact_index` labels it "ACTIVE GOVERNANCE"; `operator_decisions` OD-OP-PROT-010-001
   labels the same object "VOID FOR CURRENT ROUTING — OP-PROT-010 QUARANTINED" and adds
   "OP-PROT-011 is the sole live governance rule; OP-PROT-010, OP-PROT-009, and OP-GDN-007
   have no current governing force." Two live registers disagree about one object; both
   labels are transcribed and neither is resolved here.

5. **OP-PROT-011 — Independent-Eyes Rule.** `artifact_index`: Status "RETAINED AS TECHNICAL
   PREDICATE / SUPERSEDED AS LIVE APPROVAL ROUTER BY OP-PROT-012"; authority
   "Independent-eyes, evidence-lineage, scope, provenance, and anti-consensus-laundering
   safeguards remain active where not superseded; no case-by-case human approval gate."
   Drive id `1HK9hoO4vD2lhE3wDXLPxPXhNBbJZJnNdA01Oh1FDeqA`, modified 2026-07-25T23:25:00Z. It is
   **not** in this folder and is **not** mirrored here.

6. **OP-PROT-012.** Named by `artifact_index` only through the objects that depend on it:
   "OP-PROT-012 is the sole live approval and decision-routing rule. OP-PROT-011 remains a
   technical safeguard and provenance layer." OP-PROT-013 is logged as an "ACTIVE — OPERATIONAL
   ADJUNCT TO OP-PROT-012". Not in this folder; not mirrored here.

7. **R17.** The current Research Home (`180yfvocozAaFRxf7tY8CDrobnpi17Sv-UkQGBBWCiD8`, mirrored
   at `drive/mirrors/00_START_HERE/`) says of all of the above: "Old control documents are
   history." The stored copy is a `text/plain` export and carries no emphasis.

The two Grok-participation records in between belong to the same week and are mirrored with the
rest: **OP-GDN-004** ("The latest authenticated operator instruction states that Grok is
nonparticipating and nonvoting"), **OP-GDN-005** ("Grok/xAI is reactivated as a NONVOTING
SUPPORT LINE"), **OP-GDN-006** ("A title-only or empty artifact cannot alter governance or
participation state"). `artifact_index` still labels OP-GDN-004 and OP-GDN-006
"ACTIVE — GROK NONPARTICIPATING / NONVOTING" while `operator_decisions` OD-GROK-001 labels
OP-GDN-004 "SUPERSEDED — GROK IS AN ACTIVE CONTRIBUTOR UNDER OP-GDN-007". Both are recorded.

One observation, recorded and not acted on: the OP-GDN-006 body states that a fresh body
retrieval of OP-GDN-005 (`17CdKqSbG1kWjFmbY1U0HJJTKLK2dTP1EvxZQo2cOvn4`) "returned no
paragraphs and no readable directive text". The 2026-09-20 export of that id is 5,079 bytes and
is not empty. The difference is noted in that row's manifest note. No register was edited and
no finding was filed from here.

## What this does not establish

These eight files are reading copies of historical governance records. Storing them changes no
status: no claim, premise, obligation, gate, grade, quarantine class or protocol standing moves,
the five validity premises of Theorem D1 v2.2(2) remain OPEN and `D3-LEMMA-RN-UNIF` is not
closed. This README does not decide which protocol governs the Drive — it quotes the chain and
stops. It supplies no autonomy grant, no approval, no review-independence rule and no
organizational-independence credit to anyone, in either direction: a same-provider reviewer
still earns zero. The register labels quoted here are transcribed from
`registers/json/artifact_index.json` and `registers/json/operator_decisions.json` as they stand;
where two of them disagree, the disagreement is reported, not repaired. CLAUDE.md governs this
repository; nothing in these documents does. Nothing stored here is imported, executed or tested
by CI.

Reading copies (`exact:false`) are not the objects, and a PDF rendering is not a frozen body.
