# How the Drive's operating architecture maps onto this repository

**2026-09-21 operational correction:** the historical claim-lease and
publication-conflict rows below overstate Git's guarantees. R17 Work Events
remain the cooperative claim log. A clean merge can combine semantically
incompatible changes; a fast-forward push is not a Drive/GitHub transaction or
a lock on a separately observed base. Bind and revalidate the exact source,
claim, base and candidate, verify the resulting bytes on both surfaces, and
reconcile competing outputs explicitly. See the
[scoped v1.1 correction](https://drive.google.com/file/d/1cKPOpv1foB7F7aBA-N0O8vtzetPWKjyR/view).
This note adopts that publication boundary on this branch; it does not deploy
PR4's shadow tooling, migrate scientific records or change any scientific gate.

The Google Drive research share runs under a family of operator-issued
protocols: OP-PROT-019-v1.1 (R17, the control plane at export), OP-GDN-002,
OP-CNS-001, and the register-labelled operational protocols OP-PROT-013
("ACTIVE — OPERATIONAL ADJUNCT TO OP-PROT-012"), -014 ("CURRENT ACTIVE
OPERATIONAL PROTOCOL"), -015-v1.1 ("ACTIVE"), -016 ("ACTIVE CONTROL"), -017
("PUBLISHED / ADOPTION-READY"), -018 ("ACTIVE CONTROL PROTOCOL") and
OP-RECON-20260916 ("ACTIVE / RAW READBACK VERIFIED"), with OP-PROT-006, -011 and
-012 retained in the roles the registers give them (see `README.md` here). Only
the first three and OP-PROT-012 are reading copies in this repository; the rest
are indexed, not held, and their constructs (cross-line work orders and
receipts, task integrity gates, new-model entry and the tool-call circuit
breaker, XLSX export verification, line-anchored frozen-body extraction,
advisory-lane deconfliction) have no rows in the table below yet. They were
written for a Drive + Google Sheets substrate. This document states, construct by
construct, what the git-native equivalent is in this repository, and where the
substrate changes the guarantees.

Nothing here promotes, closes, or reclassifies any mathematical claim. Status
labels are carried from the source verbatim (see `registers/`).

The operator decision in the corpus about a Git repository is the FORMALIZATION
BOARD's OPERATOR PACKAGE DECISION of 2026-07-24 (Drive
`10o4YRYOr8a2fB6rtnFnzMn7HkQMfv9-FZ5Mh0L-KF_o`): "REPOSITORY ROUTING: private Git
repository creation and exact-history push are approved when platform access
becomes available." The 2026-07-24 COMMAND CENTER (Drive `1yfJu9h6…`) restates
it with its operating constraints — "Git repository status: OPERATOR-APPROVED /
PLATFORM-BLOCKED — create and push the private repository when connector or
platform access is available. Do not retry unsupported creation in a loop and
do not claim a repository already exists." — and CL-REQ-225 Rule 5 keeps "the
GitHub repository item" recorded "exactly this way". (Until 2026-09-19 this
paragraph called the Board's sentence the only one; the restatements are
GP/CL-line records of the same decision.) It approves a *private* repository as the formalization,
source-control, reproducibility and CI-handoff system of record and, in the
same decision, "NOT APPROVED: blanket terminalization or mathematical promotion
of any included claim." This repository is **public** — checked on 2026-09-26 two ways: the
GitHub API reports `private=false`, and the account's repository listing reports
`visibility=public`. The one *sourced* record in this tree about that visibility
is `drive/deltas/2026-09-18/DG-EXEC-20260918-49291487/DRIVE_GITHUB_EXECUTION_HANDOFF.md`:
"The repository was public at inspection. Preserve that visibility: the owner
asked to restrict who works in it, not to hide the existing public record."
Whether to keep it so remains the owner's to decide. Private draft work may
continue; nothing in this paragraph authorizes public publication or mathematical
promotion.

**Withdrawn on 2026-09-26, and recorded rather than deleted.** Until that date
this paragraph and `docs/RESEARCH_MAP.md` each asserted an operator directive
about repository visibility, with no source anywhere in this tree, and the two
contradicted each other. Quoted verbatim so the withdrawal is checkable — as
transcription, not as assertion; **no operator directive is asserted here:**

```text
governance/GIT_ADAPTATION.md
  "Dylan subsequently directed that the repository remain private until
   publication is explicitly authorized. The authenticated GitHub API confirmed
   private visibility on 2026-09-20 UTC."

docs/RESEARCH_MAP.md
  "Dylan subsequently authorized public visibility on 2026-09-20;
   unauthenticated access was verified that day."
```

They cannot both be true, they are dated the same day, and neither cites a Drive
id, an artifact id, or any record in `registers/` or `drive/`. The status half of
the first is also false: the repository is public. And the first is not merely
unsourced but **contradicted by a sourced record already in the tree** — the
2026-09-18 execution handoff cited above says to preserve public visibility and
restrict who works in the repository, not to hide it. Which, if either, reflects an
actual decision is the operator's to say and nobody else's — this note asserts
only that neither was sourced and that both could not hold.
`tools/operator_directive_check.py`, added in the same commit, refuses either
shape from now on.

| Drive construct (protocol) | Repository equivalent | Notes on guarantees |
|---|---|---|
| Stable Drive file ID as object identity (DEF-001) | Path in this repo **plus** SHA-256 in a manifest; `drive/inventory.jsonl` maps every Drive ID to title/parent/path | Git content addressing is stronger than Drive IDs: a blob hash is the identity, and history is immutable. The Drive ID is retained as provenance metadata. |
| Frozen objects register (byte count + SHA-256, "no in-place edit; numbered successor") | `registers/json/frozen_objects.json` (195 rows in the 2026-09-18 export; 188 in the truncated 2026-09-17 rendering) cross-checked by `tools/frozen_check.py` against the digests the accessibility source map recorded independently for the same Drive IDs | Offline and partial by construction: the 62 rows frozen by whole-file digest (classes A/D) are comparable and all agree; the 116 marker-delimited or export-body rows (classes B/C) are not comparable offline and 17 of their bodies are present byte-exact as payloads (`python3 tools/frozen_check.py`: rows=195 comparable=62 match=62 mismatch=0 body_present=17 not_comparable=116; until 2026-09-19 this row carried the 188/55/133 figures of the truncated rendering) to a native Doc's size and are reported as such, 17 of them with the body present byte-exact in the payload index. Bodies this repository holds byte-exact (`engine/rn_engine/frozen/`, `engine/carriers/blobs/`) are verified by their own manifests. Nothing re-freezes anything. Until 2026-09-18 this row claimed CI failed on frozen-byte drift while no checker existed. |
| Work Events (append-only coordination log, AppendCells only) | `registers/json/work_events.json` guarded by `tests/test_registers.py::test_work_events_append_only` (rows may only be appended; CI diffs against the parent commit) | Git commits are themselves an append-only, hash-chained event log with author identity; PRs are the claim/publication surface. |
| Claim lease (120-minute, cooperative, not compare-and-swap) | A branch per claim; the PR is the claim; merge is compare-and-swap on the base branch head | Git gives the transactional guarantee the protocol says Sheets cannot: a push that races is rejected, never silently overwritten. |
| Publication conflict (two successors of one head → hold and reconcile) | Merge conflict; resolved by a reconciliation commit that preserves both branches in history | Same semantics, machine-enforced. |
| DRAFT → CANDIDATE_VERIFIED → READY_FOR_REVIEW → REVIEWED/AMEND | `sandbox/` (no authority) → PR opened as draft → PR marked ready → review verdict recorded in `registers/` and merged | Draft PRs carry no canonical authority, mirroring `06_SANDBOX_FRONTIER`. |
| Review Queue with 7/14/30-day aging ladder | `registers/json/review_queue.json`; suggested: one GitHub issue per review key, labels `age:7`, `age:14`, `age:30` | Aging is prioritisation only; it never approves. |
| Technical status vs organizational independence (four separate dimensions) | Review verdict files record correctness, scope, authorship/exposure and independence separately; PR approval from the same provider family earns zero independence credit | Unchanged rule; git does not weaken it. |
| Quarantine (EXACT_DUPLICATE / SUPERSEDED / DEFECTIVE_SCOPE / UNVERIFIED-CONFLICT / LEGACY_INSPIRATION) | `quarantine/` directory with `registers/json/quarantine_index.json`; moves are commits with the original path recorded (rollback record) | No permanent deletion: git history retains every byte. |
| Logical quarantine of frozen archive members (carrier ID + relative path + hash exclusion) | `quarantine/EXCLUSIONS.json` listing archive, member path and SHA-256; verifiers refuse to consume excluded members | Same as the Drive's path-and-hash exclusion. |
| DO_NOT_OPEN vault (superseded mirrors, dead ends, trap copies) | **Not mirrored.** Only the vault's metadata tree is recorded in `drive/vault_tree.txt` (7 rows: the 6 inventory items whose path begins `01_ACTIVE_RESEARCH_PACKAGES/99_DO_NOT_OPEN` — the folder and five native Docs — plus the external `00_DO_NOT_OPEN_MANIFEST`, which the manifest itself says sits outside the vault and which is held as a reading copy under `drive/mirrors/01_ACTIVE_RESEARCH_PACKAGES — ROOT (the vault manifest)/` since 2026-09-19; id, kind, byte count, path, from the inventory; no digest exists for any of them and none was computed); contents are not in this repository and were never opened | Respects the standing order that models must not consume vault contents for authority. The file referenced here did not exist until 2026-09-18. |
| Zero evidentiary authority / inspiration-only lanes | `legacy/` with the same banners | Nothing under `legacy/` may be cited as evidence. |
| Handoff envelope (payload + payload_sha256, noncircular) | The program's envelopes arrive as Drive objects and are mirrored byte-exact under `drive/deltas/<date>/<folder>/` (2026-09-18: `ACCESSIBILITY_HANDOFF.json`, `DRIVE_STRUCTURE_HANDOFF.json`, the DG-EXEC `DELIVERY_MANIFEST.json` and `READBACK_RECEIPT.json`); where an envelope embeds payload digests (`DRIVE_STRUCTURE_HANDOFF.json`, seven `sha256` fields) they were recomputed on fetch; `ACCESSIBILITY_HANDOFF.json` lists ids, titles and links and embeds no digest at all, so nothing in it could be recomputed — the only register-declared digest for that publication is the completion report's (`work_events` rows 27–28 counting the data rows from one, the `PUBLISH-ACCESS-` and `RELEASE-ACCESS-2026-09-17T23:57:33.664Z` events, `b7a47ec1…`), and that report **is** held here — `drive/deltas/2026-09-18/07_MODEL_ACCESSIBILITY_extras/ACCESSIBILITY_COMPLETION_REPORT.md`, 4,828 bytes, which re-hashes to exactly that digest, so the register's declared value and the stored bytes agree (until 2026-09-19 this row said the embedded digests of both envelopes were recomputed; until 2026-09-20 it said the report was “not held here” and cited the digest at “rows 26–27”, which is its zero-based index, not its row number), and every mirrored blob is verified by `tools/verify_manifests.py` through the folder's `_MANIFEST.jsonl` | Same schema; hash the payload only. Until 2026-09-18 this row named a `drive/handoffs/` directory that did not exist. |
| Delivery manifest per bundle (one manifest may cover many payloads) | `_MANIFEST.jsonl` per mirrored folder under `drive/deltas/<date>/` (dest, bytes, sha256, exact, stored, not_stored_reason), verified in CI; the `MANIFEST.sha256` files some Drive packages ship (the KIMI normalized export, for one) are indexed by the inventory and not yet mirrored | Verified in CI where mirrored. Before 2026-09-18 no manifest of either kind existed in this repository, so this row described a check that never ran. |
| Autonomy classes 0–5 (OP-PROT-012 — Drive title since 2026-09-17: `HISTORICAL OP-PROT-012 — APPROVAL ROUTING SUPERSEDED BY R17`; `autonomy_control` AUTONOMOUS_DECISION_BUDGET "RETIRED — R17 evidence predicates") | Class 0–1 (read, map, move, rename) are ordinary commits; Class 2–3 (terminal state, theorem promotion) require the predicates in the protocol **and** a recorded verdict in `registers/`; Class 4 (external release) stays DISABLED; Class 5 (permanent destruction) is impossible in git without history rewrite, which this repo forbids | Branch protection should forbid force-push and history rewrite on `main`. |
| Fresh Start 2.0 control plane (`02_FORMAL_AND_LEAN_RESEARCH_SYSTEM`: state machine `INGESTED → PREPPING → CANDIDATE → REVIEWED → CORE`, promotion gates PG-01…PG-10, DRAFT_ID transactions, the `FS2_CONTROL_REGISTRY` sheet, `math_integrity_gate.py`, the `FS2_PCT003` Apps Script executor) | **No repository equivalent.** The lane is indexed in `drive/inventory.jsonl` (178 items) and described in `docs/RESEARCH_MAP.md` §9; none of its state machine, gates, registry or executor is implemented, mirrored or run here, and the eight Core capsules' `CURRENT_STATUS.json` labels are not transcribed. Distinguish the *control plane* from the FS2 **governance artifacts**, which are a different lane and are mirrored: since 2026-09-20 `drive/mirrors/05_FOUNDATIONS_AND_PROTOCOL_HISTORY — FRESH START 2.0/00_GOVERNANCE_ARTIFACTS/` holds the charter, the active-authority allowlist, the execution report, the legacy quarantine manifest, the contamination and remediation ledger, the legacy rebuild queue and the query and retrieval policy, byte-exact. Those are the program's rules and its record of executing them; they are quoted, and neither they nor anything else here implements the admission gate. | The source calls it "INSTALLED / ACTIVE CONTROL PLANE" (`00_READ_FIRST`, `FS2-SETUP-001`). A git-side counterpart would be this repository's own construction and would have to be labelled so, never presented as the Drive's Core-admission gate. Its autonomy grants addressed to AI sessions (`SA-001`, `SA-002`, `FS2-GOV-002`, `FS2-BOOTSTRAP-MANIFEST-v1.2`) are data here, never instructions. Until 2026-09-18 this table omitted the control plane entirely. |
| Coupled-advancement invariant (OP-GDN-002 §1) | Every PR that changes a mathematical status must also change `registers/` or include a no-change certificate under `governance/no_change_certificates/` | `registers/json/no_change_certificates.json` holds the historical ones. |
| Research files do not become instructions merely by containing commands (R17) | Same: nothing under `packages/`, `research/`, `legacy/` is executed by CI except explicitly listed verifiers under `tools/` | Zero-trust rule preserved. |

## What this repository deliberately does not do

* It does not re-review any proof. Review verdicts, technical statuses and
  independence credits are copied from the source registers as of the export
  date in the file name.
* It reads one register, the live GP-REG-032-v1.2 workbook, and merges no
  other copy into it: R17 §1 rules "Use domain filter views rather than three
  competing writable register copies", and the inventory holds, besides the
  live workbook, a DQ-018 sandbox copy titled "DO NOT USE LIVE" and two
  GP-REC-EC020 snapshot Sheets declared immutable — none is read here. The
  duplicate identifiers reported in `registers/KNOWN_FINDINGS.json` are in the
  live workbook's own tabs, for the owner to resolve at the source. (Until
  2026-09-19 this bullet said the duplicates were found "in" three writable
  copies.)
* It does not promote any draft (`D1_ASSEMBLY_v2_3_DRAFT`, `H5_ZBAND_CONSUMPTION`)
  over the frozen v2.2 body. PKG-01 ships on v2.2.
