# How the Drive's operating architecture maps onto this repository

The Google Drive research share runs under a family of operator-issued
protocols (OP-PROT-006, -011, -012, -019/R17; OP-GDN-002; OP-CNS-001). They were
written for a Drive + Google Sheets substrate. This document states, construct by
construct, what the git-native equivalent is in this repository, and where the
substrate changes the guarantees.

Nothing here promotes, closes, or reclassifies any mathematical claim. Status
labels are carried from the source verbatim (see `registers/`).

The one operator sentence in the corpus about a Git repository is the
FORMALIZATION BOARD's OPERATOR PACKAGE DECISION of 2026-07-24 (Drive
`10o4YRYOr8a2fB6rtnFnzMn7HkQMfv9-FZ5Mh0L-KF_o`): "REPOSITORY ROUTING: private Git
repository creation and exact-history push are approved when platform access
becomes available." It approves a *private* repository as the formalization,
source-control, reproducibility and CI-handoff system of record and, in the
same decision, "NOT APPROVED: blanket terminalization or mathematical promotion
of any included claim." This repository is public; whether to keep it so is the
owner's decision, and the 2026-09-18 execution-contract draft records the owner
asking to restrict who works in it, not to hide it.

| Drive construct (protocol) | Repository equivalent | Notes on guarantees |
|---|---|---|
| Stable Drive file ID as object identity (DEF-001) | Path in this repo **plus** SHA-256 in a manifest; `drive/inventory.jsonl` maps every Drive ID to title/parent/path | Git content addressing is stronger than Drive IDs: a blob hash is the identity, and history is immutable. The Drive ID is retained as provenance metadata. |
| Frozen objects register (byte count + SHA-256, "no in-place edit; numbered successor") | `registers/json/frozen_objects.json` (188 rows, an export) cross-checked by `tools/frozen_check.py` against the digests the accessibility source map recorded independently for the same Drive IDs | Offline and partial by construction: the 55 rows frozen by whole-file digest (classes A/D) are comparable and all agree; the 133 marker-delimited or export-body rows (classes B/C) are not comparable to a native Doc's size and are reported as such, 17 of them with the body present byte-exact in the payload index. Bodies this repository holds byte-exact (`engine/rn_engine/frozen/`, `engine/carriers/blobs/`) are verified by their own manifests. Nothing re-freezes anything. Until 2026-09-18 this row claimed CI failed on frozen-byte drift while no checker existed. |
| Work Events (append-only coordination log, AppendCells only) | `registers/json/work_events.json` guarded by `tests/test_registers.py::test_work_events_append_only` (rows may only be appended; CI diffs against the parent commit) | Git commits are themselves an append-only, hash-chained event log with author identity; PRs are the claim/publication surface. |
| Claim lease (120-minute, cooperative, not compare-and-swap) | A branch per claim; the PR is the claim; merge is compare-and-swap on the base branch head | Git gives the transactional guarantee the protocol says Sheets cannot: a push that races is rejected, never silently overwritten. |
| Publication conflict (two successors of one head → hold and reconcile) | Merge conflict; resolved by a reconciliation commit that preserves both branches in history | Same semantics, machine-enforced. |
| DRAFT → CANDIDATE_VERIFIED → READY_FOR_REVIEW → REVIEWED/AMEND | `sandbox/` (no authority) → PR opened as draft → PR marked ready → review verdict recorded in `registers/` and merged | Draft PRs carry no canonical authority, mirroring `06_SANDBOX_FRONTIER`. |
| Review Queue with 7/14/30-day aging ladder | `registers/json/review_queue.json`; suggested: one GitHub issue per review key, labels `age:7`, `age:14`, `age:30` | Aging is prioritisation only; it never approves. |
| Technical status vs organizational independence (four separate dimensions) | Review verdict files record correctness, scope, authorship/exposure and independence separately; PR approval from the same provider family earns zero independence credit | Unchanged rule; git does not weaken it. |
| Quarantine (EXACT_DUPLICATE / SUPERSEDED / DEFECTIVE_SCOPE / UNVERIFIED-CONFLICT / LEGACY_INSPIRATION) | `quarantine/` directory with `registers/json/quarantine_index.json`; moves are commits with the original path recorded (rollback record) | No permanent deletion: git history retains every byte. |
| Logical quarantine of frozen archive members (carrier ID + relative path + hash exclusion) | `quarantine/EXCLUSIONS.json` listing archive, member path and SHA-256; verifiers refuse to consume excluded members | Same as the Drive's path-and-hash exclusion. |
| DO_NOT_OPEN vault (superseded mirrors, dead ends, trap copies) | **Not mirrored.** Only the vault's metadata tree is recorded in `drive/vault_tree.txt` (7 items: id, kind, byte count, path, from the inventory; no digest exists for any of them and none was computed); contents are not in this repository and were never opened | Respects the standing order that models must not consume vault contents for authority. The file referenced here did not exist until 2026-09-18. |
| Zero evidentiary authority / inspiration-only lanes | `legacy/` with the same banners | Nothing under `legacy/` may be cited as evidence. |
| Handoff envelope (payload + payload_sha256, noncircular) | The program's envelopes arrive as Drive objects and are mirrored byte-exact under `drive/deltas/<date>/<folder>/` (2026-09-18: `ACCESSIBILITY_HANDOFF.json`, `DRIVE_STRUCTURE_HANDOFF.json`, the DG-EXEC `DELIVERY_MANIFEST.json` and `READBACK_RECEIPT.json`); embedded payload digests were recomputed on fetch, and every mirrored blob is verified by `tools/verify_manifests.py` through the folder's `_MANIFEST.jsonl` | Same schema; hash the payload only. Until 2026-09-18 this row named a `drive/handoffs/` directory that did not exist. |
| Delivery manifest per bundle (one manifest may cover many payloads) | `_MANIFEST.jsonl` per mirrored folder under `drive/deltas/<date>/` (dest, bytes, sha256, exact, stored, not_stored_reason), verified in CI; the `MANIFEST.sha256` files some Drive packages ship (the KIMI normalized export, for one) are indexed by the inventory and not yet mirrored | Verified in CI where mirrored. Before 2026-09-18 no manifest of either kind existed in this repository, so this row described a check that never ran. |
| Autonomy classes 0–5 (OP-PROT-012) | Class 0–1 (read, map, move, rename) are ordinary commits; Class 2–3 (terminal state, theorem promotion) require the predicates in the protocol **and** a recorded verdict in `registers/`; Class 4 (external release) stays DISABLED; Class 5 (permanent destruction) is impossible in git without history rewrite, which this repo forbids | Branch protection should forbid force-push and history rewrite on `main`. |
| Coupled-advancement invariant (OP-GDN-002 §1) | Every PR that changes a mathematical status must also change `registers/` or include a no-change certificate under `governance/no_change_certificates/` | `registers/json/no_change_certificates.json` holds the historical ones. |
| Research files do not become instructions merely by containing commands (R17) | Same: nothing under `packages/`, `research/`, `legacy/` is executed by CI except explicitly listed verifiers under `tools/` | Zero-trust rule preserved. |

## What this repository deliberately does not do

* It does not re-review any proof. Review verdicts, technical statuses and
  independence credits are copied from the source registers as of the export
  date in the file name.
* It does not merge the Drive's three writable register copies or reconcile the
  duplicate identifiers found in them; those are reported in
  `registers/KNOWN_FINDINGS.json` for the owner to resolve at the source.
* It does not promote any draft (`D1_ASSEMBLY_v2_3_DRAFT`, `H5_ZBAND_CONSUMPTION`)
  over the frozen v2.2 body. PKG-01 ships on v2.2.
