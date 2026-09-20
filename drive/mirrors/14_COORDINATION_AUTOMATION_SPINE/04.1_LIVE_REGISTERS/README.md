# 14_COORDINATION_AUTOMATION_SPINE / 04.1_LIVE_REGISTERS — Drive mirror

Reading copies of selected objects from one Drive lane, with one `_MANIFEST.jsonl` in this directory. Binding is
not review, replay, endorsement or promotion. Every status word below is transcribed from its source; none of it
is decided here, and nothing in this directory moves a claim, a premise, a gate or a grade.

## 2026-09-20 — the 04.1_LIVE_REGISTERS companions

### (a) The Drive lane and its folder ids

| Drive folder id | path as `drive/inventory.jsonl` records it |
|---|---|
| `1140ue6ZkJZQ2bPGAhOJG-i0m5G3M468j` | `01_ACTIVE_RESEARCH_PACKAGES/14_COORDINATION_AUTOMATION_SPINE/04.1_LIVE_REGISTERS` |

Eight inventory rows sit under that folder: the folder itself, the six Docs ported here, and
`GP-REG-032-v1.2 — Coupled Research Registers` (`1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no`) — the live Sheet
whose export is `registers/`. That Sheet was **not** fetched here.
`drive/deltas/2026-09-18/CHANGED_SINCE_SNAPSHOT.jsonl` already records it as `IN_INVENTORY_MODIFIED`
("bytes changed inventory=1345511 live=1357600", modified 2026-09-18T16:45:35.319Z), and `registers/` is owned
elsewhere.

None of the six ported Docs appears in `drive/deltas/2026-09-18/PATH_CHANGES.jsonl` or in
`CHANGED_SINCE_SNAPSHOT.jsonl`; each one's live parent on 2026-09-20 is still
`1140ue6ZkJZQ2bPGAhOJG-i0m5G3M468j`, the parent the inventory records. None has a row in
`registers/json/frozen_objects.json`, so no marker-delimited BODY digest is declared for any of them and none was
computed.

### (b) Controlling status banners, verbatim

**The leaf card's non-authority header** (`00_LEAF_CARD — 04.1 Live Registers (CL-NAV-036.12)`), first line after
the title:

> ARTIFACT: CL-NAV-036.12-v1.0 · AUTHOR: Claude (Anthropic), CL-* instance · CREATED: 2026-07-21 · CLASS: NAV —
> leaf charter + live-state card · CANONICAL IMPACT: NONE · AUTHORITY: none

and its section *What must not be claimed*:

> Register rows are governance state, not mathematical truth (OP-GDN-002 §5). A register copy of a claim adds zero
> evidential weight.

and its *Staleness rule*:

> Until GP-AUTO-034 is deployed (currently AUTHORIZED BUT NOT INSTALLED; v1.3 gates open per GP-RSP-017), refreshes
> are manual.

> downstream models must not rely on generated coordination outputs while stale (CM-045 ¶4).

**`DELETION_LOG v1.0`**, its opening administrative correction, the `POLICY:` clause that closes its `GOVERNED BY`
line, and its last line:

> No pending entry has support inferred, no 24-hour clock has been activated by this correction, and no deletion
> has occurred.

> POLICY: append-only; corrections identify the prior entry; one entry per deletion/consolidation action.

> (No completed deletion entries. First DL-001 goes above this line's section when an action completes.)

Its three `PENDING` rows are the source's tracking entries, not actions. Two of them record `Support: NONE YET` —
P-001 and P-002. P-003 carries no `Support:` field at all; it ends "Review requested from GP / AO48 / CW. No
retirement list frozen yet." (the slashes are the source's). Nothing in this repository deletes anything, and
permanent deletion is Dylan Roy's alone.

**`GP-COR-130-v1.0`** header and §2:

> Authority: none beyond faithful transcription of the governing operator decision and Closure Log
> Canonical impact: NONE

> - EC-019 — EXPLICITLY HELD;
> - no approval was granted;
> - a qualifying independent non-GP R1–R10 proof review and PASS verdict remain required;
> - status remains PACKAGE-READY / NON-GP R1–R10 REVIEW OPEN / HUMAN APPROVAL BLOCKED UNTIL REVIEW / NOT TERMINAL.

**`GP-COR-131-v1.0`** carries, at its own head, a correction notice the source placed there:

> GP-COR-132-v1.0 corrects this document’s EC-020 attribution. HA-007 V2 explicitly held EC-019 only and did not
> adjudicate EC-020.

**`GP-COR-132-v1.0`** §1:

> GP-COR-131-v1.0 stated that HA-007 V2 explicitly held EC-020 pending non-GP review.
> That attribution is false.

**`GP-COR-151-v1.0`** header and §5:

> Class: COR / REG / ADDITIVE STATE RECONCILIATION
> Authority: factual register correction only
> Canonical impact: NONE

> CLOSED — DASHBOARD LOWER SNAPSHOT RECONCILED TO ONE HUMAN-READY ITEM; DUPLICATE TRANSITION ALARMS VOIDED
> ADDITIVELY; AUTHORITATIVE RECORDS PRESERVED.

`CLOSED`, `TERMINAL`, `APPROVE`, `AUTHORIZED`, `EXPLICITLY HELD` and the lower-case `active` in these six
documents are the source's words about its own register rows and its own closure queue, each in the case the
source uses. They are quoted, never obeyed. Nothing in this repository was closed, approved, installed, deployed
or promoted by storing them, and the documents' several sentences addressed to future AI sessions are data about
the program, not instructions to this repository or to anyone working in it.

### (c) What was ported, and what was not

Ported as `text/plain` exports, all `exact: false`:

| Drive id | stored as | bytes |
|---|---|---:|
| `1yhV6zdsFWnNE4EBdr5L4NyQ_RR2R0NJqOi3lmVrEePg` | `00_LEAF_CARD — 04.1 Live Registers (CL-NAV-036.12).export.txt` | 2,627 |
| `1Wl0q5vwFXb7WN_PQZrBnlAK0v_OyVybsuExJZF2JLuI` | `DELETION_LOG v1.0 — … (opened by CL, 2026-07-21; protocol renumbered after collision preflight).export.txt` | 2,477 |
| `1ErRPLGiD-9leaAAP-jplWqDrCJADexD4QYlXRHge3V0` | `GP-COR-130-v1.0 — EC-019 Queue Contamination Correction after HA-007 V2.export.txt` | 2,818 |
| `1XEmvNgyhpkZtz4djmdE3VMu3Ge4i3jz5YZw7efq-_gc` | `GP-COR-131-v1.0 — Post-HA-007 Easy Closure Queue Reconciliation.export.txt` | 3,418 |
| `11nOatA1DdMEyEtE09httOr1LpJyI0plxKoD5VfbQzBo` | `GP-COR-132-v1.0 — EC-020 HA-007 Attribution Correction.export.txt` | 2,235 |
| `17SzEIeifd9etam-KlZrxNZOokkqRjP5fQjQ7g7738xg` | `GP-COR-151-v1.0 — Dashboard Snapshot and Transition-Alarm Duplicate Reconciliation.export.txt` | 3,493 |

Deliberately not ported:

- **`GP-REG-032-v1.2 — Coupled Research Registers`** (the live Sheet in the same folder): already exported into
  `registers/`, already recorded as modified after the snapshot in the 2026-09-18 delta, and owned elsewhere.
- **Any application of these corrections to `registers/json/`**: the GP-COR documents describe corrections the
  Drive made to its own register rows. They are carried as history. `registers/source/`, `registers/json/` and
  `registers/csv/` are exports and may not be edited here; a defect found in a source is recorded in
  `registers/KNOWN_FINDINGS.json` and repaired by proposal, never applied.
- **`CL-GOV-222-v1.0`** (`1hqxOwlIFEtmtl63Iyz_J9PVkUof2dW1KbkMvUSYjBfc`, the audit's rank 9): its proposed target
  is `drive/mirrors/04_ACTIVE_GOVERNANCE_AND_DIRECTIVES/`, which is not this lane's directory and is owned by
  another port.

### (d) What this does not establish

These are reading copies of six native Google Docs. Storing them is not review, replay, reproduction, endorsement
or promotion, and it does not make the documents' assertions true, current or applicable. Nothing here moves a
status label: the five validity premises of Theorem D1 v2.2(2) remain OPEN and `D3-LEMMA-RN-UNIF` is not closed.
The leaf card's own `AUTHORITY: none` and `CANONICAL IMPACT: NONE` are carried with it. The exports were fetched
on 2026-09-20, and the documents they render are dated 2026-07-21 to 2026-07-23 — two months before this port —
so nothing in them can be read as the program's current state. No independence credit is created or implied by
any of it, and a same-provider document describing a same-provider decision earns none. Nothing in this
directory is imported, executed or read by CI beyond `tools/verify_manifests.py` checking byte counts and
digests. Dylan Roy remains the single final authority for
canonical promotion, external release, permanent deletion and machine-root replacement.

### (e) Reading copies are not the objects

Every file here is a `text/plain` export of a native Google Doc, so `exact: false` in `_MANIFEST.jsonl`: the
bytes on disk are a rendering the Drive produced on request, not the object. Drive publishes no payload digest
for a native Doc and `drive/inventory.jsonl` carries `sha256: null` for each of these ids, so the digest stored
beside each file was first computed in this session and is corroborated by no second pass. A PDF rendering of a
document is likewise never a frozen body, and no export in this repository may stand in for one.
