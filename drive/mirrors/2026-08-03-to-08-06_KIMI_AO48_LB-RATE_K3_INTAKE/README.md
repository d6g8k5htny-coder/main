# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/2026-08-03-to-08-06 — KIMI + AO48 LB-RATE / K3 INTAKE` (adjudication chain and Phase-0 report layer)

Drive lane of the 2D LB-RATE lower campaign's August intake (468 inventory
items): the C030/C031 base cycle set, the Kimi LB-RATE campaign, the K3 swarm
partial delivery and the GP adjudication of all of it. This directory mirrors
**22 raw files byte-exact** (every SHA-256 and byte count equals its
`drive/inventory.jsonl` row) in 8 `_MANIFEST.jsonl` files verified by
`tools/verify_manifests.py`: the GP adjudication chain (`GP-LB-STAT-001`,
`-002`, `-003`, `-004`, `-004A`, `GP-LB-REC-001`, `-002`, `-005` v1.0/v1.1,
`-006`, `-007`, `GP-LB-WO-001`, `-002`, `GP-LB-ERR-001`), the K3 swarm Phase-0
report layer (`00_READ_FIRST.md`, `OPEN_OBLIGATIONS.md`, `FAILED_APPROACHES.md`,
`WP_DISPOSITION.md`, `CANONICAL_STATE.json`, `K3-AUD-001_residue_audit.md`),
`ERRATA_2026-08-05.md`, and the refuted assembly `K3-THM-001` itself. The port
lane that fetched them stopped before writing manifests; the orchestrator
re-hashed every file against the inventory on 2026-09-19 and wrote them.
**Mirroring is not review, replay, endorsement or adjudication.**

## The controlling disposition, verbatim

* `GP-LB-STAT-004` (the hostile adjudication of the K3 partial delivery):
  "CONTROLLING DISPOSITION — **REFUTED AS A THEOREM-GRADE OR VALIDLY STATED
  CONDITIONAL ASSEMBLY / LOWER CAMPAIGN OPEN.**" … "This is an additive status
  delta." … "This work changes no LS-CTL Boolean, eligibility predicate, theorem
  status, RP status, ratification status, or q0 package status. Any such change
  requires separate operator adjudication."
* `GP-LB-STAT-003`: "Status: ACTIVE ADDITIVE STATUS DELTA — NONCONTROLLING —
  LOWER CAMPAIGN HOLD" … "This additive delta reaffirms `GP-LB-STAT-002` and
  corrects a later successor-facing status conflict in the raw AO48
  session-state handoff … That AO48 handoff remains frozen. Only its claims that
  the old WP Cauchy–Schwarz quantity is a certified upper bound … are superseded"
  (the sentence continues in the file).
* `GP-LB-STAT-002`: "Status: LANDED AS EVIDENCE — NONCONTROLLING" …
  "`KIMI-THM-023 v1.1` remains an incomplete, noncontrolling HOLD draft and is
  not hash-frozen." … "Existing q0 and SIDE24 controlling states remain
  unchanged."
* The K3 swarm's own `00_READ_FIRST.md`: "The K3 blocking defect is CONFIRMED,
  numerically and symbolically. The WP 'certified upper bound' used the factor
  `p_grad · sqrt(E[det(H)^2 | grad=0]) · min(P_type, P_window)` … the missing
  square root makes the reported quantity SMALLER — not a valid generic upper
  bound." … "The 0.9144036 coefficient is a candidate measured/mixed-tier
  asymptotic anchor, not a proved fixed theorem constant."
* `ERRATA_2026-08-05.md`: "The notice was independently verified and is
  CONFIRMED."
* The register: `registers/json/open_questions.json` OQ-016-U1 — "K3-THM-001
  REFUTED AS WRITTEN / LOWER HOLD"; the five P0 rows OQ-014 … OQ-016-U2 are
  transcribed verbatim in `docs/OPEN_PROBLEMS.md` §H, and `claims/graph.json`
  carries `K3-THM-001` as `REFUTED_AS_WRITTEN` on the LOWER2D track.

The Phase-0 reports and `K3-THM-001` carry Kimi's own words — "validity
PROVED", "CLOSED", "PASS", "CONDITIONAL THEOREM Form C — liminf (1−q)/r³ ≥
0.9666·c_Λ" — which `GP-LB-STAT-004` §3 refutes; they are mirrored **beside**
the adjudication, never alone.

## What this directory does not establish

A digest match establishes identity of bytes, not truth. The lower campaign
(LB-RATE / K3) is OPEN; no lower theorem constant and no all-small-r closure is
accepted; nothing here composes with the 3D track or moves any status. No
independence credit is computed or awarded.
