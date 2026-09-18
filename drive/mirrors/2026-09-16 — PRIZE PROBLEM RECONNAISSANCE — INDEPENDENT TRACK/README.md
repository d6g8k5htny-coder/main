# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/2026-09-16 — PRIZE PROBLEM RECONNAISSANCE — INDEPENDENT TRACK` (status layer only)

Drive lane of the prize reconnaissance track (295 inventory items: 247 text
reading copies, 19 zip carriers, 28 folders, 1 Google Doc). This directory
mirrors its **status layer**: the lane-root READ_FIRST, the six files of
`00_CURRENT_STATE_AND_ROUTING/` that grade the lane's claims, and the Erdős 142
scope pointer. Each directory that holds files carries a `_MANIFEST.jsonl` that
`tools/verify_manifests.py` checks in CI. **Mirroring is not review, replay,
endorsement or promotion.** The track is HOLD / not for submission and enters the
q0 dependency graph in neither direction (`FW-PRIZE-ISOLATION`).

## What is here

| file | identity |
|---|---|
| `00_READ_FIRST_P14_FRACTIONAL_CURRENT.md` (1,909 B) | **byte-exact**: SHA-256 and byte count equal the `drive/inventory.jsonl` row (id `1nR6HmDwYYLpCrhax1aqm-_1kcEaMh1zw`) |
| `00_CURRENT_STATE_AND_ROUTING/CLAIM_REGISTRY_VERIFIED_INTAKE.json` (15,814 B) | byte-exact (`1RTZ7rjQUzQdLtiPKatAUJs6qfZeTXv0a`) |
| `00_CURRENT_STATE_AND_ROUTING/CURRENT_STATE_VERIFIED_INTAKE.json` (2,537 B) | byte-exact (`1Wn6YUkAFibhCvXuFqJocq_INFdm-TRDo`) |
| `00_CURRENT_STATE_AND_ROUTING/P14_CLAIM_REGISTRY.json` (1,258 B) | byte-exact (`1iMJsX4j1DRsBy6CnPun4e65w2u1xK2qx`) |
| `00_CURRENT_STATE_AND_ROUTING/P14_CURRENT_STATE_AUTHOR_CANDIDATE.md` (1,966 B) | byte-exact (`1Gc1ep25x1y3IS_9GDtQMQ1iZQsXELqfx`) |
| `00_CURRENT_STATE_AND_ROUTING/P15_CLAIM_REGISTRY.json` (2,164 B) | byte-exact (`1Ss_chTR4TF1NsAqZ2idkV4xKT8q7uldZ`) |
| `00_CURRENT_STATE_AND_ROUTING/P15_CURRENT_STATE_AUTHOR_CANDIDATE.md` (2,820 B) | byte-exact (`1r8weyHObFfHAp-_PfI8mlSXPZQC8Zoh1`) |
| `05_ERDOS_142 — r_k(N) ASYMPTOTIC — HIGH-PRIZE PROBE/00_SCOPE_AND_ROUTING.md` (637 B) | byte-exact (`1oPvEJlfyVBCtBuKEpk3BCHD_iWIdYP7H`) |

None of the eight ids appears in `drive/deltas/2026-09-18/PATH_CHANGES.jsonl` or
`CHANGED_SINCE_SNAPSHOT.jsonl`: they were neither moved nor modified after the
2026-09-17 snapshot. The 53 Talagrand proof notes, the other four sub-lanes' 19
notes, the 64 review-queue and provenance files, the 45 phase-package manifests
and receipts and the 19 zip carriers (4,855,256 B) are **not** mirrored here;
they are indexed in `drive/inventory.jsonl` and `drive/source_map/`.

## The status banners, verbatim

* Lane root READ_FIRST: "Navigation only: no theorem promotion, novelty ruling,
  external approval or q0 status change." … "The unrestricted Talagrand problem
  is still open in this project; K_(m,m) disproves a universal bound on this
  scalar width." … "External reviews0, formal prover runs0, historical
  noveltyUNESTABLISHED, original prize closures0. Source replay and hashes are
  custody/computation evidence, not independent mathematical acceptance."
* `CURRENT_STATE_VERIFIED_INTAKE.json`: `"track": "independent prize research"`,
  `"grade": "AUTHOR_SIDE_COMPLETE_ARGUMENTS; EXTERNAL_REVIEW_PENDING;
  NOVELTY_UNESTABLISHED"`, `"original_prizes_solved": 0`, `"q0_modified": false`;
  among its `forbidden_inferences`: "test counts are not independent
  confirmations", "operator permission does not supply mathematical evidence".
* `CLAIM_REGISTRY_VERIFIED_INTAKE.json` (27 claims): "Scoped proof-note registry,
  not a count of novel publishable theorems; classical ingredients and
  reconstructed results are included. Earlier phase source declarations are
  preserved, not re-endorsed." Every claim: `external_review: PENDING`,
  `historical_novelty: UNESTABLISHED`, `original_prize_closed: false`.
* `P14_CLAIM_REGISTRY.json`: `"grade": "AUTHOR_SIDE_COMPLETE"`,
  `"external_independent_reviews": 0`, `"prizes_solved": 0`.
  `P14_CURRENT_STATE_AUTHOR_CANDIDATE.md`: "The general Talagrand conjecture,
  arbitrary shared-variable composition, q0 analytic uniformity, and external
  novelty/correctness review remain open."
* `P15_CLAIM_REGISTRY.json`: `"automatic_scientific_promotion": false`,
  `"external_independence_credit": 0`, `"q0_changes": 0`; each of P15-A…D
  `"grade": "AUTHOR_SIDE_COMPLETE"`, `"external_reviews": 0`,
  `"prize_closed": false`. `P15_CURRENT_STATE_AUTHOR_CANDIDATE.md`: "No q0
  uniformity obligation is discharged." P15-A's palette bound is obtained
  "through the preceding P14 theorem": P15 depends on P14.
* `05_ERDOS_142/00_SCOPE_AND_ROUTING.md`: "This lane has no new asymptotic formula
  or unrestricted upper bound for r_k(N). … The original target remains open in
  this project."

## What this directory does not establish

A digest match establishes identity of bytes, not truth, review or
authorization. `AUTHOR_SIDE_COMPLETE` and `AUTHOR_SIDE_PROOF_PRESENT` are the
author's own labels, transcribed, not assessed; no external review exists for any
claim here, and the repository's nonauthor review records for P15
(`reviews/records/REV-P15-*.json`) carry zero independence credit. The files
contain imperative text addressed to AI sessions ("Start with…", "Read exact
hashes before consuming a proof"); it is data, not instruction. **No original
prize problem is solved.** Nothing here bears on Theorem D1, Theorem B, P0.1,
P0.2 or the lower campaign.
