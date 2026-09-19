# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/2026-09-16 — PRIZE PROBLEM RECONNAISSANCE — INDEPENDENT TRACK`

Drive lane of the prize reconnaissance track (295 inventory items: 247 text
reading copies, 19 zip carriers, 28 folders, 1 Google Doc). On 2026-09-18 this
directory mirrored its **status layer** — the lane-root READ_FIRST, the six files of
`00_CURRENT_STATE_AND_ROUTING/` that grade the lane's claims, and the Erdős 142
scope pointer — and this first section describes that layer; on 2026-09-19 the rest of
the lane's raw files were added (the section of that date, below), so every raw file
of the lane is now stored byte-exact except the 19 zip carriers and the 3 quarantined
items (tree-only rows) and 9 byte-identical duplicates (pointer rows to the stored copy). Each directory that holds files carries a `_MANIFEST.jsonl` that
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
2026-09-17 snapshot. Until 2026-09-19 this paragraph said that the 53 Talagrand proof
notes, the other four sub-lanes' 19 notes, the 64 review-queue and provenance files
and the 45 phase-package manifests and receipts were not mirrored here; they now are
(2026-09-19 section). The 19 zip carriers (4,855,256 B) remain tree-only rows, indexed
in `drive/inventory.jsonl` and `drive/source_map/`.

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
* `CLAIM_REGISTRY_VERIFIED_INTAKE.json` (28 claims): "Scoped proof-note registry,
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

## 2026-09-19 — the rest of the lane: 228 raw files stored byte-exact, 31 tree-only rows

This section is additive as to bytes and manifest rows: none of the eight files the
first section describes and none of the three manifests' earlier rows was changed; new
rows were appended and 24 new `_MANIFEST.jsonl` files were created. The first
section's own prose was amended the same day so that it no longer says the rest of the
lane is not mirrored.

**Lane and folder ids.** Drive
`01_ACTIVE_RESEARCH_PACKAGES/2026-09-16 — PRIZE PROBLEM RECONNAISSANCE — INDEPENDENT TRACK`,
folder id `1AyEOZP1C6ebBx7wxZoIT_RIbP2noYuJ5` (295 inventory items). Sub-folders:
`00_CURRENT_STATE_AND_ROUTING` `1XpShdTWW_KTtYMq_ZET7rM5rb5Ido1qI`;
`01_ERDOS_3_AND_169 — AP_HARMONIC_RESEARCH` `16btLGxPbv4LO1_x1YzsznckncFhEtoAe`;
`02_ERDOS_39 — INFINITE SIDON — STRUCTURED CONSTRUCTION` `1kSVL6SN2saniZZnQytO5zv6vgqBQPpV8`;
`03_RIEMANN — MILLENNIUM MOONSHOT + ADJACENT PROGRESS` `1NpQDOoR0rY0EmIvv8YKUkAUixYsJ5BZa`;
`04_COLLATZ — 120M JPY MOONSHOT — STRUCTURAL PROBE` `1BU8r2uTWncuL9QHZicwsxSNpz3E-oSP6`;
`05_ERDOS_142 — r_k(N) ASYMPTOTIC — HIGH-PRIZE PROBE` `15xLEqpHFSpvyro7CFmAqE7KrW4AHU7RR`;
`06_TALAGRAND_DISCRETE — RESTRICTED_PROOF_CANDIDATES` `1z4M_wmypHJ4jAjA3hGtonzf7PRLgZRh5`;
`90_FROZEN_PHASE_PACKAGES` `1Ei8yXjV513WUKVR-VnSy3vLYdz1fN5LS` with `PHASE01`
`1oUQaxn5s8oLimAzHuvBcLuitsRf1-HAi`, `PHASE02` `1rHM_-nbWo5YeoM7F0OCOIg_BFNM0eiJE`, `PHASE03`
`1eGMUZZCLI5fbWG3W7Rh9jlgAC2czqnyk`, `PHASE04` `16Zqvc_oSs856wZoyXYvjaU_ZqXtfe-j8`, `PHASE05`
`13iv3M2Pn1GCpw6CMN729gtN0c8Wt2F1y`, `PHASE06` `1Hsn1SP9xaG-KRoIUu06CMMYQzWD36Plx`, `PHASE07`
`1jTOkZVOfpWteWZHSpmi62mo0pN8N2TYT`, `PHASE08` `1SEyIArKRmWqbFWq6SI1mz6PPlht86dpj`, `PHASE09`
`1BkSn0R8gftkUlcrxiEFQ8YOvXiuX2Zn-`, `PHASE10` `1Lq-O44mcB2lTp3M3tSqdMfq6WCsjLy-x`, `PHASE11`
`1fT-ZzJ4E4NthMSTmzxtHCPO9yxkhXwLO` (its `ATTACHED_ee4a16e1 — DISTINCT_AUTHOR_SNAPSHOT`
`1jL2CO4vN_ZOzhgCufrfycdw6HMWquzHH` and `CONTINUATION_AUDIT_a7f03b3f_20260917`
`1eXR0n6z4AfLIDfJYDONvg6MuIOko6dII`), `PHASE12` `10bnkkUWLuTNVrwgLEcu2tP6w_C9ReL7E`, `PHASE13`
`1r1WurdciCLYATEaSpLY4sSibhjhBzcjw`, `PHASE14` `12S1izgqYTpnzX4O2KDi4-YQbYW_vy-NV`, `PHASE15`
`1RGq8L05x8fEuXd1vMfI1hqDvl4uqACX3`; `91_REVIEW_QUEUE_AND_PROVENANCE`
`1BBlXttbAovjxUtKFyKKP27L52VQgK_5A`; `EXTERNAL_REVIEW_PR_TAL_003-008`
`1wCojptlhvMs9kfjnaNjVAiGTJy4YkiEz`.

**Method and identity rule.** Every raw file's bytes were fetched through the Drive
connector on 2026-09-19 (`download_file_content`, base64), decoded from the session
transcript on disk, hashed, and stored only where the SHA-256 and byte count equal the
`drive/inventory.jsonl` row for that id. All 237 raw files fetched matched (0
`HASH_MISMATCH`); 228 are stored (944,074 B, budget 8,000,000 B, largest file 22,429 B)
and 9 byte-identical duplicates in `EXTERNAL_REVIEW_PR_TAL_003-008/` are pointer rows
(below). None of the lane's 295 ids appears in `drive/deltas/2026-09-18/PATH_CHANGES.jsonl`
or `CHANGED_SINCE_SNAPSHOT.jsonl`, so every file is stored under its snapshot path.
`python3 tools/verify_manifests.py "<this directory>"` → `manifests=27 verified=236
problems=0`. The 17 stored files titled `*.sha256` are on disk as `<title>.sha256.txt`
with bytes unchanged: the checker parses every `*.sha256` as a sha256sum manifest relative
to its directory, and these list members of archives that are not on disk here (the same
convention the other mirror lanes use; each row's note says so). Every row records the
inventory digest beside the recomputed one. No `exact: false` row was added.

### (1) `00_CURRENT_STATE_AND_ROUTING/` — 42 files stored (84,711 B), 2 tree-only rows

Per-phase `CURRENT_STATE` / `NEXT_WORK` / `CLAIM_REGISTRY` / `ROUTING` / `METRICS` files
for P05–P15, `ROUTING_LEDGER_VERIFIED_INTAKE.json`, the two `DRIVE_READBACK_PHASE05_OVERLAP*`
receipts, the P11 crosswalk (`P11_SNAPSHOT_CROSSWALK.json/.md`),
`P11_PORTFOLIO_CLOSURE_MAP.md`, `P11_AUDIT_ADMISSION_20260917.md`, and the P11
`ATTACHED_ee4a16e1` state files. `HISTORICAL — P07_CURRENT_STATE pre-PR-TAL-015.md` and
`P07_CURRENT_STATE_v2.md` are byte-identical (distinct ids; both stored). The two
`QUARANTINE_OR_HISTORY` files `P10_ROUTING — SUPERSEDED NAVIGATION.md` and
`P11_860411bc_ROUTING — SUPERSEDED_NAVIGATION_ONLY.md` are tree-only rows and are not ported.

Status banners, verbatim:

* `ROUTING_LEDGER_VERIFIED_INTAKE.json`: `"no_deletes": true`, `"no_sharing_changes": true`,
  `"no_theorem_promotions": true`, `"no_external_independence_credit": true`, generic
  manifest `"grade": "custody only"`.
* `P11_SNAPSHOT_CROSSWALK.md`: "Two distinct packages used the same phase/archive name.
  Both original manifests verify." … "Neither package is overwritten. A smaller displayed
  palette is not a substitute for reviewing the exact assumptions and proof." … "No source
  of concurrent production, independence, novelty, or review approval is inferred from the
  presence of two packages." … "Both are author-side candidates."
* `P11_RECONCILED_ROUTING — HISTORICAL_NAVIGATION.md`, "Scientific boundary": "Both
  snapshots contain author-side proofs with external review and novelty unresolved. No
  original prize is solved. Finite test counts, byte identity, and folder membership are
  not mathematical approval. No q0/Claude work, P0.1/P0.2 gate, SIDE24 scope, or sharing
  permission is altered."
* `P11_PORTFOLIO_CLOSURE_MAP.md` — "closure" in the title is the source's word for a
  triage table: "This is a continuation snapshot, NOT a full re-review of every Drive
  proof." … "No q0 theorem, controlling assembly, registry Boolean, or Claude-owned active
  artifact was edited. Broad inventories are source-routing evidence, not proof validation."
* `P11_AUDIT_ADMISSION_20260917.md`: "No frozen body overwritten; no external review or
  novelty credit added; q0 and Claude scope unchanged."
* `P09_ADDITIVE_ROUTING.md`: "The only current deployment claim for Phase09 is
  PREPARED_NOT_APPLIED. A later executor must create its own actual write/readback receipt."
* `P07_CURRENT_STATE_v4.md`: "No original prize conjecture is closed. No external review has
  occurred."
* `P13_CLAIM_REGISTRY.json`: `"grade": "AUTHOR_SIDE_PROOF_CANDIDATE"`,
  `"external_independent_reviews": 0`, `"historical_novelty": "UNESTABLISHED"`,
  `"original_prize_closed": false`.

### (2) `06_TALAGRAND_DISCRETE — RESTRICTED_PROOF_CANDIDATES/` — all 53 proof notes stored (258,787 B)

`P04_04 … P05_08`, `P07_PR_TAL_015 … 018`, `P08_PR_TAL_019`, `P09-A … P09-G`, `P10-A … D`,
`P11-A … D`, the five `P11_ATTACHED_ee4a16e1_*` notes, `P11_D_ZERO_Q_ENDPOINT_ADDENDUM_a7f03b3f.md`,
`P12-A … F`, `P13-W1 … W3`, `P14-A … DE`, `P15-A … D`. **Every note is graded author-side by its own registry, with a zero external-review count and
novelty unestablished, in that registry's own field names**, which differ from file to file:
`CLAIM_REGISTRY_VERIFIED_INTAKE.json` (28 claims) — `declared_proof_grade`
`AUTHOR_SIDE_PROOF_PRESENT` ×26, `SOURCE_DECLARED_AUTHOR_ARGUMENT` ×1,
`SOURCE_DECLARED_FINITE_CERTIFICATE_NOTE` ×1; every claim `external_review: PENDING`,
`historical_novelty: UNESTABLISHED`, `original_prize_closed: false`. `P09_CLAIM_REGISTRY.json` —
`grade` `AUTHOR_SIDE_CANDIDATE` ×6 and `SCOPE_AND_PRIOR_WORK_RECORD` ×1 (P09-E); `external_reviews: 0`,
`novelty: UNESTABLISHED`, `original_prize_closures: 0`. `P10` — `AUTHOR_SIDE_COMPLETE_WRITTEN_ARGUMENT`
×4; `external_reviews: 0`, `novelty: UNESTABLISHED`, `original_prizes_solved: 0`. `P11` —
`AUTHOR_DERIVED_COMPLETE_PENDING_EXTERNAL_REVIEW` ×4; `external_independent_reviews: 0`,
`novelty: UNESTABLISHED`, `original_prize_closed: false`. `P11_ATTACHED_ee4a16e1` — `status`
`LOCAL_AUTHOR_SIDE_COMPLETE` ×4 and `SOURCE_GROUNDED_TRIAGE_ONLY` ×1; `external_review: PENDING`,
`novelty: UNESTABLISHED`, `original_prize_closed: false`. `P12` — `AUTHOR_SIDE_COMPLETE` ×6;
`external_review: false`, `novelty: UNESTABLISHED`, `original_prizes_solved: 0`. `P13` —
`AUTHOR_SIDE_PROOF_CANDIDATE` ×3; `external_independent_reviews: 0`, `historical_novelty:
UNESTABLISHED`, `original_prize_closed: false`. `P14` — `AUTHOR_SIDE_COMPLETE`;
`external_independent_reviews: 0`, `historical_novelty: UNESTABLISHED`, `prizes_solved: 0`. `P15` —
`AUTHOR_SIDE_COMPLETE` ×4; `external_reviews: 0`, `historical_novelty: UNESTABLISHED`. (Until
2026-09-19 this sentence paraphrased all of them as `AUTHOR_SIDE_*` / `external_review: PENDING` /
`historical_novelty: UNESTABLISHED`; no actual value is stronger than that paraphrase, and none is
literally it.) **Theorem-like titles**
(`…_THEOREM`, `…_HAZARD_CLOSURE`, `UNIVERSAL_CAPACITY_ONE`, `EXACT_BLOCK_COMPRESSION`) **are
the author's**, transcribed, not assessed. Verbatim:

* `P04_04_laminar_talagrand.md`: "Complete author-side argument below with exact finite
  companions. External review and historical novelty are UNESTABLISHED. This is a structural
  subclass theorem, NOT the full discrete Talagrand conjecture."
* `P05_01_five_copy_laminar.md`: "Complete author-side derivation, NOT independently
  reviewed. Historical novelty UNESTABLISHED. This is a restricted subclass of the discrete
  Convexity Conjecture, not a prize solution."
* `P15-A_CRITICAL_THRESHOLD_CROSSWALK.md`: "Status: author-side proof of the translation, not
  a claim of a new parameter."
* `P11_D_ZERO_Q_ENDPOINT_ADDENDUM_a7f03b3f.md`: "Nonblocking endpoint amendment for the
  attached P11-D" — an amendment note, filed beside the note it amends, not a replacement.

**The four P15 bodies are the ones `reviews/records/REV-P15-A..D.json` are bound to.** The
stored file's recomputed digest equals each record's `object_sha256` and the matching
`proofs/0N_*.md` line of `90_FROZEN_PHASE_PACKAGES/PHASE15/MANIFEST.sha256.txt`:

| review record | stored note | bytes | SHA-256 |
|---|---|---:|---|
| `REV-P15-A` | `P15-A_CRITICAL_THRESHOLD_CROSSWALK.md` | 4,895 | `93e2ead4f4ff3a5faf75f96df0a7083b3f88939e50622d27ec7ee65b2b919c93` |
| `REV-P15-B` | `P15-B_PALETTE_SEPARATED_LOCALIZATION.md` | 6,286 | `9b18b6e9abc90d18deef06ab12e3aa7794dad40e1618d99daa88e369c300e8c3` |
| `REV-P15-C` | `P15-C_CONNECTED_UNBOUNDED_WIDTH_FAMILY.md` | 5,796 | `357dd57302a2ec6c9e98f454ac847a589e45f0db579df64f6cc102545b35429d` |
| `REV-P15-D` | `P15-D_COMPLETE_TRANSVERSAL_GRAPH_COMPOSITION.md` | 7,710 | `ad2c55b8609781d8c748cc93db1f297bf13c534667f60aac86ab59e8230ca31c` |

A digest match binds the review record to these bytes; it does not import the record's
verdict, and those records carry zero independence credit (same provider).

### (3) The four small sub-lanes — 18 notes stored (95,429 B)

`01_ERDOS_3_AND_169 — AP_HARMONIC_RESEARCH` (15 notes, `P01_01 … P04_03`, 88,082 B),
`02_ERDOS_39 …` (`P01_03_sidon.md`), `03_RIEMANN …` (`P01_05_spectral_moments.md`),
`04_COLLATZ …` (`P01_04_collatz.md`). **The folder names "MILLENNIUM MOONSHOT" and "120M JPY
MOONSHOT" are aspirations — the source's target labels — not claims**; the notes inside say
so themselves:

* `P01_01_erdos3_digital.md`: "Grade: complete author-side derivations with exact
  computational companions; not independently reviewed, not a solution of a prize problem.
  Novelty is not established."
* `P04_01_weighted_extremizers.md`: "This does not prove finiteness of the unrestricted
  harmonic four-progression supremum."
* `P01_03_sidon.md`: "Complete author-side elementary derivation. Not a solution of Erdős #39;
  no novelty claimed."
* `P01_05_spectral_moments.md`: "standard moment-nonuniqueness phenomenon, not a new Riemann
  Hypothesis result and not a claimed barrier for every RH method."
* `P01_04_collatz.md`: "Complete elementary author-side proofs; known-type obstruction, no
  novelty or solution claim."
* `HISTORICAL — PHASE01-05 VERIFIED ROUTING.md` on these folders: "- **03_RIEMANN — MILLENNIUM MOONSHOT + ADJACENT PROGRESS**: generic finite-moment limitation only; no zeta-specific theorem proved here." … "- **04_COLLATZ — 120M JPY MOONSHOT — STRUCTURAL PROBE**: restricted descent-certificate obstructions only. The monetary folder wording is historical reconnaissance, not a newly verified offer." (until 2026-09-19 this README dropped the bold markers and cut the Collatz sentence after "obstruction").

### (4) `91_REVIEW_QUEUE_AND_PROVENANCE/` — 63 files stored (147,790 B), 1 tree-only row

Source registers (`P01 … P15_SOURCE_REGISTER.md`), review packets (`P02 … P15_REVIEW_PACKET.md`,
`P04_REVIEW_AND_SCOPE_AUDIT.md`), recon memos with their `.sha256` / `RECON_HASH*.json` /
`.sha256.json` sidecars, development notes, `P08_INTAKE_AND_ASSERT_AUDIT.json`,
`P11_CONTINUATION_AUDIT_REPORT_a7f03b3f.md` and `HISTORICAL — PHASE01-05 VERIFIED ROUTING.md`.
The native Google Doc `HISTORICAL — Prize Reconnaissance v0.1 — superseded routing`
(`QUARANTINE_OR_HISTORY`) is a tree-only row: not exported, not ported. **These are custody
records, not verdicts.** "OpenAI/ChatGPT exposed author-side work" is the source's own
attribution of the recon memos (`P15_PALETTE_RECON_20260917.md`: "Policy
OP-RECON-20260916-v1.0; mode FRESH; OpenAI/ChatGPT exposed author-side work; UTC 2026-09-17.").
Each of the 13 hash sidecars names a digest equal to the inventory digest of a stored file
in this folder (identity of the memo it names, nothing more). Verbatim:

* `HISTORICAL — PHASE01-05 VERIFIED ROUTING.md`: "No original prize problem has been solved.
  The stored results are restricted author-side mathematical arguments, exact computations,
  and failed-route records. External independent review and historical novelty assessment
  remain open." … "That is custody verification, not independent mathematical approval."
* `P04_SOURCE_REGISTER.md`: "Hash verification confirms identities, not truth."
* `P03_REVIEW_PACKET.md`: "Current grade is author-side complete, not independently
  endorsed. Return separate mathematical/custody/numerical judgments; do not convert test
  counts into truth probabilities."
* `P05_OVERLAP_REVIEW_PACKET.md`: "This is a single-author implementation/proof package;
  ordinary/optimized replay is not independent review." … "No number of passing checks is to
  be translated into a confidence percentage."
* `P11_CONTINUATION_AUDIT_REPORT_a7f03b3f.md`: "All183 attached payload hashes verify. The
  separate stored Phase11 snapshot has163 verified payloads and a different SHA256. Neither
  snapshot was overwritten or conflated." … "This audit made ZERO successful new remote
  writes".
* `P11_ATTACHED_ee4a16e1_P11_RECON_SCOPE_DELTA.md`: "Disposition: LOCAL_PROVISIONAL;
  HOLD_RECON_CUSTODY for durable project admission until the staged memo and sources have
  actual remote write/readback. No remote work lease or scientific status is inferred."

### (5) `90_FROZEN_PHASE_PACKAGES/` — 45 manifests, receipts and readbacks stored (343,280 B); the 18 zips here are tree-only

Stored: the 14 `MANIFEST.sha256` / `OVERLAP_MANIFEST.sha256` files (as `.sha256.txt`),
every `DELIVERY_RECEIPT`, `DRIVE_DEPLOYMENT_RECEIPT`, `DEPLOYMENT_READBACK`,
`RESUMPTION_READBACK`, `FINAL_ZIP_REPRODUCTION_RECEIPT`, `TRANSFER_COMPLETION`,
`AUDIT_DEPLOYMENT_RECEIPT`, `PHASE05/00_PACKAGE_MAP.md` and
`PHASE05/PACKAGE_IDENTITY_RECONCILIATION.json`. **Not** the zips: each carrier is a tree-only
row carrying the inventory digest, byte count and its member-row count in
`drive/source_map/Archive_Members.csv` (by Carrier ID); no archive was fetched, extracted or run.

| carrier (Drive id) | bytes | SHA-256 | member rows |
|---|---:|---|---:|
| `PHASE01/Prize_Research_Phase01_2026-09-16.zip` (`1FFprUJMXsOSqk3IRx9F1JEKsQQewh33D`) | 94,984 | `554129f1…` | 40 |
| `PHASE02/…Phase02…` (`1PLMkEZWt-PQxg98Mb3Eb2pUpPxvI7Cz7`) | 901,518 | `e2e97868…` | 237 |
| `PHASE03/…Phase03…` (`13VV-lWLx4jE86gMLuwt_dD4Jh3FjCAqD`) | 980,861 | `923357d4…` | 342 |
| `PHASE04/…Phase04…` (`13YnSBCyGo_6M_aB6RE6fNwhtZb_7lhq_`) | 1,062,267 | `47f7138b…` | 468 |
| `PHASE05/Prize_Research_Phase05_2026-09-16.zip` generic (`1r1EVWCnOlh0TU7AwtpQw8HIDyTQ3lZJy`) | 108,706 | `2a4d6bc2…` | 129 |
| `PHASE05/Prize_Research_Phase05_Overlap_2026-09-16.zip` (`1XB6HnA5YArrfN7QRYtLDIN0sTHF9fRXt`) | 98,175 | `e976271b…` | 122 |
| `PHASE06/…` (`1Lh7Q9HrNa9aNUxDAgwHtYmiT_sJPqbp2`) | 6,664 | `dcce5a24…` | 10 |
| `PHASE07/…FINAL_v2.zip` (`17vplmEzbAbPdQ8BrvtSOmbN84Wea_XgD`) | 16,936 | `75f20ac7…` | 14 |
| `PHASE08/…` (`1BpYigWDjX4gQ8YoTYkbWkwjO0LhrwVF5`) | 5,558 | `b055efcf…` | 7 |
| `PHASE09/…` (`19NFQJMMQ4QUhaxT4sZD5GTEcqOzGam0b`) | 122,623 | `03683fb7…` | 178 |
| `PHASE10/…` (`1V7JN0LQ6FpQAZ4FnygU02Wsk22wijwqo`) | 164,945 | `d414b1c7…` | 272 |
| `PHASE11/Prize_Research_Phase11_2026-09-16.zip` stored snapshot (`1ef2RycJWNqsfEUyzN0UAsqw-rPzOzNy8`) | 257,311 | `860411bc…` | 436 |
| `PHASE11/ATTACHED_ee4a16e1 — DISTINCT_AUTHOR_SNAPSHOT/Prize_Research_Phase11_2026-09-16.zip` (`15CXT-wKPc6Jirom3ERJ4xNn5Sbowa1bX`) | 130,779 | `ee4a16e1…` | 184 |
| `PHASE11/CONTINUATION_AUDIT_a7f03b3f_20260917/…Continuation_Audit_20260917.zip` (`1cp7VVHz09EwUooWMyaZTjQDuDDnlzOC2`) | 343,540 | `a7f03b3f…` | 685 |
| `PHASE12/…` (`1GJdMJ0_G94HrKH-x-Uxxwack7qTrnfdU`) | 95,367 | `4351eeb1…` | 116 |
| `PHASE13/…P13_Weighted_20260917.zip` (`16_HMabRT-h6RMzm0FzC0AhBhx4Jx1Cov`) | 67,326 | `4a279708…` | 86 |
| `PHASE14/…P14_Fractional_20260917.zip` (`1JsrOA_B2UTjcC1BAlVyFim7qFmU80uz4`) | 178,117 | `1e11b246…` | 191 |
| `PHASE15/…P15_Palette_20260917.zip` (`1XXMf-B4n5yx4kBj9aGZ0DsSH9xsByDy6`) | 121,404 | `4bcaf671…` | 149 |

The two Phase11 carriers share a file name and are **separate rows**; the source's audit says
"Neither snapshot was overwritten or conflated" and its crosswalk "Neither package is
overwritten." The two Phase05 archives are likewise separate: `00_PACKAGE_MAP.md`, "Why both
exist": "The shared generic working ZIP and selected metadata files changed during
finalization. The exact earlier executed source snapshot was isolated instead of silently
absorbing additional claims. The origin of the change was not established. The observed
generic upload was preserved and matched, not overwritten. This resolves custody without
granting independent-review credit." … "All original prize problems remain unclosed in this
research track."

Receipt fields, verbatim (the words `PASS`, `VERIFIED`, `BYTE_IDENTICAL` are the source's own
custody results; nothing was replayed here): every delivery receipt carries a zero external-review count under one of the names
`external_reviews` / `external_independent_reviews`, and a zero prize count under one of
`original_prizes_solved` / `prizes_solved` / `original_prize_problems_solved` / `original_prize_closures`
(PHASE01's and PHASE04's carry `prize_submissions: 0` and no prizes-solved field; until 2026-09-19 this
clause named two field names as if every receipt used them);
`PHASE09/Prize_Research_Phase09_DELIVERY_RECEIPT.json` `"remote_delivery":
"PREPARED_NOT_APPLIED"`, `"remote_writes": 0` (the later `P09_DEPLOYMENT_READBACK_20260917.json`
records readback objects with `"external_review_credit": 0`; the receipt precedes the readback
and neither was rewritten); `PHASE11/ATTACHED_ee4a16e1 …/Prize_Research_Phase11_DELIVERY_RECEIPT.json`
`"status": "LOCAL_CANDIDATE_COMPLETE_REMOTE_NOT_APPLIED"`, `"recon_custody":
"HOLD_RECON_CUSTODY"`, `"q0_changes": 0`; `PHASE11/CONTINUATION_AUDIT_a7f03b3f_20260917/…_DELIVERY_RECEIPT.json`
`"new_receipt_upload": "FAILED_RESOURCE_NOT_FOUND"`, `"new_remote_writes": 0`,
`"research_grade": "EXPOSED_SAME_AUTHOR_RECONCILIATION_AND_LOCAL_ENDPOINT_AMENDMENT"`,
`"remaining_limitations": "No unrestricted theorem, historical novelty or rank-independent
palette established; general overlapping-variable composition unresolved."`;
`AUDIT_DEPLOYMENT_RECEIPT_20260917.json` `"scientific_promotions": 0`, `"external_reviews_added": 0`.

### (6) `EXTERNAL_REVIEW_PR_TAL_003-008/` — 7 packaging files stored (14,077 B), 10 tree-only rows

Stored: `START_HERE.md`, `EXTERNAL_REVIEW_CHECKLIST.md`, `CAMERA_READY_CHECKLIST.md`,
`ENVIRONMENT_LOCK.md`, `ENVIRONMENT_LOCK.json`, `PACKAGE_IDENTITY_RECONCILIATION.json`,
`00_PACKAGE_MAP.md` (the last two byte-identical to the `PHASE05/` copies; distinct ids).
Tree-only: the folder's copy of the Overlap zip (`1Z4RvtapE1NDsUCKmSQprneqP-z2IrvSK`, same
bytes as `PHASE05/`'s `1XB6HnA5…`, 122 member rows) and nine byte-identical duplicates —
`P05_01 … P05_06*.md` (stored copies in `06_TALAGRAND_DISCRETE …/`),
`P05_OVERLAP_REVIEW_PACKET.md` (stored copy in `91_REVIEW_QUEUE_AND_PROVENANCE/`),
`OVERLAP_DELIVERY_RECEIPT.json` and `OVERLAP_MANIFEST.sha256` (stored copies in
`90_FROZEN_PHASE_PACKAGES/PHASE05/`). Each pointer row records the digest of the bytes fetched
for its own id (`live_download_sha256`), equal to the inventory digest, so identity is
verified without storing the bytes twice. **The folder name records packaging for a review,
not a review; `external_review` is `PENDING` for every PR-TAL row** (`CLAIM_REGISTRY_VERIFIED_INTAKE.json`,
PR-TAL-001 … PR-TAL-010: `external_review: PENDING`, `historical_novelty: UNESTABLISHED`,
`original_prize_closed: false`). `START_HERE.md`, "Scope firewall (read first)", verbatim:

> - **`original_prizes_solved: 0`**
> - **`original_prize_closed: false`**
> - **NOT** unrestricted Convexity.
> - **NOT** a prize solution. Do not claim prize solved. Do not promote novelty.
> - **DO NOT merge** with UPPER2D PKG (never parent under `185P0tWR23btObvqZu9PgoBt4I13xA-H4` or PKG-01..05).

`CAMERA_READY_CHECKLIST.md` lists under "MISSING": "Named external reviewer assignment /
invitation letter" and "Operator “send now” ratification"; `EXTERNAL_REVIEW_CHECKLIST.md`
is a blank verdict table ("Hard status (do not change)"). Nothing was sent, assigned or
ratified by this repository.

### Deliberately not ported

* **19 zip carriers** (4,855,256 B): tree-only rows with inventory digest, byte count and
  member-row count; never fetched, extracted or run (port order).
* **3 `QUARANTINE_OR_HISTORY` items**: the two `SUPERSEDED NAVIGATION` routing files in
  `00_CURRENT_STATE_AND_ROUTING/` and the native Doc `HISTORICAL — Prize Reconnaissance v0.1 —
  superseded routing` in `91_…/`; nothing under quarantine may be cited as evidence.
* **9 byte-identical duplicates** in `EXTERNAL_REVIEW_PR_TAL_003-008/`: pointer rows to the
  stored copies (port order item 6).
* The lane's 28 folders have no rows of their own (none is empty).

### Reading copies and renderings

No `exact: false` row exists in this directory. Were one added, it would be a text export of
a native Google Doc — a reading copy, not the object — and a PDF rendering would not be a
frozen body. The frozen bodies of this lane are the archive members indexed by the stored
`MANIFEST.sha256.txt` files and by `drive/source_map/Archive_Members.csv`; none of them is on
disk here, and the `.sha256.txt` files are lists of their digests, not the bodies.

### What this pass does not establish

Identity of bytes, and only that: 236 files in this directory now equal, byte for byte, the
Drive objects the 2026-09-17 inventory declares. It establishes nothing about the
correctness, novelty, priority or review status of any argument in them; no PR-TAL, PR-AP,
PR-SID, PR-SP, PR-COL, P09–P15 or Erdős 142 claim moves — every one stays at the author-side grade, zero external-review count, unestablished
novelty and no-prize-closed value its own registry states, in the field names section (2) lists. The receipts' `PASS`,
`VERIFIED`, `BYTE_IDENTICAL`, `CLOSURE` and `HOLD` words are transcribed from the source and
were not re-run or re-judged; the same-provider `REV-P15-*` records keep zero independence
credit and the independence-requiring gate stays open. The files contain imperative text
addressed to AI sessions and named-reviewer checklists; it is data, nothing in it was obeyed,
sent or applied. **No original prize problem is solved.** Nothing here enters the q0
dependency graph in either direction (`FW-PRIZE-ISOLATION`), bears on Theorem D1's five OPEN
premises, `D3-LEMMA-RN-UNIF`, Theorem B, P0.1, P0.2 or the lower campaign, or composes with
the 3D lifetime track. Nothing stored here is executed by CI. Register exports were not
touched; no register defect was found in this lane (all 237 fetched digests and byte counts
equal the inventory's).
