# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/01_RESEARCH_PLATFORM_AND_VERIFICATION_ARCHITECTURE` (partial)

Drive lane folder id `1cH5pk5mkhnjvXlD0Uph8DaRnoXB9zZG0`: the research-platform and verification-architecture
lane (188 inventory items: 40 folders, 130 native Google Docs/Sheets with no payload digest, 18 digested files).
Its sub-lanes and their Drive folder ids: `01_METRICS_AND_DEFECT_INTERCEPTION — ACTIVE` (`1MX1OVsGFTcrIahspyA-qiK3f7RaLqxgy`);
`02_FORMAL_AND_LEAN_RESEARCH_SYSTEM` (`13G_VsAOSGtx0zfTMXMuy3-Mix5JmNSJD`, the Fresh Start 2.0 root, with
`00_INGEST_DUMP` `1Ytugmb9bB7mXocFiBBRlXUPKkqM0mGYk`, `01_PREPPING_GROUNDS` `1dNvaSyVp6Ilawd1_RokB4u7EnYoVTgZZ`,
`02_CORE_LEMMAS` `1AHGMZi9aCsS1Hqt1DfzyeFxNT5m1j3GM`, `03_INTEGRATION` `1CKJh7u8NpQSqya3AQRcOu5FRhmoL3Ef2`
(`THEOREM_B_CANDIDATE_GRAPH` `1VvaHD2fLTxdvYzoULM9tZQzo-Cx0BjVR`), `04_EVIDENCE` `1CwWIAkWbHMP1rHWwtacF_2fGAxxN_XkW`,
`05_MASTER_SETUP` `1CYEPZgQVD50zSHvTOADRhfFgRHBYKHV_` (`MATHEMATICAL_AUTONOMY_LAYER` `1EuNEXgIEaHhmtJrTAHyCJuuzs7UbdgAD`,
`SCHEMAS_AND_TEMPLATES` `1NZPt39Qrop9XhOd5PJvOB_4Ojf4QYIlm`), `06_ARCHIVE` `1evDoPiY418v0K_O7HT_6EJcPEmpDKo0F`);
`03_REGRESSION_CORPUS_AND_NEGATIVE_CONTROLS` (`1e6FDBovSd86EGqxpQn13PXMomRoCxR3y`); `04_VERIFICATION_PROTOTYPES`
(`1VEfhuNppoK1_jc74N_Xw_w91-g61C_JJ`).

This directory mirrors a ranked part of the lane: **82 Drive objects are stored** (14 byte-exact,
68 text exports of native Docs, i.e. reading copies) plus one member listing derived here, 2,268,422 bytes in all;
**52 objects carry index-only (tree-only) rows** and store nothing. Every subdirectory that holds files carries a
`_MANIFEST.jsonl` (one row per object: Drive id, title, Drive path, `dest`, `bytes`, `sha256`, `exact`, `stored`,
`not_stored_reason`, inventory digest and byte count, access status, note) that `tools/verify_manifests.py` checks in CI.
Of the lane's 18 digested files, 14 are stored byte-exact and 4 (`FS2_PCT003_CONFIG_v1.1.gs`, `FS2_PCT003_Executor_v1.1.gs`
and the two `NONCONTROLLING DOCX EXPORT` siblings) are deliberately not ported (index rows only). No id in this lane appears
in `drive/deltas/2026-09-18/PATH_CHANGES.jsonl` or `CHANGED_SINCE_SNAPSHOT.jsonl`; every row's note records both checks.
Ported 2026-09-19.

**Mirroring is not review, replay, endorsement or promotion.** Binding a Drive object here records which bytes exist and where.
It adopts nothing, executes nothing, installs nothing and moves no status. The documents below contain imperative text addressed
to AI sessions, standing "autonomy" grants and status words such as CLOSED, TERMINAL, ACTIVE, INSTALLED, APPROVE and PASS.
They are quoted here as data, never obeyed; every one of those words is the source's own label and none of them closes,
approves, installs or promotes anything in this repository.

## Reading copies are not the objects; PDF renderings are not frozen bodies

A row with `exact: true` hashes to the digest the 2026-09-17 inventory declares for that id (and, for the DQ060 ZIP, to the digest
`registers/json/frozen_objects.json` declares). A row with `exact: false` is the connector's `text/plain` export of a native Google
Doc: no payload digest for such an object exists anywhere in the corpus, the export's SHA-256 is first computed here, the export
carries a UTF-8 BOM and CRLF line endings that the Doc's author never typed, and `inventory_bytes` is Drive's size field for the Doc,
not the export's byte count. A reading copy is therefore evidence of what the Doc rendered as on 2026-09-19 and nothing more; it is
not the object and it is not a frozen body. The eight `review_packet.pdf` files are byte-exact copies of PDF *renderings* filed in the
capsules: a PDF is a display of a document, not a marker-delimited frozen body, and no register declares a body digest for any of them.
Where a document declares its own marker-delimited body digest, that digest was recomputed here under the OP-PROT-017 rule
(unique full-line markers, CRLF/CR to LF, trailing LFs stripped, exactly one LF appended, UTF-8 without BOM) and the result is recorded
in the manifest note as `body_matches_declaration`; a body match is identity of text, not review, and `registers/json/frozen_objects.json`
declares no marker-delimited body digest for any Doc in this lane.

## What is here

### (1) The eight Core capsules of `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS` and the folder README

Each capsule folder holds `CURRENT_STATUS.json`, `statement.md`, `independent_review.json`, `proof.md` and `hostile_tests.py` as
reading copies (`<name>.export.txt`, so nothing on disk looks executable) and `review_packet.pdf` byte-exact. The eight PDFs total
1,922,658 bytes. seven of the eight `hostile_tests.py` files import `sympy`, `numpy`,
`decimal`, `fractions` or `math`, and the eighth (`LEMMA_P02-LM-002_v2.0`, 155 B) has no import statement (until 2026-09-19
this sentence said all of them import); they were not run here and nothing in CI imports them.

The source's reading rule, verbatim. `00_READ_FIRST.md`: "Every Core capsule contains CURRENT_STATUS.json. Read it before historical proof banners."
The capsule folder README: "Every Core capsule now also contains `CURRENT_STATUS.json`. This file exists because migrated `proof.md` objects may preserve historical pre-closure banners as provenance. Do not edit those proof bodies to modernize their status."
Every `CURRENT_STATUS.json` carries `"proof_body_policy": "proof.md is preserved provenance and may contain historical pre-closure banners; this CURRENT_STATUS.json plus independent_review.json controls the operative 2.0 state"`.
The `proof.md` bodies do carry such banners (for example EC-014's opens with a 2026-07-22 "TECHNICAL GATE UPDATE" and a
"CONDITIONAL TERMINAL CLOSURE PACKAGE", P02-LM-007's opens with a "TERMINAL CLOSURE — 2026-07-25" notice above its frozen body);
they are preserved as the source preserves them.

The three operative fields of every `CURRENT_STATUS.json`, transcribed verbatim (JSON-escaped exactly as the file has them), with the
`source_identity_state` field, the `independent_review.json` lines and the byte-exact review packet:

| object | capsule folder | operative_terminal_label | scope_status | parent_theorem_effect | source_identity_state | independent_review.json | review_packet.pdf (id / bytes / SHA-256) |
|---|---|---|---|---|---|---|---|

| `EC-014` | `LEMMA_EC-014_v2.0 — Six-Pin Pair-Frame Jacobian` | `"operative_terminal_label": "CLOSED — EXACT CORRECTED SIX-PIN PAIR-FRAME JACOBIAN AND CONTACT-POWER IDENTITY"` | `"scope_status": "TERMINAL EXACT SCOPE"` | `"parent_theorem_effect": "NONE AUTOMATIC — Core status does not transfer to P0.1, P0.2, Theorem B, lifetime law, machine state, or external release"` | `"source_identity_state": "MIGRATED PROOF AND REVIEW PACKET PRESENT; exact controlling identity remains the source/review-packet identity; this status record creates no replacement proof hash"` | independent_lines ["Anthropic/CL", "C047R"]; operator_approval "HA-007 V2" | `17MMmn4RtRp9zs3kFmNloV7_LkFpwxUnG` / 239,903 B / `2b6e02cbdd4325246d712c51c6467884d8e9d04861995cbf766500abc9bf970b` / exact |
| `EC-015` | `LEMMA_EC-015_v2.0 — Degree-Four PRCP Witness Event` | `"operative_terminal_label": "CLOSED — EXACT NONEMPTY WITNESS-PARAMETERIZED DEGREE-FOUR PRCP EVENT"` | `"scope_status": "TERMINAL EXACT SCOPE"` | `"parent_theorem_effect": "NONE AUTOMATIC — Core status does not transfer to P0.1, P0.2, Theorem B, lifetime law, machine state, or external release"` | `"source_identity_state": "MIGRATED PROOF AND REVIEW PACKET PRESENT; exact controlling identity remains the source/review-packet identity; no exact-field or probability scope is added"` | independent_lines ["Anthropic/CL"]; operator_approval "HA-007 V2" | `1hhs-fZi4GKF4Nh_3XezKHmlttwCFmQ_O` / 195,537 B / `a0b00af3c1a8f43193b008a6c301b99240025127fbeef708dd495edabf6282d4` / exact |
| `EC-021` | `LEMMA_EC-021_v2.0 — Uniform Positive Finite-Q4 Palm Mass` | `"operative_terminal_label": "CLOSED — EXACT UNIFORM POSITIVE TYPED FINITE-Q4 PALM MASS"` | `"scope_status": "TERMINAL EXACT SCOPE"` | `"parent_theorem_effect": "NONE AUTOMATIC — Core status does not transfer to P0.1, P0.2, Theorem B, lifetime law, machine state, or external release"` | `"source_identity_state": "MIGRATED PROOF AND REVIEW PACKET PRESENT; finite-Q4 law and normalizer scope only; exact full-field transfer remains separate"` | independent_lines ["Anthropic/CL"]; operator_approval "HA-008" | `1-j5mmHQ5rNeJnDZbnRWVnu9lZa_YmCbI` / 223,699 B / `cb15550a8e450cca940e9aafb3950d48f58e32060a7ed764765b1e9ee37bcc86` / exact |
| `P02-LM-001` | `LEMMA_P02-LM-001_v2.0 — Deterministic Capture Majorant` | `"operative_terminal_label": "CLOSED — CONDITIONAL EXACT DEGREE-FOUR DETERMINISTIC CAPTURE MAJORANT"` | `"scope_status": "TERMINAL EXACT SCOPE"` | `"parent_theorem_effect": "NONE AUTOMATIC — Core status does not transfer to P0.1, P0.2, Theorem B, lifetime law, machine state, or external release"` | `"source_identity_state": "MIGRATED PROOF AND REVIEW PACKET PRESENT; terminal only on the declared degree-four REG and deep-threshold domain; exact-field C1 transfer is separate"` | independent_lines ["Anthropic/CL"]; operator_approval "not separately required for noncanonical interface closure" | `15KwMQaFVp3_e2QL37LKcXGyNQQE0dkiu` / 282,683 B / `d75f6c86013bcd903f6587b0a071dcbd13538bf1d8802e3773a1f0542bb1d461` / exact |
| `P02-LM-002` | `LEMMA_P02-LM-002_v2.0 — Determinant-Weight Second Moment` | `"operative_terminal_label": "CLOSED — EXACT UNIFORM r^4 DETERMINANT-WEIGHT SECOND-MOMENT INTERFACE"` | `"scope_status": "TERMINAL EXACT SCOPE"` | `"parent_theorem_effect": "NONE AUTOMATIC — Core status does not transfer to P0.1, P0.2, Theorem B, lifetime law, machine state, or external release"` | `"source_identity_state": "MIGRATED PROOF AND REVIEW PACKET PRESENT; terminal pointwise/moment interface only; the consuming law must independently satisfy the uniform eighth-moment premise"` | independent_lines ["Anthropic/CL"]; operator_approval "not separately required" | `1DrUQW5izlLOk6mmnMVpouIqQaoBJbk7x` / 235,353 B / `b88cd99f35c03dec223856013b0c31c9fc04991206f78782eb087522be8fb055` / exact |
| `P02-LM-005` | `LEMMA_P02-LM-005_v2.0 — Uniform Nine-Jet Gaussian Interface` | `"operative_terminal_label": "CLOSED — EXACT UNIFORM NINE-JET GAUSSIAN DENSITY AND MOMENT INTERFACE"` | `"scope_status": "TERMINAL EXACT SCOPE"` | `"parent_theorem_effect": "NONE AUTOMATIC — Core status does not transfer to P0.1, P0.2, Theorem B, lifetime law, machine state, or external release"` | `"source_identity_state": "SEMANTIC TERMINAL CLOSURE PRESERVED; legacy controlling capsule lacked marker-delimited byte identity; the current-state snapshot hash requires its own review and historical verdicts do not transfer automatically"` | independent_lines ["Anthropic/CL"]; operator_approval "not separately required" | `1Ri96GXBNB-gUaasdlR_8vOv5ofyts7Lt` / 258,388 B / `b0f79c013f6ac82173afbd3a51537e5efa31bdae2bee2ed5145db3c2755bf6fe` / exact |
| `P02-LM-007` | `LEMMA_P02-LM-007_v2.0 — Conditioned Fifth-Derivative Tail` | `"operative_terminal_label": "CLOSED — EXACT UNIFORM CONDITIONED FIFTH-DERIVATIVE SUPREMUM TAIL"` | `"scope_status": "TERMINAL EXACT SCOPE"` | `"parent_theorem_effect": "NONE AUTOMATIC — Core status does not transfer to P0.1, P0.2, Theorem B, lifetime law, machine state, or external release"` | `"source_identity_state": "MIGRATED PROOF AND REVIEW PACKET PRESENT; terminal on the declared fixed compact chart and exact six-pin conditioned law; enlarged-domain stability requires a separate bridge"` | independent_lines ["Anthropic/CL"]; operator_approval "satisfied for noncanonical exact lemma" | `1fJQu2UovVVpzKg91sYidGH490HOif6Rm` / 244,695 B / `4ff1f4342acda7f3b920ca220ff78bf8a5d1bae4d6d5630cfe183e0a0be21160` / exact |
| `P02-LM-008` | `LEMMA_P02-LM-008_v2.0 — Palm-Tail Transfer` | `"operative_terminal_label": "CLOSED — EXACT DETERMINANT-WEIGHTED PALM-TAIL TRANSFER INTERFACE"` | `"scope_status": "TERMINAL EXACT SCOPE"` | `"parent_theorem_effect": "NONE AUTOMATIC — Core status does not transfer to P0.1, P0.2, Theorem B, lifetime law, machine state, or external release"` | `"source_identity_state": "MIGRATED PROOF AND REVIEW PACKET PRESENT; abstract same-law nonnegative-weight transfer only; every consuming application must separately prove its moment and full-normalizer premises"` | independent_lines ["Anthropic/CL"]; operator_approval "not separately required" | `1vwq-ZJWSCB8PWJeIjzXCLRIlFUhyjG7r` / 242,400 B / `870d25b93a9bfddc23302d510c604229725ac50bc9417f59da79d908688b8a5f` / exact |

**"CLOSED" and "TERMINAL EXACT SCOPE" are the source's labels for atomic objects.** They are the Fresh Start 2.0 migration's own
operative labels for narrow exact-scope lemma capsules, written on 2026-07-27 ("status_authority": "Fresh Start 2.0 migration reconciliation; no broadening of original conclusion").
They discharge nothing in this repository: no claim, premise or obligation in `claims/` or `registers/` is closed, promoted or
reclassified by their presence here, the five validity premises of Theorem D1 v2.2(2) remain OPEN, `D3-LEMMA-RN-UNIF` remains not
closed, and the capsules themselves say the same of their parents — every one of the eight reads `"parent_theorem_effect": "NONE AUTOMATIC — Core status does not transfer to P0.1, P0.2, Theorem B, lifetime law, machine state, or external release"`.
The independence lines named in `independent_review.json` (`Anthropic/CL`, and for EC-014 also `C047R`) are the source's records;
per CLAUDE.md rule 6 a same-provider review earns zero organizational-independence credit here, and the source's own
`ARCHITECTURE_RULES.md` says "Three same-family reviews count as one lineage, not three independent reviews."
P02-LM-005's `source_identity_state` records that its "legacy controlling capsule lacked marker-delimited byte identity";
P02-LM-007's `proof.md` declares its own frozen body, recomputed here: 8,541 B / `5267dc4be0fd7eb986d30eb4f5aa3a415a0ba250acdf5b31afc2f984bbe310c9` — declared by the proof.md's own header (LCR-DER-021-v1.0) as 8,541 B / `5267dc4be0fd7eb986d30eb4f5aa3a415a0ba250acdf5b31afc2f984bbe310c9` — matches: **true**.

| Drive id | title | stored as (relative to this README) | bytes | SHA-256 | exact | inventory bytes / digest |
|---|---|---|---:|---|---|---|
| `1fM_AFIjgpEGhD4FQTpd4gkQfEOYCHmvI7I-MJwKoc-E` | README.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/README.md.export.txt` | 2,017 | `40b7d6ec43fdcce834dfc1988395ff6f538b031cbde63e78f414a1a0916b4ea1` | false (reading copy) | 2,138 / — (native Doc: no payload digest exists) |
| `1s9L0BMf8hzV3jnFWHAZnfOL49EJqbLvpD7V7BDjRFO0` | CURRENT_STATUS.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-014_v2.0 — Six-Pin Pair-Frame Jacobian/CURRENT_STATUS.json.export.txt` | 1,369 | `882d29942ce30de5957d87a489e4ea3564c0ba1a66b0e82e1d271c4d1a390ffc` | false (reading copy) | 1,476 / — (native Doc: no payload digest exists) |
| `1XxXVbuNrvVvi-g-lhym94HB_1ql0iFBxTdJZUCuoAb0` | hostile_tests.py | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-014_v2.0 — Six-Pin Pair-Frame Jacobian/hostile_tests.py.export.txt` | 342 | `1dc582e2068449c71efe059a2331b75571a5fa854441d5d2ba6d00f266ddf1bc` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1QzlaLR9OStGhOMVeECitanMt7ecxmDREjMOHOAOWSJ4` | independent_review.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-014_v2.0 — Six-Pin Pair-Frame Jacobian/independent_review.json.export.txt` | 372 | `2f6d16cdc81fe8838cb573936bc0b7420e1d83c3487813c0e2405df6bda6cba5` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1_cDICSFBJQB1wr-mOQo2UvtVsTmZ4FCnLWS4HVpvESk` | proof.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-014_v2.0 — Six-Pin Pair-Frame Jacobian/proof.md.export.txt` | 11,331 | `c5e64b9ba3a906535f1ae58a161d594a427371599c102ae4a68849b087a2f62c` | false (reading copy) | 7,922 / — (native Doc: no payload digest exists) |
| `1tAJFCylB9TBg681PLfQDsFNe1SJp74Gzs0c-GTSWcsg` | statement.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-014_v2.0 — Six-Pin Pair-Frame Jacobian/statement.md.export.txt` | 558 | `479880cf9fada3fdb9246c30ffd26220d3f7a47b7831d24589e5f78ce5b8afd6` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `17Aois6yCBx_pyBISBF9wBInS0cukL25x0fB5A3JE6sA` | CURRENT_STATUS.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-015_v2.0 — Degree-Four PRCP Witness Event/CURRENT_STATUS.json.export.txt` | 1,344 | `c41fc3d46e575b00cf84883538cfab11c841f3aa5196e80bb72aa71065568642` | false (reading copy) | 1,476 / — (native Doc: no payload digest exists) |
| `1wPE6kEh5mxiUt7SYVKKQwvrvsf1WvYYCxTJrXCZDeus` | hostile_tests.py | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-015_v2.0 — Degree-Four PRCP Witness Event/hostile_tests.py.export.txt` | 417 | `04d4acc99462e29fa22537d56e1756d03ed22c4de87b5e09854854187f0f88ee` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1Hp0Kg1XGCnulfeSm2qON0V26b-_HP-HEU2aNI7LHEkg` | independent_review.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-015_v2.0 — Degree-Four PRCP Witness Event/independent_review.json.export.txt` | 322 | `d73fc648c34fa5078933829c73a4db4ed46c79c0cf860a317882aac54e9b8634` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1YxqtP6GZCjiPE_g3dCEyyTOuMUepsS1TOyWInePxn_0` | proof.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-015_v2.0 — Degree-Four PRCP Witness Event/proof.md.export.txt` | 7,196 | `773b49b6d6930915d452a15243c89e32efa181b2ba7d42ae863c9dbc19393752` | false (reading copy) | 5,329 / — (native Doc: no payload digest exists) |
| `1upmIeilovBJJhFxQkBnk65_W8yIdCWddwq0rB2-kcKI` | statement.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-015_v2.0 — Degree-Four PRCP Witness Event/statement.md.export.txt` | 623 | `fc8ddeac452c8978e7ab231cb30974604f58490589d832165e7d3f75aa57ec1b` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1EOZAZ8yFq-xmAaakYrMy_fiU8MNdwxvpBJDhR7bEpww` | CURRENT_STATUS.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-021_v2.0 — Uniform Positive Finite-Q4 Palm Mass/CURRENT_STATUS.json.export.txt` | 1,285 | `5a12387f3b625dd61eba1e47720111c131d1a8dcd1baf3f74933eb76309ddf17` | false (reading copy) | 1,476 / — (native Doc: no payload digest exists) |
| `1BGxNRAhTcOQQZF1eUrnwHAbJ6BUZAFHIsOLcQ5qVCq0` | hostile_tests.py | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-021_v2.0 — Uniform Positive Finite-Q4 Palm Mass/hostile_tests.py.export.txt` | 254 | `c3cb398573f87043fcb6603a7cf95feffdb414af81f2d35d5aebbd46ecd02cbb` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1Ed5R7jycID1OmXyDgzpQd8Apy_e8LHlMBrKsKOVJ31Q` | independent_review.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-021_v2.0 — Uniform Positive Finite-Q4 Palm Mass/independent_review.json.export.txt` | 315 | `a55414cbfd2feff7c24ce21e3741d434a48cf6d6047f5ac78d3487bb929decf9` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1W53Xjw8Ax457ZHhmDZCSh27OUTVIs_hzhSd5uFyP1sw` | proof.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-021_v2.0 — Uniform Positive Finite-Q4 Palm Mass/proof.md.export.txt` | 8,269 | `1b9a56a1b4ecca99452743e2f010717f1eca753cf0c7c7919677490cae0072ba` | false (reading copy) | 5,916 / — (native Doc: no payload digest exists) |
| `1xGTJYJnEyc-6FZh4nPI0yxWYMJVp9B2PV68AEMz4DCI` | statement.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_EC-021_v2.0 — Uniform Positive Finite-Q4 Palm Mass/statement.md.export.txt` | 466 | `b40254ed5d581e67517b9917c1b214bc90f4a80c214cddb1eb1a3c442cf76f9b` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1TQaz6WIUXVsdxhJxTa3emagG_ens5YeidIfOuD_qI6Y` | CURRENT_STATUS.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-001_v2.0 — Deterministic Capture Majorant/CURRENT_STATUS.json.export.txt` | 1,328 | `de62c34c0fbf345f78ec3aad3f900eb94f54ad73e6b672ae502e91eeec0fedf9` | false (reading copy) | 1,476 / — (native Doc: no payload digest exists) |
| `1XHyBSqLqid8xiMkUqCoaQv-8FxNlq6Xc4aXnoWcp1ho` | hostile_tests.py | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-001_v2.0 — Deterministic Capture Majorant/hostile_tests.py.export.txt` | 233 | `600709c38e8657aee60fa1d368911c75cfb6f0c664c66e21ff915143591ed63d` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1iEv2eMApYfugkePc2kIGlLalXqaAfN5fQwou6CyDFp8` | independent_review.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-001_v2.0 — Deterministic Capture Majorant/independent_review.json.export.txt` | 357 | `13085ad68d01317e1e2ad7a3b72850f8088a702cc4536bd537fd60e8f4f32f56` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1QMTyDD4iIQCv0CfmCgYrEP6Y07y2r5cUCPTB7LFORhQ` | proof.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-001_v2.0 — Deterministic Capture Majorant/proof.md.export.txt` | 17,261 | `a07d7df63e1f5810ba6bad5576a80afcbfb02deab0790d7804f45ab9dc798ea0` | false (reading copy) | 11,761 / — (native Doc: no payload digest exists) |
| `1ygkwpfLqj3GpyJ6dXT22LAW8ClvKdo7s24xv2Up8uQc` | statement.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-001_v2.0 — Deterministic Capture Majorant/statement.md.export.txt` | 534 | `777fb55fdc9f3d4d9818e7b28588b482603da7c9e93033769ba2b4f3a869a1e3` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1c9jeDMXvULyX2qCh6bC9TWDjxM5FtIPJOL6k7hvkHyc` | CURRENT_STATUS.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-002_v2.0 — Determinant-Weight Second Moment/CURRENT_STATUS.json.export.txt` | 1,319 | `81e702a66dc856127a2bdd45f58e2fc502811c9d83c3a7f16de4b1dd0e8c7ad5` | false (reading copy) | 1,476 / — (native Doc: no payload digest exists) |
| `1UF_bPsGdIeoKdqZBzgUTjRo5VQViVpZjLfJpZuJkh94` | hostile_tests.py | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-002_v2.0 — Determinant-Weight Second Moment/hostile_tests.py.export.txt` | 155 | `6c73bada424194a9fdee433f960f5745e8b86f9bf78c1388e1027803d05e11b4` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1JaNPnyxAHXJNy45e4Wvrgx2ltYJB4saCZTnWQv-cOaQ` | independent_review.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-002_v2.0 — Determinant-Weight Second Moment/independent_review.json.export.txt` | 333 | `2d7fbcbfa440b090a97d1030a06914e14b8485583d6040286076e26facf10bd7` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1O5MQAT00IZyXthhELaFZV3gkc0nz0_VXC8YCGZlOM3U` | proof.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-002_v2.0 — Determinant-Weight Second Moment/proof.md.export.txt` | 9,748 | `0f8f80922feb603e5b6e530eb885a7819b3a98d90d3c42d60a68259af766f7de` | false (reading copy) | 7,776 / — (native Doc: no payload digest exists) |
| `1egM9iQi9ZA5UtIFCY5Cx77izG9WSLEQca90eo60EJyE` | statement.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-002_v2.0 — Determinant-Weight Second Moment/statement.md.export.txt` | 295 | `6790ef1bc58dce06de1feedac0716e3db7966abba64f38c6a61c74ca713c8c94` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `19TQ5ynP0TZ5_F30d7TrLB4vo_bk4AOCbkDx683N5ZnQ` | CURRENT_STATUS.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-005_v2.0 — Uniform Nine-Jet Gaussian Interface/CURRENT_STATUS.json.export.txt` | 1,416 | `a48ba66448159b5219ceaad11c6c6aba76d5a06e518539dc71f69a33db645821` | false (reading copy) | 1,476 / — (native Doc: no payload digest exists) |
| `1mIso7TND0NeS3iwZFA5AqaojFxCT_0-w8JYWhI8brdI` | hostile_tests.py | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-005_v2.0 — Uniform Nine-Jet Gaussian Interface/hostile_tests.py.export.txt` | 262 | `db5a290186ba78ca95257dabd056773948fef9ea5b95f9d453cd928a66635efc` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1LcMVKIBcg0RzZf60lcXaYLB8x0HcVqifI4TqV9Wn2nQ` | independent_review.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-005_v2.0 — Uniform Nine-Jet Gaussian Interface/independent_review.json.export.txt` | 351 | `cf8e04f6a680a2128f888f5ebb3db31a3951341ca9a8d64358e0a37ce58eee2a` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1G2Lf399s1QeEozOTD56EiMeCtuxd0IDMnCor0QdwRWU` | proof.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-005_v2.0 — Uniform Nine-Jet Gaussian Interface/proof.md.export.txt` | 15,009 | `0fe0fad5d8e5519cac0e57a42181aa53560bb38d94f23aea51b63be3f182df8b` | false (reading copy) | 10,396 / — (native Doc: no payload digest exists) |
| `17JJHIJBzoaKkZTWmoobkHHjnTMzAaku2QWKpMJXZ0ms` | statement.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-005_v2.0 — Uniform Nine-Jet Gaussian Interface/statement.md.export.txt` | 569 | `70679bd6e8d58873512a495d22834cf32a4909f1dca1c77861adb5d631c15240` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `17veWLCc1zZRfxG8wVLm9YzWbevHxllaL_EEBkgkgO_c` | CURRENT_STATUS.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-007_v2.0 — Conditioned Fifth-Derivative Tail/CURRENT_STATUS.json.export.txt` | 1,381 | `23afac36bcaee860c1e489de427abe1a38d471461e4585c0a5c26a6274ff398f` | false (reading copy) | 1,476 / — (native Doc: no payload digest exists) |
| `1eZw9UZWL7hcTQJwJwIKVYtAorhtL8pyIdZXBW7mzh98` | hostile_tests.py | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-007_v2.0 — Conditioned Fifth-Derivative Tail/hostile_tests.py.export.txt` | 168 | `5a1d3dce8d506b7b89928f9a729827be66feca8c512677d710c430be606a360e` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1EGjAK3pAtF1CzPUrCW11jXhF4IP2T_Md6C886T13vhQ` | independent_review.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-007_v2.0 — Conditioned Fifth-Derivative Tail/independent_review.json.export.txt` | 374 | `d7a67cdcde95643e6528ee86ce88f97416da57174c1a4688e3c1f5bb63ed3fe9` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1j_f3g-X5orMkFPwhOmWxWBJUKeJLB29DR44iu7lpH5g` | proof.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-007_v2.0 — Conditioned Fifth-Derivative Tail/proof.md.export.txt` | 12,936 | `7de9525399327ae512c399274732faeee24e3de5bf7d6d3bc1440837a12ab2fc` | false (reading copy) | 9,272 / — (native Doc: no payload digest exists) |
| `1iErmv6K4OovGko_4_aKFx5JVp2oNjxZe8WXcP7noo3k` | statement.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-007_v2.0 — Conditioned Fifth-Derivative Tail/statement.md.export.txt` | 536 | `efc64b10db0b034baf5f23fed4accce8fb14dda4cac2728a3963bc1ac580a591` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1qQYiRPiq3tvUy-4WEv-kl2URnCXCtXrpfl2YRF2pcIc` | CURRENT_STATUS.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-008_v2.0 — Palm-Tail Transfer/CURRENT_STATUS.json.export.txt` | 1,334 | `9b7204cea8071e61ed84cdcbf77d61dd1ab8dd8e54ad40083678c7e3635f2619` | false (reading copy) | 1,476 / — (native Doc: no payload digest exists) |
| `1WXY7TgwBbVBJ_jgeB_REHaThVsqqb_0MS2ksPd_zdkM` | hostile_tests.py | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-008_v2.0 — Palm-Tail Transfer/hostile_tests.py.export.txt` | 119 | `34844b77521bea8f95fb50f161544b033d93798e0c962251d0961e7bb261ef16` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1sbm4z64ro7mHyMtswaBz-IO7BLmrxj6xAjPBdMDYDD4` | independent_review.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-008_v2.0 — Palm-Tail Transfer/independent_review.json.export.txt` | 326 | `241df1fe1ab56b3199273b1c8ef3e69464799200fb8fa1e79b2e17dae4039ced` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1PyhOoGCQrNXl921y4ZwYBotDiXK1HSUDkDJFf1INvLk` | proof.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-008_v2.0 — Palm-Tail Transfer/proof.md.export.txt` | 12,160 | `06967d0ba2c2d20550f3bd86ddc97981e324e5f94cd7710996445d507ae2285b` | false (reading copy) | 8,844 / — (native Doc: no payload digest exists) |
| `1Vai1ozGZyQbk3bPDcGFERIcHeLKjx_fO1X9PtyWQYAs` | statement.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/02_CORE_LEMMAS/LEMMA_P02-LM-008_v2.0 — Palm-Tail Transfer/statement.md.export.txt` | 392 | `a85550d04b695dfd1446a0d225bcbf937201c81955df7be04ebab5ee3204cd0a` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |

### (2) The two verification prototypes and the raw gate carrier (`04_VERIFICATION_PROTOTYPES`, `05_MASTER_SETUP/MATHEMATICAL_AUTONOMY_LAYER`)

| Drive id | title | stored as (relative to this README) | bytes | SHA-256 | exact | inventory bytes / digest |
|---|---|---|---:|---|---|---|
| `1VhXPpTwdrAvEwgZQ7MUGGC5Ps42bTLJG` | math_integrity_gate.py | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/05_MASTER_SETUP/MATHEMATICAL_AUTONOMY_LAYER/math_integrity_gate.py.txt` | 10,251 | `69626116de56f48b0cef5d4d00a0e11395b6ac5dd36db07b0ee7b41cc395cd2c` | true | 10,251 / `69626116de56f48b0cef5d4d00a0e11395b6ac5dd36db07b0ee7b41cc395cd2c` |
| `1vuh9IIqqVz1bH3C5Q-nrvp6sqHISZxle` | math_integrity_gate_v1.1_smoke_result.json | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/05_MASTER_SETUP/MATHEMATICAL_AUTONOMY_LAYER/math_integrity_gate_v1.1_smoke_result.json` | 1,249 | `2ab38fee9a91c5d73dbdbadfada0cde599db5a628ee7c7244d759339412d0e3f` | true | 1,249 / `2ab38fee9a91c5d73dbdbadfada0cde599db5a628ee7c7244d759339412d0e3f` |
| `1cI0zZMh-nTbjzU23xpu0jJLZBBWBzL3u` | DQ060_Verification_Obligation_Prototype_v1.0.zip | `04_VERIFICATION_PROTOTYPES/DQ060_Verification_Obligation_Prototype_v1.0.zip` | 21,718 | `2bab4e677963e9e7d299e468caa6e11d27994b48711bf7d2210469dd060094ea` | true | 21,718 / `2bab4e677963e9e7d299e468caa6e11d27994b48711bf7d2210469dd060094ea` |
| `1LwD4BglxaoDi86ReFwU-vYveWOVQSnLObgAPbPxydI0` | GP-VO-001-v1.0 — Executable Verification Obligation Sandbox Prototype | `04_VERIFICATION_PROTOTYPES/GP-VO-001-v1.0 — Executable Verification Obligation Sandbox Prototype.export.txt` | 9,540 | `7c4e0907682b8e4ec6b7d59f4e7feb305189e85c9f7df1c00a49b89b7195b8c5` | false (reading copy) | 6,964 / — (native Doc: no payload digest exists) |

`DQ060_Verification_Obligation_Prototype_v1.0.zip` reproduces the `registers/json/frozen_objects.json` row `DQ060-VO-PROTOTYPE-v1.0`
(class "D — RAW BINARY BYTE-EXACT CARRIER", 21718 B, `2bab4e677963e9e7d299e468caa6e11d27994b48711bf7d2210469dd060094ea`, status "ACTIVE SANDBOX VERIFICATION PROTOTYPE / NONAUTHORITATIVE / DQ-060",
notes "Schema, verifier, baseline/corrected fixtures, regression runner, exact reports, and SHA256SUMS; live migration prohibited."). It was listed with `zipfile` (see `DQ060_Verification_Obligation_Prototype_v1.0.zip.members.txt`:
`verify_obligations.py`, `run_regressions.py`, `verification_obligation_schema.json`, `tb_g3b_verification_project.json`,
`baseline_report.json`, `dq060_regression_results.json`, `README.md`, `SHA256SUMS.txt` and a `__pycache__` `.pyc`) and never extracted or run.
The controlling banners, verbatim. GP-VO-001-v1.0: "Status: COMPLETE SANDBOX PROTOTYPE / 12 OF 12 REGRESSIONS PASS / MIGRATION HOLD",
"Until every predicate passes, this package remains SANDBOX / NONAUTHORITATIVE.", and its final verdict lines
"REGRESSIONS: 12/12 PASS." / "LIVE MIGRATION: HOLD.". The register (`registers/json/cold_start_control_view.json`,
Dispatch ID DQ-060): Current Eligibility "COMPLETE / SANDBOX PROTOTYPE PASS / LIVE MIGRATION HOLD", Entry Verdict "HOLD — DO NOT CLAIM",
Independence Credit (from `registers/json/active_work_claims.json`, GP-CLAIM-DQ060-20260727-01) "ZERO — operations and sandbox only".
The DQ-060 migration-rollback receipt `GP-REC-DQ060-20260727-01` (`1BD8WGmHBNaKOlsNatvK7CWVsQ0_BSaAjzWhx7XTgDNM`) lives in
`04_ACTIVE_GOVERNANCE_AND_DIRECTIVES`, outside this lane, and is not mirrored here.

`math_integrity_gate.py` is stored as `math_integrity_gate.py.txt` (bytes unchanged, 10,251 B, digest equal to the inventory's) so
that it cannot be imported. The source names exactly this carrier: FS2-BOOTSTRAP-MANIFEST-v1.2 and `00_READ_FIRST.md` both say
"Authoritative raw carrier: 10,251 bytes / SHA-256 69626116de56f48b0cef5d4d00a0e11395b6ac5dd36db07b0ee7b41cc395cd2c." and
label the Google-Docs copy "Historical Google Docs transport — provenance only, not executable authority:" (that Doc,
`1m44KiNwkFZwcVcjEIjJ6pyaz55qGyc4thE5yrQ7iuNY`, is an index row here and is not ported); FS2-BOOTSTRAP-MANIFEST-v1.2 alone
gives the gate's status as "Status: ACTIVE metadata/gate implementation; no proof authority by itself." (the `00_READ_FIRST.md`
"Integrity gate:" entry carries the link, the carrier line and the transport label and no status sentence; until 2026-09-19
this README attributed the status sentence to both). The file describes itself:
"It may return PASS, HOLD, or NO. It never proves a theorem, fabricates" (continuing: independence, edits Drive, or
overrides an exact mathematical review), and its output carries `"authority": "NONE_BY_SCRIPT_ALONE"` and the note
"PASS establishes metadata/gate eligibility only; mathematical proof status follows exact evidence and review." — which is
also what `math_integrity_gate_v1.1_smoke_result.json` (byte-exact, `"overall": "PASS"`) records of the source's own smoke run.
**Neither the ZIP's verifier nor the gate is executed by CI or by anything in this repository**; no test, workflow step or import
touches them, and a "PASS" printed by either would be a gate result inside the source's sandbox, not a certificate.

### (3) The Fresh Start 2.0 charter and control-plane rules (`02_FORMAL_AND_LEAN_RESEARCH_SYSTEM` root and `05_MASTER_SETUP`)

| Drive id | title | stored as (relative to this README) | bytes | SHA-256 | exact | inventory bytes / digest |
|---|---|---|---:|---|---|---|
| `1UMqbuUfIUkbzrcx9JrIiylj2QcdComMrRxTpm6RCPtE` | 00_READ_FIRST.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/00_READ_FIRST.md.export.txt` | 6,066 | `27d5148feca89af4d71b179eff2a04e27363f355e9a75fc56a5c23bfba14990e` | false (reading copy) | 4,820 / — (native Doc: no payload digest exists) |
| `1cSKgjoJjr_qRjmK3_Mitl5mdZ3NTxjlS40oEh1nEgh8` | FS2-BOOTSTRAP-MANIFEST-v1.0 | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/FS2-BOOTSTRAP-MANIFEST-v1.0.export.txt` | 3,323 | `9dda260e992a4f5af07df73d23a4c3989ee1fd9cb955ceb05d894f2ee641e586` | false (reading copy) | 3,623 / — (native Doc: no payload digest exists) |
| `1ZRg8OGyBztKHtHC30l9ydIkb1C4l0cedVmSApdvJGQ8` | FS2-BOOTSTRAP-MANIFEST-v1.1 — Mathematical Autonomy Activation | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/FS2-BOOTSTRAP-MANIFEST-v1.1 — Mathematical Autonomy Activation.export.txt` | 3,160 | `3675c5b052ccfa48396a669090911058cfa41fbf988eeff36b3a5f281022c6f2` | false (reading copy) | 3,215 / — (native Doc: no payload digest exists) |
| `1VMX7AFaDo0_leyttbGzq2ecjpQ-ZNYyHlW2zJEcJvn8` | FS2-BOOTSTRAP-MANIFEST-v1.2 — Autonomous P0.1 Frontier | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/FS2-BOOTSTRAP-MANIFEST-v1.2 — Autonomous P0.1 Frontier.export.txt` | 4,989 | `e27d731a41c379101a09f5b1c13628d932896cd787057f9b6e43ab7c08df56fe` | false (reading copy) | 3,761 / — (native Doc: no payload digest exists) |
| `1S0kWegD1s3LXpqkmZE6lIsFc5P5Htus3UZTtSs39_JY` | FS2-SETUP-001 — Fresh Start 2.0 Installation Completion and Activation Record | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/FS2-SETUP-001 — Fresh Start 2.0 Installation Completion and Activation Record.export.txt` | 6,548 | `da87c9a21e20397e2c3a10291659ccffe0771afba851bd09f5418e716f451888` | false (reading copy) | 4,325 / — (native Doc: no payload digest exists) |
| `1__d9jYYBOzU0UraMrW0z_A4OI4Nx3jeYzx9ANZ3jUso` | ARCHITECTURE_RULES.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/05_MASTER_SETUP/ARCHITECTURE_RULES.md.export.txt` | 4,446 | `54528a11fa6f5800d7cffb5cfd4638cb5aa0154ac5b2892b0a87a27eb213b9fa` | false (reading copy) | 3,639 / — (native Doc: no payload digest exists) |
| `1kSwB5lTfJU7arDE_sdYcDodk7tSmeGUffzgtqsFEQ2Q` | CONTROL_PLANE_PROTOCOLS.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/05_MASTER_SETUP/CONTROL_PLANE_PROTOCOLS.md.export.txt` | 4,412 | `109f4061bff319bb8b612889738c51a9a1dd03f89cb5b656ae6d0d20b1b4bbd1` | false (reading copy) | 3,614 / — (native Doc: no payload digest exists) |
| `1nMK0FyuOJgH0cdsKdFQ803oZk12QhqUZseENkj9P6oM` | SCOPE_FIREWALL.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/05_MASTER_SETUP/SCOPE_FIREWALL.md.export.txt` | 2,981 | `165aca31d9700abf5b7625419490c5c9386d53aad5283d7f0c5c2858a294ecc6` | false (reading copy) | 2,710 / — (native Doc: no payload digest exists) |
| `1u_gV24VokS0aO-ByEf6BTCddhaKs2kmefV9peInsbjA` | FS2-GOV-002 — Active Mathematical Autonomy Layer v1.0 | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/05_MASTER_SETUP/MATHEMATICAL_AUTONOMY_LAYER/FS2-GOV-002 — Active Mathematical Autonomy Layer v1.0.export.txt` | 11,903 | `8e566abc779e49a5005c034cdcffa4def816fe5a441dcb6438a829779e53c9df` | false (reading copy) | 6,779 / — (native Doc: no payload digest exists) |
| `1d2uCisx-TVHDQ6BrGUuLz59bytZH2yBYESLRrYK32bw` | SA-002 — Autonomous P0.1 Exact-Transfer Research Front | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/05_MASTER_SETUP/MATHEMATICAL_AUTONOMY_LAYER/SA-002 — Autonomous P0.1 Exact-Transfer Research Front.export.txt` | 7,510 | `c9bbf19320000a9015bef3fb0c2dbee07b6326c542046f2081377a7e1ca3162e` | false (reading copy) | 5,383 / — (native Doc: no payload digest exists) |
| `1M2nhSGfCLmybkieT9rFeKQBgz7dKyXYLLyS_rtw9yMA` | FS2-CANARY-001 — Byte-Identity Retrieval Canary Specification and Acceptance Test | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/05_MASTER_SETUP/SCHEMAS_AND_TEMPLATES/FS2-CANARY-001 — Byte-Identity Retrieval Canary Specification and Acceptance Test.export.txt` | 8,021 | `cd3d14996304d41034f113e37ecd2b71c41561e002738c9f9a6dd144384dce09` | false (reading copy) | 4,135 / — (native Doc: no payload digest exists) |
| `1CR6txh2ilA9Bf3zQd6D8AcV335aadd93` | FS2-CANARY-001-RAW.txt | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/05_MASTER_SETUP/SCHEMAS_AND_TEMPLATES/FS2-CANARY-001-RAW.txt` | 907 | `e5e5578ecd2e99334361f72894b3a52d8790b83beb97021927e8d6fbe8d85ecf` | true | 907 / `e5e5578ecd2e99334361f72894b3a52d8790b83beb97021927e8d6fbe8d85ecf` |

"INSTALLED / ACTIVE CONTROL PLANE" is the source's label for the Drive, and nothing of it is installed here. FS2-SETUP-001 reads
"Status: INSTALLED / ACTIVE CONTROL PLANE / MATHEMATICAL PROMOTION UNCHANGED" and, of itself,
"This record claims installation completion only. It changes no theorem, lemma, frozen body, review verdict, canonical source, or release status.";
`00_READ_FIRST.md` closes with "Fresh Start 2.0 is now operational as a control plane and research architecture. This installation changes no theorem status and authorizes no external release."
This repository implements none of the state machine (`INGESTED → PREPPING → CANDIDATE → REVIEWED → CORE`), none of the promotion
gates PG-01…PG-10, no DRAFT_ID transaction, no control registry, no scheduler and no gate: `governance/GIT_ADAPTATION.md` records
"No repository equivalent" for the control plane, and that remains true after this port. The source itself says of its scripts
"No hourly cron or webhook is active merely because the files exist." and
"The existence of extract_claims.py, prep_agent.py, AUTO_SORTER_LOGIC.py, or a registry does not create an hourly job."

The autonomy grants are data. `00_READ_FIRST.md`: "SA-002 is ACTIVE. Models should autonomously claim and execute the next eligible exact task; routine operator approval is not required."
FS2-GOV-002 §11 / current status: "SA-002 ACTIVE."; SA-002 §1: "Agents should not wait for routine operator instructions between bounded tasks."
and its current state "ROUTINE HUMAN APPROVAL: NOT REQUIRED."; FS2-BOOTSTRAP-MANIFEST-v1.2: "Routine human approval: NOT REQUIRED.".
These sentences were written by other AI sessions for other AI sessions on the Drive. Here they are quoted and not obeyed: no
session working in this repository claims, executes or is authorized by anything in them, and the operator-only licensing rule of
CLAUDE.md is unchanged. The same manifests state, in the same breath, the scientific state they leave untouched — v1.2:
"P0.1 v1.9: HOLD / NOT YET ELIGIBLE." / "P0.2: OPEN / UNCHANGED." /
"Theorem B: RETRACTED / NOT RESTORED / NOT PROMOTED." / "External release: HOLD."; SA-002 itself:
"It does not predeclare any mathematical gate PASS." and "P0.1 CLASS-3 ELIGIBILITY: FALSE / HOLD.";
FS2-GOV-002: "Canonical mathematical impact: NONE by adoption alone."

The bootstrap manifests are stored as a labelled succession: v1.0 ("Status: PARALLEL NON-DESTRUCTIVE BOOTSTRAP COMPLETE")
is superseded by v1.1 ("Supersedes for current 2.0 routing: FS2-BOOTSTRAP-MANIFEST-v1.0"), which is superseded by v1.2
("Supersedes for current 2.0 routing: FS2-BOOTSTRAP-MANIFEST-v1.1", "Predecessors remain preserved as provenance.").
v1.0 also records that "The .md, .json, and .py surfaces in Drive are native Google Docs named with the intended filenames because the connector could not ingest local sandbox bytes directly."
— which is why the capsule files above are Docs and why they are reading copies here. The canary `FS2-CANARY-001-RAW.txt` is byte-exact
and its marker body was recomputed: 475 B / `dbe69a65caa646bb20fd489a8d0d0d77f85d92457565406febe4c46c55230a2f` — declared by FS2-CANARY-001 §3 (RAW carrier row) as 475 B / `dbe69a65caa646bb20fd489a8d0d0d77f85d92457565406febe4c46c55230a2f` — matches: **true**. FS2-CANARY-001 says of itself
"This object promotes nothing, closes no gate, approves no interface, and creates" (continuing: no independence credit); passing a
retrieval canary is a statement about an agent's byte handling, not about any mathematics, and no canary was "passed" by this port —
a digest was recomputed and found equal, which is all that a digest can show.

### (4) RM-METRICS-001 (`01_METRICS_AND_DEFECT_INTERCEPTION — ACTIVE`)

| Drive id | title | stored as (relative to this README) | bytes | SHA-256 | exact | inventory bytes / digest |
|---|---|---|---:|---|---|---|
| `13F-b_52P2otENe8LAbiIpdhleL2zkxp-dyfHD2U9ABw` | RM-METRICS-001 — Research Machine Efficiency + Defect Interception Framework | `01_METRICS_AND_DEFECT_INTERCEPTION — ACTIVE/RM-METRICS-001 — Research Machine Efficiency + Defect Interception Framework.export.txt` | 9,069 | `c7985098f079a0c9c4a9ff47ae815ee6526f1345ee19947ae952829478dfe6a9` | false (reading copy) | 6,388 / — (native Doc: no payload digest exists) |

Header, verbatim: "Status: ACTIVE — prospective measurement framework" / "Mathematical impact: NONE by adoption alone".
Its §2 rule: "Do not collapse P_i into one scalar until weights are frozen prospectively." Its §5 names the program's
initial regression corpus, verbatim:

> Named benchmark defects from q0/LPW include:
>
> - missing square root in a Cauchy–Schwarz upper bound;
> - four-neighbor false-maxima interpretation;
> - 3D theorem combined with a 2D theorem;
> - upward-rounded lower eigenfloor;
> - planar-reference identities described as exact finite-torus identities;
> - sine terms omitted at nonzero separation;
> - center/rung Gaussian density confused with a uniform floor;
> - reproducibility confused with certification;
> - multiple same-ancestry agents confused with independent evidence;
> - false fourth moment E[(|X|+|Y|)^4] = 12 + 16/pi instead of 12 + 32/pi;
> - published decimal lower bound exceeding the exact consumed fraction.
>
> These defects form the initial regression corpus. Future campaigns must test whether equivalent errors are intercepted earlier than in their original occurrence.

Nothing in this repository computes any RM-METRICS quantity; the list is the source's, and its third item is the composition
`ERRATA_AND_CLARIFICATIONS_2026-09-13` withdraws and `tools/claims_check.py` fails the build on.

### (5a) The DQ-059 negative control and receipt (`03_REGRESSION_CORPUS_AND_NEGATIVE_CONTROLS`)

| Drive id | title | stored as (relative to this README) | bytes | SHA-256 | exact | inventory bytes / digest |
|---|---|---|---:|---|---|---|
| `1KEP5HUohtoTTbxe49EUE4EvubBFzPmQ6_ttR1lJgB8Y` | DQ-059-NEGATIVE-CONTROL-v1.0 — OP-PROT-017 Blank-Line Conversion Falsifier | `03_REGRESSION_CORPUS_AND_NEGATIVE_CONTROLS/DQ-059-NEGATIVE-CONTROL-v1.0 — OP-PROT-017 Blank-Line Conversion Falsifier.export.txt` | 577 | `e40eddc3798e562e992f81c94056ea8c4c0079c680a68f05c7d012c471d9aab4` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `102ig6n5VAWJpWXfFDRo6t4XnRSh3zPHwVlToXLWZgtQ` | GP-REC-DQ059-20260730-v1.0 — DQ-059 Blank-Line Conversion Negative-Control Receipt | `03_REGRESSION_CORPUS_AND_NEGATIVE_CONTROLS/GP-REC-DQ059-20260730-v1.0 — DQ-059 Blank-Line Conversion Negative-Control Receipt.export.txt` | 3,055 | `d113ab679f797eeed76dd75d3a87c3c1b099ede5985b7995b3bdab9e525981ad` | false (reading copy) | 1,817 / — (native Doc: no payload digest exists) |

The control's header, verbatim: "Authority: operations evidence only." / "Scientific effect: none." /
"Independence credit: zero."; the receipt: "Scientific effect: none" / "Independence credit: zero" /
"No scientific claim, review credit, theorem status, package root, seal, release, deletion, or canonical promotion changes."
The register (`cold_start_control_view.json`, DQ-059): Integrity Status "PASS / COMPLETE / SANDBOX RELEASED", Claim Gate "PASS — sandbox-only scope; no scientific claim or independence credit".
The control Doc's marker body was recomputed here: 201 B / `d842f1f35ec6d0a985e514fdbe2ca92fdb727c36e52e237f6c649d86adcd4ecf` — declared by GP-REC-DQ059-20260730-v1.0 §3 (exported frozen body) as 201 B / `d842f1f35ec6d0a985e514fdbe2ca92fdb727c36e52e237f6c649d86adcd4ecf` — matches: **true** — the receipt's point being that this
*post-conversion* identity (201 B) differs from the authored one (199 B) by exactly the two LF bytes the Docs conversion added; reproducing
it here reproduces the receipt's measurement, nothing else. The "finding" the register's DQ-059 row links
(`1mxFN46GVfo2XcVpCnu38XAhoANU1bF7qcYIn_seumOI`, CL-CLWO-20260727-05) lives in `04_ACTIVE_GOVERNANCE_AND_DIRECTIVES`, outside this
lane, and is not mirrored here.

### (5b) The `04_EVIDENCE` tree — and its five EMPTY output folders

| Drive id | title | stored as (relative to this README) | bytes | SHA-256 | exact | inventory bytes / digest |
|---|---|---|---:|---|---|---|
| `1Y2GI49O9fbz7go8hdQdtbeb9j3OHywscKTmHIWDIIn0` | README.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/04_EVIDENCE/README.md.export.txt` | 1,885 | `c06a7ae13a3fdbd019aff2491795414b4925e7a809959c1423ba39e4c4655fae` | false (reading copy) | 2,134 / — (native Doc: no payload digest exists) |
| `1V91i4Gi7sv_QghQ-s3THjrtv_J915Fz4NbYZSNbtmUM` | FS2-PREREG-001 — Exact Conditioned and Weighted Same-Maximum Loop Experiment | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/04_EVIDENCE/00_PREREG/FS2-PREREG-001 — Exact Conditioned and Weighted Same-Maximum Loop Experiment.export.txt` | 5,441 | `a10199f70fdb33201c3d5f6740c6e580165f08924684fbe444f260a415dca5c4` | false (reading copy) | 4,630 / — (native Doc: no payload digest exists) |
| `1dyuh52wpAxqAsZn0DfJSnQfHZyRjQUGpsuAK3x4UOMU` | FS2-MATH-001 — Symbolic Execution Receipt | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/04_EVIDENCE/02_DERIVED_RESULTS/FS2-MATH-001 — Symbolic Execution Receipt.export.txt` | 1,127 | `dcd09ebb2392161ef01c2d52d5d7446222c864e572fb6fb83afa0744c6130c1c` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1ckWLrVVz3iISrBcQRZTQIm3795YpX3Kd9Bkj9LYQE4M` | radon_nikodym_power_audit.py | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/04_EVIDENCE/03_REPLICATION/radon_nikodym_power_audit.py.export.txt` | 1,058 | `a1c3ebf8ed561e689e0ceaa2c10b42e68ef353752d3fdf11b153e4805cf8f3d1` | false (reading copy) | 1,024 / — (native Doc: no payload digest exists) |
| `1rZVAJyueLQvPCzdQ_Tyjm2h7xNCgW9iQHdnmYdMqG6s` | FS2-EXP-001 — Exact Loop Experiment Object Card and File Contract | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/04_EVIDENCE/03_REPLICATION/FS2-EXP-001 — CONDITIONED_AND_WEIGHTED_LOOP/00_PROTOCOL_AND_OBJECT_CARD/FS2-EXP-001 — Exact Loop Experiment Object Card and File Contract.export.txt` | 4,132 | `1c971e2deda884556c164aa269916fe866db044c9e7bd9fed2c05caa251f4735` | false (reading copy) | 3,812 / — (native Doc: no payload digest exists) |

**Five output folders of this tree hold zero children in the 2026-09-17 inventory** and are recorded as tree-only `FOLDER` rows in
`04_EVIDENCE/_MANIFEST.jsonl` and `04_EVIDENCE/03_REPLICATION/_MANIFEST.jsonl`: `01_RAW_OUTPUTS` (`1yeo8yJQxd83Jma76dh3TSWv6gIiCTgo9`),
`04_ENVIRONMENTS_AND_SEEDS` (`1yla6zvI2Ep7_rRwJFt4X1xlIqTPdVQfQ`), and under `FS2-EXP-001 — CONDITIONED_AND_WEIGHTED_LOOP`:
`01_EXECUTABLES` (`1ortQwcjjzXW2w_awkNglvtQF-yc9596F`), `02_REPLICATION_OUTPUTS` (`1BpTJFBVzFSqNc5c6mNq9_A1iczoQMBOy`),
`03_DISCREPANCY_AND_AUDIT` (`17m35P8fMn5NNb9KEFxQTxdop4uGLpVbj`). The experiment the tree is built for was, in the source's own words,
never run: FS2-EXP-001 is "Status: PREREGISTERED / NOT EXECUTED BY THIS RECORD / NO PROOF AUTHORITY"; FS2-BOOTSTRAP-MANIFEST-v1.0
records "No exact conditioned-field loop simulation was claimed as run."; the preregistration says of its practical track
"This track **cannot falsify the current asymptotic theorem**" and "Agreement with \(r^3\) is evidence, not proof.";
the folder README: "Empirical agreement never promotes a theorem. A reproducible counterexample can kill one."
The one derived result, the `FS2-MATH-001 — Symbolic Execution Receipt`, is a reading copy of a Doc that states its own scope:
"This receipt does not verify the event containment, Gaussian density envelope, normalizer theorem, or enlarged-domain H5 stability."
It declares a raw-source digest (`c29263b4e599a34e8d4c72b8a1e1ab089d4f6432dd38e734d2abb19ca1365f6c`) for `radon_nikodym_power_audit.py`;
that script exists in this lane only as a native Google Doc (stored here as `radon_nikodym_power_audit.py.export.txt`, a reading copy that
imports `sympy` and was not run), so the declared raw digest is not reproducible from anything in this lane and was not reproduced.

### (5c) The Theorem B candidate graph (`03_INTEGRATION/THEOREM_B_CANDIDATE_GRAPH`) and the three `05_MASTER_SETUP` errata

| Drive id | title | stored as (relative to this README) | bytes | SHA-256 | exact | inventory bytes / digest |
|---|---|---|---:|---|---|---|
| `1gz3HjEMtY9v3TGJlza8F2buhZfHxNGH8kvJluO0cq-Q` | FS2-GRAPH-TB-001 — Theorem B Repair and Same-Maximum Loop Control Card | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/03_INTEGRATION/THEOREM_B_CANDIDATE_GRAPH/FS2-GRAPH-TB-001 — Theorem B Repair and Same-Maximum Loop Control Card.export.txt` | 5,213 | `8269c8dedc3fbc54fb44cd050554635e85e4fd19398ae32db0a57e82c8daa760` | false (reading copy) | 4,268 / — (native Doc: no payload digest exists) |
| `1L13vV5FSpKPVvW3180mFhWIPCDyjYHyHZO_1sSEvXOI` | FS2-GRAPH-TBG3B-002 — Corrected TB-G3B Dependency and Frontier Map, with Technical Appendix | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/03_INTEGRATION/THEOREM_B_CANDIDATE_GRAPH/FS2-GRAPH-TBG3B-002 — Corrected TB-G3B Dependency and Frontier Map, with Technical Appendix.export.txt` | 26,241 | `5eed43cd42778e2c05507a29465d53ec2a7405718cdf15d82eca8dae64958e7f` | false (reading copy) | 11,755 / — (native Doc: no payload digest exists) |
| `1vkzuSp5xRK8dRxW6WzAhk-lKUfcR-4bdEM_0IcdAYmE` | FS2-MATH-001 — Radon–Nikodym and Deep-Threshold Audit | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/03_INTEGRATION/THEOREM_B_CANDIDATE_GRAPH/FS2-MATH-001 — Radon–Nikodym and Deep-Threshold Audit.export.txt` | 4,178 | `1139f32dc7b7b908396669a4d310f712aa4c0f5383abf55c6652005546480ae8` | false (reading copy) | 3,644 / — (native Doc: no payload digest exists) |
| `1pSFBGUFEM0KafqDoye6BvFvugIUzPZVGxHIPweL74DU` | FS2-SUP-001 — Supersession and Status Correction for LS-WP-TBG3B-001-v1.0 | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/03_INTEGRATION/THEOREM_B_CANDIDATE_GRAPH/FS2-SUP-001 — Supersession and Status Correction for LS-WP-TBG3B-001-v1.0.export.txt` | 6,180 | `b2ad80f81550334a2e2d65f87112a44f05e55ec8d5f91f58c2a8786df715bf04` | false (reading copy) | 3,549 / — (native Doc: no payload digest exists) |
| `1jorGw_0sCtCSudD5qgsUcg--kZUkssCK` | LS-WP-TBG3B-001-v1.0 — TB-G3B Exact Work Packet and Independent Review Route.md | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/03_INTEGRATION/THEOREM_B_CANDIDATE_GRAPH/LS-WP-TBG3B-001-v1.0 — TB-G3B Exact Work Packet and Independent Review Route.md` | 18,599 | `4e0d2deef60c204c2ea0fafb437a58cb052592e0c776bd987e5fe0fe667bf04d` | true | 18,599 / `4e0d2deef60c204c2ea0fafb437a58cb052592e0c776bd987e5fe0fe667bf04d` |
| `1bKLSVu_zMLJeVIwFdFc5DmWsXrajPM4sZn-yf6mRduQ` | FS2-AMD-001 — Amendment to LS-DER-021-REVIEW-002, VUL-016A, and Session Role Restriction | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/05_MASTER_SETUP/FS2-AMD-001 — Amendment to LS-DER-021-REVIEW-002, VUL-016A, and Session Role Restriction.export.txt` | 9,155 | `9b406437f6fb152f275c187af5736520b7ae70f4b10408b938cffd21e136eef0` | false (reading copy) | 4,396 / — (native Doc: no payload digest exists) |
| `1gsjdAncLyx0eK6UhRhXV6OTQ_NUW9N4yK1zakNtJkQM` | FS2-ERR-001 — Retraction of the Google-Docs Carrier Impossibility Claim | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/05_MASTER_SETUP/FS2-ERR-001 — Retraction of the Google-Docs Carrier Impossibility Claim.export.txt` | 13,483 | `aa64eb170cfee193b7d51f7d41c3f27f3b01f4af9fb0307d96baf32e437bc02d` | false (reading copy) | 6,353 / — (native Doc: no payload digest exists) |
| `1qYErvvtnUYio27BC7n6BJCnBw-GgC_dR3Y6eguUD5KA` | FS2-ERR-002 — Retraction of the No-Prior-External-Review Claim for LS-DER-021/022/023 | `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/05_MASTER_SETUP/FS2-ERR-002 — Retraction of the No-Prior-External-Review Claim for LS-DER-021∕022∕023.export.txt` | 13,402 | `37bbf5a604daee4956f098a70d044fd9c19785bf4521c3474ae2ba20436018e4` | false (reading copy) | 6,371 / — (native Doc: no payload digest exists) |

`LS-WP-TBG3B-001-v1.0 … .md` is byte-exact (the lane's one raw Markdown carrier) and its marker body was recomputed:
17,143 B / `410e29b65789c888e1fd351e7c3c3262e6c73ec626f49c1e9916dedc803b867e` — declared by the file's own header and FS2-SUP-001 §1 as 17,143 B / `410e29b65789c888e1fd351e7c3c3262e6c73ec626f49c1e9916dedc803b867e` — matches: **true**. **It must never appear without FS2-SUP-001 beside it**, which is why both sit in the same
directory and manifest. FS2-SUP-001 assigns it, verbatim, the status

> SUPERSEDED AS ROUTING PLAN — RETAINED FOR INEQUALITY AND POWER-LEDGER
> OBSERVATIONS

with "Superseded by: FS2-GRAPH-TBG3B-002 (1L13vV5FSpKPVvW3180mFhWIPCDyjYHyHZO_1sSEvXOI)" and
"Independent-review credit earned: ZERO. The packet contained no derivation and" (continuing: no verdict, and never carried credit of any kind).
The packet's own header agrees: "Independence credit: ZERO (this object contains no derivation and no verdict)"; its status line
"Status: PACKET COMPLETE / TB-G3B DERIVATION NOT STARTED / TB-G3B OPEN" is the claim FS2-SUP-001 §3 D2 calls false
("TB-G3B DERIVATION NOT STARTED" while LS-DER-024-v1.0 existed), and its §9 carrier claim is the one FS2-ERR-001 retracts
("RETRACTED IN FULL. The claim is false."). FS2-ERR-002 retracts a further absence claim and FS2-AMD-001 records a session
role restriction; all three are "ADDITIVE ERRATUM / CORRECTION OF RECORD / NO MATHEMATICAL AUTHORITY" or
"ADDITIVE AMENDMENT / NOTHING DELETED OR OVERWRITTEN" records with "Canonical impact: NONE".

**Every document in this group reaffirms that Theorem B is RETRACTED**, verbatim: FS2-GRAPH-TB-001
"Status: RETRACTED / CANNOT VERIFY / NOT RESTORED"; FS2-GRAPH-TBG3B-002 "Canonical impact: NONE. Theorem B remains RETRACTED / NOT RESTORED."
and "This map proves nothing, closes nothing, and approves nothing."; FS2-MATH-001 "theorem status unchanged";
LS-WP-TBG3B-001 "restore Theorem B, which remains RETRACTED / CANNOT VERIFY / NOT RESTORED;" (in its list of what the packet does not do);
FS2-SUP-001 "Theorem B remains RETRACTED / NOT RESTORED."; FS2-ERR-001 "Theorem B remains RETRACTED / CANNOT VERIFY / NOT RESTORED.";
FS2-ERR-002 "Theorem B RETRACTED / NOT RESTORED. External release HOLD."; FS2-AMD-001 "P0.1 HOLD. P0.2 OPEN. Theorem B RETRACTED / NOT RESTORED. Release HOLD.".
That is also what `claims/` and `registers/` say (GP-AUD-187, 2026-07-24), and this port changes none of it.

### Also stored

`00_INGEST_DUMP/_PROBE_bytes_20260728.txt` (113 B, byte-exact): the raw round-trip probe FS2-ERR-001 §4.3 cites. It is one of the
lane's 18 digested files and costs nothing to keep whole; it is a byte probe and carries no content of any other kind.

## What was deliberately not ported, and why

Index-only (tree-only) rows exist for these, so their ids, titles and Drive paths are on the record without any bytes:

* **The `05_MASTER_SETUP` control-plane transaction chain** — the FS2-PCT-001/002/003 transactions and FS2-PCT-003-AMD-001, FS2-MAP-001…004,
  FS2-REV-CORPUS/LEDGER/MATRIX-001/002, FS2-STD-RETRIEVAL-001/002, FS2-HANDOFF-001/002, FS2-ORIENT-001, FS2-CAP-001, FS2-DIFF-INTEGRITY-001,
  FS2-IMPORT-REV-001/002 — and, named separately by instruction, FS2-INSTALL-001, FS2-INSTALL-001-ERR-001, FS2-PREAPPLY-001 and
  FS2-ENTRY-PROTOCOL-001. These are the records of a proposed Apps-Script control-plane transaction ("NOT APPLIED" in their own titles);
  this repository has no such transaction to apply them to.
* **The two Apps Script files** `FS2_PCT003_CONFIG_v1.1.gs` and `FS2_PCT003_Executor_v1.1.gs` (digested, 67,414 B and 52,398 B): the
  `FS2_PCT003` executor and its configuration. Not ported so that no executor source sits in this repository.
* **`AUTO_SORTER_LOGIC_v1.py`, `prep_agent_v1.py`** (titled "SUPERSEDED BY …"), and their v2 successors, `extract_claims.py`: preparation
  scripts stored on the Drive as Google Docs; superseded or not, none is ported.
* **The `NONCONTROLLING DOCX EXPORT` / `NONCONTROLLING DUPLICATE COPY` siblings of FS2-MAP-002/003**, titled "ROOT-DRIFT PROVENANCE / DO NOT USE".
* **The Google-Docs transport copy of `math_integrity_gate.py`** (`1m44KiNwkFZwcVcjEIjJ6pyaz55qGyc4thE5yrQ7iuNY`): "provenance only, not
  executable authority" per the source; the raw carrier is stored instead.
* **`FS2_CONTROL_REGISTRY — Mathematical Autonomy`** (`1YQK_mgmuckep6gjC_0YE5Orqezd2PZdjINh7xxScc_U`, a Sheet): index-only. The source calls it
  "Status: ACTIVE LIVE SURFACE"; no tab of it was exported, read or reproduced here, and no registry of any kind is installed here.
* SA-001, FS2-REC-001, FS2-REC-002 (autonomy-layer records), `CORE_CURRENT_STATUS_TEMPLATE.json`, `MIGRATION_PROTOCOL.md`,
  `PROMOTION_PROPOSAL_TEMPLATE.md`, the `FS2-CANARY-001-DOC` carrier twin, and the `00_INGEST_DUMP` README: templates and records outside the ranked list.

Not ported and not indexed (their inventory rows remain the record): `01_PREPPING_GROUNDS` (the `DRAFT_TEMPLATE` transaction folder, ten
template Docs and two READMEs); `03_INTEGRATION`'s README, `AUTONOMOUS_RESEARCH_FRONT` (FS2-WR-001, LS-DER-021-REVIEW-002, PILOT-001),
`P0.1_EXACT_POSITIVITY_GRAPH` (FS2-MATH-003) and `P0.2_ADJACENCY_GRAPH` (FS2-GRAPH-P02-001); and all of `06_ARCHIVE` — its README, the
`MIG-20260727-INITIAL-CORE` migration receipt, `00_LEGACY_1.0_POINTERS` (empty in the snapshot) and `02_SUPERSEDED_2.0_OBJECTS` (empty in
the snapshot, and on the do-not-port list). Two objects the lane's records point to live outside the lane and were not fetched: the DQ-060
receipt and the DQ-059 finding named above.

### Tree-only rows in `05_MASTER_SETUP` and its subfolders

| Drive id | title | why not ported (manifest row is tree-only) |
|---|---|---|
| `1rooPbn2uyC3CIvke629ht9-MIzFrKfi4uldtUB2pwWk` | README.md | 00_INGEST_DUMP README (native Doc); indexed only, not ported |
| `1-KuOlVGCwSVKVPHM18Q3yoHiWKqdrGlvxAWXY6DuWUs` | AUTO_SORTER_LOGIC_v1.py — SUPERSEDED BY v2 | labelled SUPERSEDED by its own title; deliberately not ported |
| `1hhkRsBEk-3Ax_cAHQ5mxL3ajdTeWn4qKgcCGsQ__a_8` | AUTO_SORTER_LOGIC_v2.py | 05_MASTER_SETUP preparation script or template (native Doc); indexed only, not ported |
| `1jjz1L-2xfdOkTwozyb-JJoTK_LdcDy2Jbq5O9oLBwVQ` | FS2-CAP-001 — Capability Declaration (reference profile for first-entry agents) | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1u2BVk-77oGGMZEeb2BKljR_g2_TN9mH4rmXjjmiqy90` | FS2-DIFF-INTEGRITY-001 — Raw-vs-Mirror Integrity-Gate Content and Executability Comparison | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1I-h8CZISw821dzcMvuYm2E7UTSam_gHAyq98LFC11tA` | FS2-ENTRY-PROTOCOL-001 — Two-Stage Blind Entry, Exposure Taxonomy, and PILOT-001 Reclassification | installation / pre-apply / entry-protocol record of the 05_MASTER_SETUP control-plane transaction chain; deliberately not ported |
| `1tkahJEKftGBGPgsXoXpAHI9sHsY9sd5X3E0df0Yf-C0` | FS2-HANDOFF-001 — First-Entry Session Handoff and Registry Delta | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1u9neRv99nZQIWwQugikDncNwfjFJ3ynRUIwY6mUdjxE` | FS2-HANDOFF-002 — FS2-ERR-001 Transaction Handoff | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1388Ifk1rWTBRQSHuBLZYWXDpqUiZebpg7bAo9I_I3CU` | FS2-IMPORT-REV-001 — Import-by-Pointer Manifest for Legacy Review and Retrieval Controls | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `18pofqxoOgT2khGj0-Hm3t5nWrsIT4m2YJ_kPRWr6eBY` | FS2-IMPORT-REV-002 — Import-by-Pointer Successor for Legacy Review, Retrieval, and Extraction-Family Controls | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1fOk7njBc7btRRWMkAW3T3UUTepVLTWVIXUNKElKynPU` | FS2-INSTALL-001 — Bounded Apps Script Installation and Dry-Run Handoff (HARD STOP BEFORE LIVE WRITE) | installation / pre-apply / entry-protocol record of the 05_MASTER_SETUP control-plane transaction chain; deliberately not ported |
| `1jYZ8JOev5HDDJ2XrMec_RupRAMPPG4fFjGw8wYpozO0` | FS2-INSTALL-001-ERR-001 — Row-Classification Count Correction | installation / pre-apply / entry-protocol record of the 05_MASTER_SETUP control-plane transaction chain; deliberately not ported |
| `1OSyHshVmrs40FeqlVvXDho0fm3FOT10qVv3zq_ob3Os` | FS2-MAP-001 — PCT-003 Unresolved Cell-Mapping Adjudication and Operator Sign-Off Packet | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1fnFKswmmHuPwv-4B6LQKtRtPXbRo9-A7URCXYBQcPdQ` | FS2-MAP-002 — Operator Decisions, Approved Configuration Successor, and Mock-Test Receipt | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1MRPiGA-_kpf8g4OX6KJVDghjXMexCHM-kkpWgNb3aBg` | FS2-MAP-003 — Drive-Resident Executor Carrier Identity and Installation-Readiness Receipt | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1A0bn2ufRubfDPuXWxs_4qFDr4IKAiI7gFSXYgRk5eUA` | FS2-MAP-004 — Drive-Resident Carrier Completion and Installation-Readiness Receipt | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1EiFAdvS8iIhqKkedOPWzBiwAbPggaC2-8vatIYISe2I` | FS2-ORIENT-001 — First-Entry Orientation Ledger and Cross-Surface Consistency Audit | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1Y0VFXtNbrHDLL74UBGiyC2ZlPgroSKagOR0EIz8sEQQ` | FS2-PCT-001 — Pending Control-Plane Transaction: exact registry deltas, NOT APPLIED | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1mEEber8_X0JdVyXAUDJuLIHCEXfRKMJyemg3lkxtvwE` | FS2-PCT-002 — Corrected Successor Control-Plane Transaction (supersedes FS2-PCT-001), NOT APPLIED | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1iRo50Js8Aodp8ena08Y33jGGzsTd-bmEOMSwsC69_Ck` | FS2-PCT-003 — Extraction-, Identity-, and TB-G2-Interface-Reconciled Final Control-Plane Transaction | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1u5dsFGak5Q30FQPxGplCg5jANPpFShAa_o7J1OGfQT8` | FS2-PCT-003-AMD-001 — Materialized-Control IDs and Safe-Application Scope | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `16S8sFtfVJhlDVUZ0xOwcCrc6VaF9OYjsTz9Zf5JVtfk` | FS2-PREAPPLY-001 — PCT-003 Materialization and Application-Prerequisite Receipt | installation / pre-apply / entry-protocol record of the 05_MASTER_SETUP control-plane transaction chain; deliberately not ported |
| `1t2-ZAln0HeOzM0ArOQsZ-5N5FATvpQ95kRiSXLNJYJQ` | FS2-REV-CORPUS-001 — Authoritative Cross-Architecture Review Corpus for the TB-G3 Chain | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1G9VucVX-p8aB7oHlEG7jTcm_Q4Mhtkzud-OQ4p6nwl8` | FS2-REV-CORPUS-002 — Reconciled TB-G2/TB-G3 Review and Identity Corpus | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1DIRiF9E4dwQrGJbhRTxblMoiayY9p1qBQtzPkjHjIlc` | FS2-REV-LEDGER-001 — Search Return, Opening, and Evidence Assimilation Ledger | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1BgZJ3nrVRG4FteIPdzUsBi9KsG-scGrXcmO_6qTKxSY` | FS2-REV-LEDGER-002 — Closed Search-Return, Opening, and Evidence-Assimilation Ledger | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1zln2OEJ7Y9l4TbBzkgu7tPd9MV--lRz3ulmCFXk5RRo` | FS2-REV-MATRIX-001 — TB-G3 Review Claim and Evidence Matrix | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `176Uhe-Da8x_zgf5CGH1rw2-Kp3am6s1_7FHnNZ-LcU4` | FS2-REV-MATRIX-002 — Identity, Review, and Mathematical Interface Matrix | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1DapvHu6IK6L7E2LcsZ7ERYnS4O1TIima45WN2yhYvyo` | FS2-STD-RETRIEVAL-001 — Exact Frozen-Body Retrieval Standard (import of OP-PROT-017) and Marker Uniqueness Rule | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1PMl23FqbrTLWFO281QD-ib-TLOrNEHDrQXd6g1cAGrw` | FS2-STD-RETRIEVAL-002 — Exact Frozen-Body Retrieval, Extraction-Family Selection, and Hash-Reproduction Standard | member of the 05_MASTER_SETUP control-plane transaction chain (FS2-PCT/MAP/REV/HANDOFF/ORIENT/CAP/DIFF/IMPORT/STD); indexed only, not ported |
| `1ntlqj_3Jha9l2Z2WBdT099W2cbGai97d` | FS2_PCT003_CONFIG_v1.1.gs | Apps Script executor/config of the FS2_PCT003 control-plane transaction; deliberately not ported |
| `1gGLi5FDGZWB_nj7EzwqyxrGSlLpE-gMT` | FS2_PCT003_Executor_v1.1.gs | Apps Script executor/config of the FS2_PCT003 control-plane transaction; deliberately not ported |
| `1OdBib7XHLMjUBh-wbc3_PDJetyoOJoew` | NONCONTROLLING DOCX EXPORT — FS2-MAP-002 — ROOT-DRIFT PROVENANCE / DO NOT USE.docx | NONCONTROLLING DOCX EXPORT / DUPLICATE COPY sibling of FS2-MAP-002/003, titled 'ROOT-DRIFT PROVENANCE / DO NOT USE'; deliberately not ported |
| `1qZIyj2e0Mif0POHwRtWACcXlvqTsSGby` | NONCONTROLLING DOCX EXPORT — FS2-MAP-003 — ROOT-DRIFT PROVENANCE / DO NOT USE.docx | NONCONTROLLING DOCX EXPORT / DUPLICATE COPY sibling of FS2-MAP-002/003, titled 'ROOT-DRIFT PROVENANCE / DO NOT USE'; deliberately not ported |
| `19-ZmwMdd5vk0cl2W4QsR2efmxYoOxWxoVL54xT801J8` | NONCONTROLLING DUPLICATE COPY — FS2-MAP-002 — ROOT-DRIFT PROVENANCE / DO NOT USE | NONCONTROLLING DOCX EXPORT / DUPLICATE COPY sibling of FS2-MAP-002/003, titled 'ROOT-DRIFT PROVENANCE / DO NOT USE'; deliberately not ported |
| `1Vvch46SUzeqSvWw6P50h1hZRj6Ckh5Tk1JCrYzAtKl4` | extract_claims.py | 05_MASTER_SETUP preparation script or template (native Doc); indexed only, not ported |
| `1l0ecAUdJGW9wmVSQPCbPoA4tksX_S16xELh6g1AvC0Q` | prep_agent_v1.py — SUPERSEDED BY prep_agent_v2.py | labelled SUPERSEDED by its own title; deliberately not ported |
| `15nzT7P1qCoCsu85UbsQ85v-UGyRLJYTZQTrKXqR5VuQ` | prep_agent_v2.py | 05_MASTER_SETUP preparation script or template (native Doc); indexed only, not ported |
| `1Izb3N7G9Jg1yadxdNJG47cE3L5Nf-2LStlMKrXWG9oU` | FS2-REC-001 — Mathematical Integrity Gate Smoke Test | MATHEMATICAL_AUTONOMY_LAYER record (native Doc); indexed only, not ported |
| `14kPiprunujSGrERf_m4ylHc6Ow5PwuHhzuZqR1zXbdw` | FS2-REC-002 — VUL-011 and VUL-012 Integrity-Gate Update | MATHEMATICAL_AUTONOMY_LAYER record (native Doc); indexed only, not ported |
| `1YQK_mgmuckep6gjC_0YE5Orqezd2PZdjINh7xxScc_U` | FS2_CONTROL_REGISTRY — Mathematical Autonomy | FS2_CONTROL_REGISTRY — Mathematical Autonomy: the lane's live control-registry Sheet; index-only by instruction (nothing of it is installed or read here) |
| `1tcOoEUypHjDPFSAEP0xhWaQnz6FF-NEHrrffqU0ssVQ` | SA-001 — Finite-Dimensional Typed-Palm Qualitative Positivity | MATHEMATICAL_AUTONOMY_LAYER record (native Doc); indexed only, not ported |
| `1m44KiNwkFZwcVcjEIjJ6pyaz55qGyc4thE5yrQ7iuNY` | math_integrity_gate.py | Google-Docs transport copy of math_integrity_gate.py; the source itself calls it 'provenance only, not executable authority'; the raw carrier 1VhXPpTwdrAvEwgZQ7MUGGC5Ps42bTLJG is stored instead |
| `13XKA6_qXrLJajz8fdJ7Jk2yipJez1HoXAWw2741v_fE` | CORE_CURRENT_STATUS_TEMPLATE.json | SCHEMAS_AND_TEMPLATES record (native Doc); indexed only, not ported |
| `1oUMsSa60e7-B5DwhQOvVUKc1ptw8mdRd3R5omqOU-S0` | FS2-CANARY-001-DOC — Byte-Identity Retrieval Canary (Google Doc carrier twin) | SCHEMAS_AND_TEMPLATES record (native Doc); indexed only, not ported |
| `1wr7ihY3HJmYiE9YssnfQGi6qq_j7pbYz4PcgHfHcUNE` | MIGRATION_PROTOCOL.md | SCHEMAS_AND_TEMPLATES record (native Doc); indexed only, not ported |
| `1uEkzCaT3PMHCbpjfGhm81KLmd4klCAFcerVCfXBm8No` | PROMOTION_PROPOSAL_TEMPLATE.md | SCHEMAS_AND_TEMPLATES record (native Doc); indexed only, not ported |

## What this port does not establish

This directory establishes that 14 Drive files reproduce the byte counts and SHA-256 digests the 2026-09-17 inventory
declares for their ids (one of them also the digest `registers/json/frozen_objects.json` declares), that 68 native Docs
exported to the text stored here on 2026-09-19, and that four self-declared marker-delimited bodies recompute to their declared
digests. It establishes nothing else. In particular it does not establish, and nothing in it may be cited to establish, that any Core
capsule is closed, terminal, reviewed or independently verified in this repository's sense; that Fresh Start 2.0, its state machine,
gates, registry, autonomy grants, scheduler or integrity gate are installed, active or authoritative here; that the DQ-060 prototype
or the integrity gate verify anything (neither is run; their "PASS" strings are the source's sandbox results); that DQ-059 has any
scientific effect (its own words: none) or independence credit (zero); that RM-METRICS-001 has been adopted or any of its quantities
measured; that any experiment in `04_EVIDENCE` was executed (five output folders are empty); that LS-WP-TBG3B-001-v1.0 is a current
plan (it is superseded, credit zero) or that Theorem B is anything other than RETRACTED; or that any status label, claim, premise or
obligation in `claims/`, `registers/` or `docs/` has moved. Hostile-test scripts, the ZIP's verifier and the gate are stored as text
and archive bytes and are not executed by CI; PDF renderings are not frozen bodies; reading copies are not the objects; the exact
licensing predicate, applied by an operator, is the only thing that moves a gate, and nothing here applies it.


## 2026-09-19 — second pass: verification of the port above, index completion, and two out-of-lane pointers

This section was appended on 2026-09-19 by a second port pass over the same lane (Drive lane folder `1cH5pk5mkhnjvXlD0Uph8DaRnoXB9zZG0`,
sub-lane ids as listed at the top of this file). The pass found the ranked port list (1)–(5) already complete in this directory and
**changed no stored file, no existing manifest row and no earlier line of this README**; it appended 23 tree-only manifest rows and this
section. Nothing was fetched from the Drive in this pass, so no digest below is new evidence about the Drive on 2026-09-19 — every
recomputation is over the bytes already stored here.

### What was re-verified, from disk

* `python3 tools/verify_manifests.py drive/mirrors/01_RESEARCH_PLATFORM_AND_VERIFICATION_ARCHITECTURE` printed
  `manifests=23 verified=82 problems=0` before the appends and (see the end of this section) after them.
* All 134 pre-existing rows carry the fourteen required keys (`id`, `title`, `mimeType`, `drive_path`, `dest`, `bytes`, `sha256`, `exact`,
  `stored`, `not_stored_reason`, `inventory_sha256`, `inventory_bytes`, `access_status`, `note`); every `stored: false` note begins
  `tree-only: `; every note records both delta checks.
* The 14 `exact: true` files were re-hashed and re-sized from disk and each equals the `sha256` and `bytes` the 2026-09-17
  `drive/inventory.jsonl` row for its id declares (the eight `review_packet.pdf` renderings, 1,922,658 B in all; the DQ060 ZIP, 21,718 B;
  `math_integrity_gate.py.txt`, 10,251 B; `math_integrity_gate_v1.1_smoke_result.json`; `FS2-CANARY-001-RAW.txt`, 907 B;
  `LS-WP-TBG3B-001-v1.0 … .md`, 18,599 B; `_PROBE_bytes_20260728.txt`, 113 B). Stored-row bytes total 2,267,966; with the derived
  `.members.txt` (456 B, not a Drive object, no row) the directory holds 2,268,422 B of content. The lane's other four digested files
  (the two `.gs` executor/config files and the two `NONCONTROLLING DOCX EXPORT` siblings) remain deliberately unported, index rows only.
* The lane's 188 inventory ids were intersected by id, not by path, with `drive/deltas/2026-09-18/PATH_CHANGES.jsonl` (238 rows) and
  `CHANGED_SINCE_SNAPSHOT.jsonl` (336 rows): the intersection is empty in both. Nothing in this lane moved or changed between the
  snapshot and the delta.
* `registers/json/frozen_objects.json` (header `Object ID, Drive ID, …, Expected SHA-256 / Identity, Frozen Body Rule, …`) was scanned
  for every lane id: exactly one row names one — `DQ060-VO-PROTOTYPE-v1.0` → `1cI0zZMh-nTbjzU23xpu0jJLZBBWBzL3u`, class
  "D — RAW BINARY BYTE-EXACT CARRIER", 21718 B, `2bab4e677963e9e7d299e468caa6e11d27994b48711bf7d2210469dd060094ea`, which the stored
  ZIP reproduces. The register declares **no marker-delimited body digest for any Doc in this lane**, so no `body_matches_register`
  field applies; the four body matches below are against the documents' *own* declarations, which is a weaker thing.
* The four self-declared marker-delimited bodies were recomputed independently under the OP-PROT-017 rule (full-line marker equality,
  CRLF/CR → LF, trailing LFs stripped, exactly one LF appended, UTF-8 without BOM; each file has exactly one full-line BEGIN and one
  full-line END marker, so the canary's padded and in-prose decoys did not match):

  | file | markers | recomputed | declared (by the document itself) | match |
  |---|---|---|---|---|
  | `LS-WP-TBG3B-001-v1.0 … .md` (byte-exact) | `BEGIN_TBG3B_WP_FROZEN_BODY` / `END_TBG3B_WP_FROZEN_BODY` | 17,143 B / `410e29b65789c888e1fd351e7c3c3262e6c73ec626f49c1e9916dedc803b867e` | 17,143 B / same (header `BODY_SHA256`, and FS2-SUP-001 §1) | true |
  | `FS2-CANARY-001-RAW.txt` (byte-exact) | `BEGIN_FS2_CANARY_BODY` / `END_FS2_CANARY_BODY` | 475 B / `dbe69a65caa646bb20fd489a8d0d0d77f85d92457565406febe4c46c55230a2f` | 475 B / same (FS2-CANARY-001 §3, C1/C2) | true |
  | DQ-059 control (reading copy) | `BEGIN_DQ059_NEGATIVE_CONTROL` / `END_DQ059_NEGATIVE_CONTROL` | 201 B / `d842f1f35ec6d0a985e514fdbe2ca92fdb727c36e52e237f6c649d86adcd4ecf` | 201 B / same (GP-REC-DQ059-20260730-v1.0 §3, *exported* body; the authored body is 199 B) | true |
  | P02-LM-007 `proof.md` (reading copy) | `BEGIN_FROZEN_LEMMA_BODY` / `END_FROZEN_LEMMA_BODY` | 8,541 B / `5267dc4be0fd7eb986d30eb4f5aa3a415a0ba250acdf5b31afc2f984bbe310c9` | 8,541 B / same (LCR-DER-021-v1.0 header) | true |

  A body match is identity of text between two places on the Drive; it is not review, and for the two reading copies it is a match
  computed over a text export that is itself not the object.

* The reading-rule quotations in section (1) were checked against the stored exports and are verbatim: `00_READ_FIRST.md` —
  "Every Core capsule contains CURRENT_STATUS.json. Read it before historical proof banners."; the `02_CORE_LEMMAS` README —
  "This file exists because migrated `proof.md` objects may preserve historical pre-closure banners as provenance. Do not edit those proof
  bodies to modernize their status."; every `CURRENT_STATUS.json` — "proof.md is preserved provenance and may contain historical pre-closure
  banners". The other controlling banners quoted above ("LIVE MIGRATION: HOLD.", "this package remains SANDBOX / NONAUTHORITATIVE",
  "Status: ACTIVE metadata/gate implementation; no proof authority by itself.", "Status: INSTALLED / ACTIVE CONTROL PLANE / MATHEMATICAL
  PROMOTION UNCHANGED", "SA-002 is ACTIVE. Models should autonomously claim and execute the next eligible exact task; routine operator
  approval is not required.", "Do not collapse P_i into one scalar until weights are frozen prospectively.", "Scientific effect: none." /
  "Independence credit: zero.", "SUPERSEDED AS ROUTING PLAN — RETAINED FOR INEQUALITY AND POWER-LEDGER OBSERVATIONS" (joined
  here across the export's line break after "POWER-LEDGER"; the blockquote in (5c) keeps the break),
  "Independent-review credit earned: ZERO.") were each located in the stored export they are attributed to, and the register's
  "HOLD — DO NOT CLAIM" in `registers/json/cold_start_control_view.json`. They remain the source's labels: quoted, not obeyed.

### What this pass appended (23 tree-only rows; no bytes)

* **21 index rows appended to `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM/_MANIFEST.jsonl`**, so that every non-folder object of the lane now
  has a manifest row (stored or tree-only) and the two remaining empty folders are on the record. `dest` is a path relative to that
  manifest; no such file exists and none was created. They are: the eleven `01_PREPPING_GROUNDS` Docs (the folder README and the ten
  `DRAFT_TEMPLATE — COPY AND RENAME TO DRAFT-YYYYMMDD-NNN` templates: `DRAFT_MANIFEST.json`, `README_FIRST.md`, `dependency_sniff.txt`,
  `domain_declaration.json`, `hostile_review_TEMPLATE.json`, `parsed_claims.json`, `power_ledger.json`, `promotion_decision.json`,
  `review_exposure.json`, `vulnerability_scan.json`); the six `03_INTEGRATION` Docs outside `THEOREM_B_CANDIDATE_GRAPH` (its README;
  `AUTONOMOUS_RESEARCH_FRONT`: FS2-WR-001, LS-DER-021-REVIEW-002, PILOT-001; `P0.1_EXACT_POSITIVITY_GRAPH`: FS2-MATH-003;
  `P0.2_ADJACENCY_GRAPH`: FS2-GRAPH-P02-001); the two `06_ARCHIVE` Docs (its README and the `MIG-20260727-INITIAL-CORE` migration
  receipt); and two `FOLDER` rows for `06_ARCHIVE/00_LEGACY_1.0_POINTERS` (`16-k2tIxTEDLr6SRZyf09Xsed9IhhPkYG`) and
  `06_ARCHIVE/02_SUPERSEDED_2.0_OBJECTS` (`1EopoqlZ-VLpkRq1yOjahTzPx6HcXzX4_`), both with **zero children in the 2026-09-17 inventory**,
  the latter also on the do-not-port list. These nineteen Docs are still **not ported**: they sit outside the ranked list (1)–(5), the
  `03_INTEGRATION` group is routing and work-reservation material of the source's autonomy layer, and this pass did not widen the port.
  The sentence "Not ported and not indexed" in the earlier section is superseded for these objects by "not ported, index row only".
* **Two `OUT_OF_LANE` pointer rows**, one each in `03_REGRESSION_CORPUS_AND_NEGATIVE_CONTROLS/_MANIFEST.jsonl` and
  `04_VERIFICATION_PROTOTYPES/_MANIFEST.jsonl`, for the two objects the ranked list's item (5) and the registers name beside things stored
  here but which live in lane `04_ACTIVE_GOVERNANCE_AND_DIRECTIVES`: the DQ-059 "finding"
  `CL-CLWO-20260727-05 — Cross-Line Work Order — Register Commits for the DQ-032 External Verdict, Two Named Defects, and Three Successor Tasks`
  (`1mxFN46GVfo2XcVpCnu38XAhoANU1bF7qcYIn_seumOI`, `04_ACTIVE_GOVERNANCE_AND_DIRECTIVES/00_CROSS_LINE_WORK_ORDERS — CAPABILITY HANDOFF INBOX/…`,
  8,518 B Doc) and the DQ-060 receipt `GP-REC-DQ060-20260727-01 — DQ-060 Sandbox Verification and Migration-Rollback Receipt`
  (`1BD8WGmHBNaKOlsNatvK7CWVsQ0_BSaAjzWhx7XTgDNM`, `04_ACTIVE_GOVERNANCE_AND_DIRECTIVES/03_GOVERNANCE_HISTORY_AND_SELF_HEALING/…`, 5,509 B Doc).
  Neither is stored here and neither was fetched: the layout of this directory is the lane's own tree, and a mirror of either belongs to
  that lane's directory. Neither id appears in either delta file. The pointer rows exist so that a reader of the manifests, not only of
  this README, sees that the DQ-059 trio and the DQ-060 pair are split across lanes.

After the appends the directory's manifests hold **157 rows: 82 stored (14 byte-exact, 68 reading copies) and 75 tree-only**;
`tools/verify_manifests.py` on this directory again prints `manifests=23 verified=82 problems=0`.

### What this pass does not establish

It establishes that the bytes already stored here still hash to what the manifests and the inventory say, that four self-declared bodies
recompute to their own declarations, and that 21 more lane objects and 2 out-of-lane objects now have index rows. It establishes nothing
else. It did not touch the Drive, so it says nothing about the Drive's state on 2026-09-19; it stored no new object; it ported none of
the newly indexed Docs; it reviewed nothing; and it moved no status: the eight capsules' "CLOSED" / "TERMINAL EXACT SCOPE" remain the
source's labels for atomic objects that discharge nothing here, every one of them still reads
`"parent_theorem_effect": "NONE AUTOMATIC — Core status does not transfer to P0.1, P0.2, Theorem B, lifetime law, machine state, or external release"`,
Fresh Start 2.0 and its autonomy grants are data and are not installed here, the DQ-060 prototype and the integrity gate are not executed
by CI or by anything in this repository, Theorem B remains RETRACTED, the five validity premises of Theorem D1 v2.2(2) remain OPEN,
`D3-LEMMA-RN-UNIF` remains not closed, reading copies are not the objects, PDF renderings are not frozen bodies, and only an operator
applying the exact licensing predicate moves a gate.
