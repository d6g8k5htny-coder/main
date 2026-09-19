# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/2026-09-15 — KIMI FINAL INTAKE — UPPER2D STAGE E + H5 + ASSEMBLY` (partial)

Drive folder id `1bRmeImIMT-6kc-zBNZVoWW1aYNPBPLWs` (56 inventory items, 47 files). This directory
mirrors the current D1 assembly of the 2D upper track and the two 2026-09-15 Anthropic-side proposals
that sit next to it: 4 byte-exact markdown files, 51,562 bytes.
`_MANIFEST.jsonl` in each subdirectory is checked by `tools/verify_manifests.py`. **Mirroring is not
review, replay, endorsement or promotion.** Ported 2026-09-19.

## Coverage of the lane

| top-level subfolder | inventory items (files) | objects mirrored here |
|---|---:|---:|
| `05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15` | 31 (28) | 2 |
| `(files directly in the lane folder)` | 7 (1) | 0 — not ported by this lane |
| `03_H5_PROMOTION_AND_RUNG_CERTIFICATES` | 6 (6) | 0 — not ported by this lane |
| `04_STAGE_E_REVIEWS_AND_DEFECT_FINDINGS` | 5 (5) | 0 — not ported by this lane |
| `01_CURRENT_ASSEMBLY_AND_STATE` | 3 (3) | 2 |
| `02_H4_EVENT_LEVEL_REPAIR` | 2 (2) | 0 — not ported by this lane |
| `06_UNMIRRORED_FROZEN_CARRIERS_BYTE_NATIVES` | 2 (2) | 0 — not ported by this lane |

## What is here

| Drive id | title | stored as (relative to this README) | bytes | SHA-256 | exact | frozen body: bytes / SHA-256 / rule / matches register |
|---|---|---|---:|---|---|---|
| `1v4z492iAzk5NcOrR47IJHGkIgfsRACpC` | D1_ASSEMBLY_v2_2.md | `01_CURRENT_ASSEMBLY_AND_STATE/D1_ASSEMBLY_v2_2.md` | 20,078 | `7ca114f0b38680d8bb987c097de10f3faf884ae3b05c3ca47215af5df081c174` | true | 18,311 / `490ad6b2f14176fe8cf5af363fb94dc73a8bc5523f608e5ab2a42ff749b235f6` / OP-PROT-017, leading blank lines stripped (= CL-REG-001 marker_strip_LF here) / **true** |
| `1oFNNK4D6BEVuBjTL9jERKynXxnNL_f4t` | D1_ASSEMBLY_v2_2_REGISTER_NOTE.md | `01_CURRENT_ASSEMBLY_AND_STATE/D1_ASSEMBLY_v2_2_REGISTER_NOTE.md` | 11,880 | `3ebc3c5ad911b2116fd18767e45260a421438d81df0b72c799c081d5886a97d3` | true | 10,752 / `c1d5e95d97e019574d6b6b09e6e606fd6e0cbc1fe3aa2e0a54953de6e4f2c470` / OP-PROT-017, leading blank lines stripped (= CL-REG-001 marker_strip_LF here) / **true** |
| `1RQE2B3EY9IP5MGZyeAtXBpX4ahY5AhfC` | D1_ASSEMBLY_v2_3_DRAFT.md | `05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15/D1_v2_3_DRAFT/D1_ASSEMBLY_v2_3_DRAFT.md` | 16,141 | `15053f8bf3ad103d87c4f3bed82c0d5897d86533dbbe2cb365b24bdc0cce7d69` | true | 14,734 / `2ec888daeb9e5b51e3f1cf6b1e502db8a3a9fb93b9ac81dd14c8e3e00da8dec4` / OP-PROT-017 / no declared digest to compare |
| `18R0wwSFHa--ZMdLRV3ElMO0T5u5YD3lk` | H5_ZBAND_CONSUMPTION_2026-09-15.md | `05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15/H5_ZBAND/H5_ZBAND_CONSUMPTION_2026-09-15.md` | 3,463 | `9445bdd3beba72c7533152aed08b115cccb20a107f6f0bb57c65b97eda6827a9` | true | — |

The frozen-body digests of `D1_ASSEMBLY_v2_2.md` (`490ad6b2…`, 18,311 B) and
`D1_ASSEMBLY_v2_2_REGISTER_NOTE.md` (`c1d5e95d…`, 10,752 B) are the ones `claims/graph.json`,
`docs/RESEARCH_MAP.md` and `docs/FULL_DOCS_MATH_READ.md` cite; they are reproduced from the stored bytes
under the rule the files themselves name (`marker_strip_LF` per CL-REG-001: text strictly between the
`BEGIN_FROZEN_BODY` / `END_FROZEN_BODY` lines, surrounding whitespace stripped, one trailing LF). The
draft's body digest is first computed here; no register declares one. The whole-file digests are the
inventory's.

## The status banners, verbatim

* `D1_ASSEMBLY_v2_2.md`: "**Artifact:** D1-ASM-20260915-v2.2 … This v2.2 is the current strongest form and supersedes all prior status lines." §5: "**Strongest conditional theorem:** Theorem D1 v2.2(2) — 1 − q(r, 6/5) ≤ C·r³ for all 0 < r ≤ r₀ (r₀ existential), conditional on five named validity premises." §3: "**OPEN — VALIDITY premises of Theorem (2):** OBL-D1-PROMOTE (H5 lane + D1; sub-obligations OBL-H5-JETMOD / OBL-H5-ZBAND / OBL-H5-REMOTE-THRESHOLD); D3-LEMMA-RN-UNIF (rung + uniform parts; foundations); PERC-DECAY (B1.dir-far, B2-far, B4.rem; percolation lane); OBL-B1-BRANCH(loop|B1) (branch-control lane; constant-level); **the B4.loc dam-line tube certificate** (… asserted-not-established — scope ruling V2 …)". `claims/graph.json` adds, of the rung: "D3-LEMMA-RN-UNIF is used at the rung only and is precisely stated, NOT closed."
* `D1_ASSEMBLY_v2_2_REGISTER_NOTE.md`: "**Scope:** register actions only — the frozen v2.2 body (490ad6b2…) is NOT altered; everything below takes effect at the next issuance (v2.3)."
* `D1_ASSEMBLY_v2_3_DRAFT.md`: "**Agent:** Claude (Anthropic family) acting for the assembly role while Kimi is dark (2026-09-15 → 09-30). **Status:** PROPOSED." — "v2.2 remains the last Kimi-issued form; this v2.3 becomes the current strongest form only upon operator promotion." — §5: "conditional on **TWO** named validity premises: OBL-D1-PROMOTE, D3-LEMMA-RN-UNIF."
* `H5_ZBAND_CONSUMPTION_2026-09-15.md`: "AUTHOR: Claude (Anthropic) · CLASS: OBL-DISCHARGE (H5 promotion lane, sub-obligation) · STATUS: PROPOSED · AUTHORITY: none · CANONICAL IMPACT: OBL-H5-ZBAND OPEN → DISCHARGED (consumption grade) upon promotion; no theorem statement changes; no frozen carrier edited."

The v2.3 draft and the H5_ZBAND note are **proposals**. The status this repository carries for Theorem
D1 v2.2(2) is the register's: five validity premises OPEN, `D3-LEMMA-RN-UNIF` not closed
(`docs/OPEN_PROBLEMS.md`, `claims/graph.json`). Mirroring the draft promotes nothing and discharges
nothing (CLAUDE.md rule 1); only an operator applying the exact licensing predicate moves a gate.

## How these bytes got here

Every object was fetched through the Drive connector (`download_file_content`, base64) and the
base64 was decoded from the session transcript straight to disk, so no model retyped any byte.
For a raw file the SHA-256 and byte count were recomputed from disk and had to equal the
`drive/inventory.jsonl` row (2026-09-17 snapshot) or the bytes were not stored; every stored raw
file here passed. A native Google Doc has no payload digest anywhere in the corpus: its text export
is stored as `<title>.export.txt` with `exact: false` (a reading copy), and where the export carries
full-line `BEGIN_*BODY` / `END_*BODY` markers the marker-delimited body digest is recorded and
compared with the digest the registers declare for that Drive id. `_MANIFEST.jsonl` holds one row
per object and `tools/verify_manifests.py` re-checks every digest in CI. Archives are not extracted
and nothing was executed; `<name>.zip.members.txt` is a derived member listing, not a Drive object.

## What this directory does not establish

Nothing here verifies, promotes, closes, discharges or reclassifies any claim, premise or obligation.
A digest match is identity of bytes; a body digest equal to the one the claim graph cites is identity
of the body, not a verdict on the theorem. The two-premise form of the v2.3 draft is not the
repository's status for D1(2); OBL-H5-ZBAND is not discharged here; the gate files (`d1_falsify_v3.py`,
`d1_falsify_v4.py`) and every carrier the ledgers pin are not mirrored here and were not re-executed.
No 2D bound is composed with the 3D track, no original prize problem is solved and no independence
credit is awarded.

## 2026-09-19 — `05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15/RN_UNIF_2026-09-16/`

`CL-RNU-001_RN-UNIF_ENGINE_STATUS_AND_CLOSURE_PLAN_2026-09-16.md` (12,291 B,
SHA-256 `488b1b0c…`, Drive `16o9u_KgysJv9lnR93PNXCqc-MdrW9QL6`), byte-exact
against the inventory. Its header: "STATUS: PROPOSED · AUTHORITY: none ·
CANONICAL IMPACT: NONE — the lemma is NOT closed by this document." It is the
source of the `chi2_grad_bound` slack record in `research/slack/registry.py`
("`chi2_grad_bound` = 1.57e14 with χ² = 1.94e-6"; "True |∇χ²| at (5,0) … 1.563e-5
(vs the engine's bound 1.57e14 — 1e19 slack)"; "the adaptive polar certifier is
defined and never invoked"; "its driver is likewise unwritten"). D3-LEMMA-RN-UNIF
stays OPEN; nothing here closes it.

## 2026-09-19 — lane completion pass (33 further objects, all byte-exact)

Drive lane `01_ACTIVE_RESEARCH_PACKAGES/2026-09-15 — KIMI FINAL INTAKE — UPPER2D STAGE E + H5 + ASSEMBLY`,
folder id `1bRmeImIMT-6kc-zBNZVoWW1aYNPBPLWs`. Subfolder ids, as the inventory records them:
`01_CURRENT_ASSEMBLY_AND_STATE` `1gFCj2IzPQGKOImqmAmesgYc7Z7RTiOnb` ·
`02_H4_EVENT_LEVEL_REPAIR` `1RXWIRH9Gdg5pj6IJssYVLG6dz-0scouG` ·
`03_H5_PROMOTION_AND_RUNG_CERTIFICATES` `1fVdPlwhtL8hxMiRoD6TxNJaNAHOCDy4y` ·
`04_STAGE_E_REVIEWS_AND_DEFECT_FINDINGS` `1DkIMa9N2gVooK-B_Zf1npvQCYUHtVpzs` ·
`05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15` `1J7Ly5v-XQWTT_AqqAGfsr1kdXmwyxvE3`
(`D1_v2_3_DRAFT` `14AGgILTCsrjLFvsEruDDnYaslekHEwzB`, `H5_ZBAND` `1aQbmEmiqATx50B9yDZiF4LfRe49zAfba`,
`RN_UNIF_2026-09-16` `1FmKnSQRpHc6EYCUEIl07FQDECUZsiEqG`) ·
`06_UNMIRRORED_FROZEN_CARRIERS_BYTE_NATIVES` `17HwJNlnMWavengx74rl9KDIqXOXyig2i`.
Nothing already in this directory was modified; rows were appended and this section added.

### The controlling status banners, verbatim

* `CL-STATE-001_CONSOLIDATED_STATE_2026-09-15.md`: "STATUS: PROPOSED — non-authoritative until the operator promotes it; supersedes nothing by itself" · "AUTHORITY: none · CANONICAL IMPACT: NONE until promoted" · "**Kimi is dark until 2026-09-30; the snapshot is final until then.**"
* `CL-OBL-001_H5-ZBAND_DISCHARGE_AND_RN-UNIF_WORK_ORDER_2026-09-15.md`: "STATUS: PROPOSED · AUTHORITY: none · CANONICAL IMPACT: NONE until the owning lanes consume/execute." · "NOTE 2026-09-16: … §2 and Appendix A are SUPERSEDED by CL-RNU-001 … Retained as the record of the reasoning that led there."
* `00_LANDING_NOTE.md`: "All PROPOSED, non-authoritative, no frozen carrier edited."
* `CURRENT_STATE_DELTA_2026-09-15.md`: "The matching 2D upper theorem is NOT closed at unconditional all-small-r theorem grade." · "historical v2.0/v2.1 failure findings remain provenance, not current authority." It is ported **as the dated 2026-09-15 return note it is**, not as the current premise list: `00_LANDING_NOTE.md` says it "is superseded by the register note in the same folder", and `CL-ERR-001` E6 calls it stale.
* `CL-ERR-001`: "STATUS: PROPOSED · AUTHORITY: none · CANONICAL IMPACT: NONE — **no theorem in the tree is weakened by any item**."
* `CL-AUD-001`: "STATUS: PROPOSED · AUTHORITY: none · CANONICAL IMPACT: NONE (confirms; asserts no new theorem content)" · "Nothing here bears on the mathematical validity of the frozen theorem bodies beyond what their own gates check; those bodies were not re-derived line by line."
* `CL-LEDGER-001`: "STATUS: PROPOSED (for the operator to append; the frozen ledger is not edited) · AUTHORITY: none."
* `CL-REG-001`: "STATUS: PROPOSED · AUTHORITY: none · CANONICAL IMPACT: NONE (record-integrity only; no theorem content)"; one claim is "**UNRESOLVED** — none of 9 candidate rules reproduces it".
* `CL-PIN-001`: "CAVEAT: these are the bytes present at snapshot time. They are NOT certified to be the bytes that generated the rung-2/3 banks." (the file breaks the line after "generated")
* `CL-GROK-CLOSE-001`: "This closes the **session work**, not D3-LEMMA-RN-UNIF, not D1 v2.3, not any prize problem." · "`NOT-CLAIMED` for all theorems." · "The research program itself is **not** CORE-CLOSED." · loose end 2: "Scale-T₄ cell CLOSE rows in `RNU_EXECUTE_RECEIPT.json` — must not be cited as certified cells."
* `CL-RNU-002`: "STATUS: PROPOSED / AUTHORITY: none · CANONICAL IMPACT: NONE (the lemma is not closed; this is certified-numerics infrastructure)."
* `CL-RNU-003_PIECE1_RUN_PIECE2_CHARACTERISED_2026-09-17.md` (quoted from the bytes fetched on 2026-09-19; the file is NOT stored here — its payload is the one quarantine key `Q-RN5-MOMENT-004` names, so its row is tree-only and these two sentences cannot be re-verified from this directory): "STATUS: PROPOSED · AUTHORITY: none · CANONICAL IMPACT: NONE yet" · "Nothing here changes a frozen carrier."
* `CL-RNU-003_T4_PUSH_2026-09-16.md`: "STATUS: PROPOSED. D3-LEMMA-RN-UNIF is NOT closed. AUTHORITY: none. Frozen engine not edited." · of its T₄: "This is a **candidate** … That implication is not proved."
* `RNU_EXECUTE_RECEIPT.json` / `.md`: `"status": "PROPOSED"`, `"lemma_closed": false`; "T4 used here is INTERNAL scale envelope 1.853e+4, not env_form."
* `RNU_T4_PUSH_RECEIPT.json`: `"status": "PROPOSED"`, `"lemma_closed": false`; "Wick/Bures/ratio 4-jets are NOT separately certified."
* `H5_RUNG2` / `H5_RUNG3`: "H5 — RUNG 2 certified (2026-09-15): r = 0.025, E_w(r)/r³ = 664.3979" / "RUNG 3 certified … 661.4712"; "New package; frozen documents unaltered."; "Falsifier does not trip at rungs 1–3." **They certify those rungs only.**
* `H5_PROMOTE_UPDATE_2026-09-15.md` §5: the dense κ = 1/8 modulus is "the displayed modulus (dense certified sampling + explicit fit); the CERTIFIED band enclosure (interval-r lattice sums) remains OBL-H5-JETMOD with its content unchanged."
* `H5_TOTALS_FREEZE_2026-09-15.md`: "NOTE (honest lineage): these bytes are the DRIFTED content …, not the v1 consumed content."
* `D1_V2_3_RECEIPTS.txt`: "STATUS: PROPOSED — becomes the current strongest form only upon operator promotion; v2.2 (490ad6b2…) remains the last Kimi-issued form."
* Stage-E reviews, their own dispositions: `REVIEW_proof.md` "DISPOSITION: **FAIL** — three independent grounds, each sufficient"; `REVIEW_provenance.md` "DISPOSITION: **FAIL**"; `REVIEW_scope.md` "**DISPOSITION: FAIL** — three exact scope violations"; `REVIEW_topology.md` "Dispositions: targets 1–5, 7 PASS; target 6 FAIL (exact lemma, exact gap, counterexample, and repair all stated above)." (target 6 is the H4-JC joint carrier; the file breaks the line after "exact lemma,"); `REVIEW_numerics.md` "F-1 (FAIL-grade, certification-chain gap on the consumed remote constant)". All are reviews of the **frozen D1 v2.0** package.
* `H4_JC_EVENT_LEVEL.md`: "Theorem H4-JC as frozen in H4_CLOSURE.md §2.3 (body `a67d50b9…`) is FALSE as stated" (joined across the file's line break; the file has no backticks around the file name) · "Frozen carriers are untouched."
* `PERC_DECAY.md`: "**The o(r³)-order reading of the far lanes is NOT reachable with this lane's machinery, and the registered percolation-decay input would not deliver it either.**" · "The far lanes are **Θ(r³), not o(r³)**".
* `CL-MIRROR-001_MANIFEST.sha256`: "These are NOT new artifacts … Any copy landed here must re-hash to the value on its row before it is cited."

Every "DISCHARGED", "CLOSED", "ADJUDICATED YES", "TWO premises" or "current strongest form" sentence in
these documents belongs to that PROPOSED tier or to a register note "effective at the next issuance".
No operator has applied any of them here. The frozen `D1_ASSEMBLY_v2_2.md` body (`490ad6b2…`) keeps **all
five validity premises OPEN**, and `D3-LEMMA-RN-UNIF` is not closed.

### What was ported, and what deliberately was not

Ported byte-exact (33 objects this pass; digest and byte count recomputed from disk and equal to the
2026-09-17 `drive/inventory.jsonl` row in every case): the four lane-state documents (`CL-STATE-001`,
`CURRENT_STATE_DELTA`, `00_LANDING_NOTE`, `CL-OBL-001`); the six H5 rung notes in `03_`; the six
RN_UNIF documents and receipts in `RN_UNIF_2026-09-16/`, including `rnu_execute.py` stored as
`rnu_execute.py.txt`; the remaining `05_` CL documents plus `D1_V2_3_RECEIPTS.txt` and
`FREEZE_H5_ZBAND.txt`; the five Stage-E reviews and the two H4 event-level repair files; and
`PERC_DECAY.md` with `CL-MIRROR-001_MANIFEST.sha256` (stored as `…​.sha256.txt`) in `06_`.

Deliberately not ported, recorded as `stored:false` rows in the manifests:

* `d1_falsify_v4.py` (`1uTcWaYLtJUszT7iBEWzI1Xa6J7_nw9E6`, 20,558 B) — the gap that wants these bytes
  targets `engine/carriers/blobs/`, which another agent owns; this mirror writes only under its own
  directories, so the carrier gap stays open rather than being relocated here.
* `09152026OKComputer_Project_Gap_Closure.zip` (`1vSI-evINWskhXVyiZ0slt-rLT74sPpXH`, 30,148,285 B) —
  far over the per-file and per-lane size caps, and the audit's own list says "index and member digests
  only". No member was recovered, so the archive-member premise sources the claim graph cites
  (`ADDENDUM_2026-09-15_H3_BAND_FLOOR.md`, `ADDENDUM_2026-09-15_B4LOC_PERC_DECAY.md`,
  `B4LOC_DAMLINE.md`, `H3_BAND_CEIL.md`, `H5_PROMOTE.md`) remain unmirrored.
* The RN_UNIF-folder duplicate of `CL-GROK-CLOSE-001` (`1Y_3zFonLsFIAHP5KSkUfsJXqHZXIUAL2`,
  quarantine `Q-R17-DUP-001`, EXACT_DUPLICATE) — only the keeper `1Hc8…` is here.
* `rnu_ds3_scalar_SUPERSEDED.py` (`1v7JAh_FbXHYMaL7W6DNwEoLnHPwQ5U21`, quarantine `Q-R17-RN-OLD`,
  SUPERSEDED) — not mirrored and not consumable; "Source portfolio explicitly retires scalar DS3."

`CL-RNU-003_PIECE1_RUN_PIECE2_CHARACTERISED_2026-09-17.md` is **not** stored, and its row in
`RN_UNIF_2026-09-16/_MANIFEST.jsonl` is `stored:false`. It is under quarantine `Q-RN5-MOMENT-004`,
class DEFECTIVE_SCOPE, scope "Section 3 claim that the near integrand with sqrt(E dy^4) is certified, and
the derived near/remote forecasts. Far progress and declared partial coverage are retained." The audit's
gap list suggested mirroring the bytes with the exclusion key attached, but `tools/quarantine_check.py`
invariant 3 forbids an excluded payload digest from appearing in **any** repository manifest, and that
firewall outranks the suggestion: the bytes were fetched, hashed to the inventory digest, and then
removed rather than carried. The exclusion is enforced here, not annotated. Its successor object is
mirrored in `drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM/RN5_NEAR_MOMENT_REPAIR.md`.

Two body digests were reproduced from the stored bytes and agree with the digest their own lane
declares elsewhere: `PERC_DECAY.md` → 7,409 B `5137a811…` under rule `marker_raw` (the rule
`CL-REG-001` records for it, "marker_raw only"), and `H4_JC_EVENT_LEVEL.md` → 16,241 B `42ee88da…`,
the value in the sibling `H4_JC_REPAIR_FREEZE.txt`. `registers/json/frozen_objects.json` declares no
BODY digest for either Drive id, so `body_matches_register` stays null in the manifest rows; a body
match is identity of the body, not review.

### What this pass does not establish

Nothing here verifies, promotes, closes, discharges or reclassifies any claim, premise or obligation.
A SHA-256 match is identity of bytes; it says which bytes exist, not that any bound holds. The
PROPOSED layer these documents carry — two-premise D1, OBL-H5-ZBAND "DISCHARGED at consumption grade",
B4.loc "CLOSED", B4.rem "ADJUDICATED YES", PERC-DECAY "RESTATED" — has not been applied: Theorem
D1 v2.2(2) remains CONDITIONAL on its five named validity premises, `D3-LEMMA-RN-UNIF` Piece 1 and
Piece 2 remain OPEN with receipts at `lemma_closed: false`, and OBL-D1-PROMOTE, OBL-H5-JETMOD,
OBL-H5-ZBAND and OBL-H5-REMOTE-THRESHOLD remain OPEN. The H5 rung certificates certify their own rungs
and discharge nothing. The Stage-E reviews are same-program reviews at zero organizational-independence
credit, their FAIL verdicts are on v2.0/v2.1 and are provenance rather than current authority, and no
review route moves. `rnu_execute.py.txt` and the two zips are inert bytes: nothing was executed,
nothing was extracted, no test, workflow step or import in this repository reads them, and no bound
carrier was registered. No 2D upper object here is composed with the 3D SIDE24 track
(CL-STATE-001: "3D — separate family … firewall holds"), and no original prize problem is touched.

Reading copies (`exact: false`) are not the objects: the only non-exact entries in this directory are
the derived `*.zip.members.txt` listings, which are written here and are not Drive objects. A PDF
rendering of any of these documents would not be a frozen body either; the frozen body is the
marker-delimited byte range of the object itself, under the rule its own lane names.
