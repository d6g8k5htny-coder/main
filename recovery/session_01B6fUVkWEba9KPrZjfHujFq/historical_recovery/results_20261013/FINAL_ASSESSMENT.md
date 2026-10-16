# Final assessment: session 01B6 historical recovery

**Scientific effect: NONE.** This assessment classifies the historical record. It does not change the status of any theorem, lemma or register item.

## Shares (copied from assessment_counts.json)

| Share | Classes | Gaps |
|---|---|---|
| Recovery / custody (located, cross-link or carrier hunt) | B, F, G, H, I | **38 / 155** |
| Partially solved previously | C | **73 / 155** |
| Open (genuinely open, or reconstructable only) | A + J | **31 / 155** |
| Superseded or refuted | D + E | **13 / 155** |

By class: A 27, C 73, D 12, E 1, F 2, G 24, H 1, I 11, J 4. No gap is class B. No case showed that a complete solution once existed while its carrier is now missing. The recovery/custody share includes the I gaps, which are referenced but still unlocated.

## Blocker most likely already solved somewhere: PREMISE-Z-LOWER (G98)

The normalizer floor Z_r ≥ c r² on (0, 0.025] at b=6/5 blocked the Q0-C101/C103 cubic-rate route (main#116). Its pieces already exist, accepted, in GitHub:

- The r→0 half is the parent's existential floor (5.5), proved in Math- blob dfed3b8d and accepted before the snapshot in main#112 comment 5841352633 (item N6). It is also D4 RN-FIXED-REMOTE-WINDOW (11), Math- blob b383bfcc, accepted in main#76 comment 5841783172 at proof-sketch depth. Post-snapshot, the Anthropic C1 review accepts the same floor.
- The separated range [δ, 0.025] is derived from C102 in main#121 comment 5842152602, the very comment that declared the premise BLOCKED. It says the two halves join into one constant on the whole interval.
- The same theorem is stated a third time as P02-LM-006 / LCR-DER-019 (duplicate class F, G123).

What is missing is a composition record bound to the C103 law with the direction t fixed (C103 leaves t free), plus one review. The verifier moved the gap from H/P0 to G/P1 (present in GitHub, not cross-linked). Dissent on record: the live-Drive lane's verifier would keep it at C, because no single carrier states the floor on all of (0, 0.025]. The explicit-radius Drive certificate H3_BAND_FLOOR is **not** established: it shares the refuted iv_Phi tail-conversion step. The Drive-only H3_CLOSURE.md (custody sha256 387e4bae…) is self-graded and existential.

Runner-up: the far-zone input jets for D3-LEMMA-RN-UNIF Piece 1 (ENV-RESCOV G04, LOGQ-TAIL G06). RN-FIELD-001 and its certificate (custody a19382f9…, hardening blob c4065c67) carry them at r=1/20. They are author-side and never replayed, so they are a candidate, not a closure.

## Blocker that appears genuinely open: the Piece-2 annulus cover (G03)

D3-LEMMA-RN-UNIF Piece 2 needs a certified, non-overlapping cover of the near annulus 0.1 ≤ ∣y∣ ≤ 5. No historical cover exists:

- CL-RNU-003 §3 records that the crude Taylor-model bounds are unusable inside d < 3 (custody 59b8f002…, verified exact).
- The only older route used the crude-spine envelope_v, which RN5 shows has the wrong determinant-moment power (custody ac89f60b…, verified exact).
- RN5 supplies the corrected integrand and machinery, with a few boxes and point certificates. It states that these are not a coverage certificate.

G03 gates the remote threshold (G20), the D1 rung display (G22), D3-LEMMA-RN-UNIF itself (G01) and the conditional D1 v2.x upper chain (G23, G102). Other genuinely open items in the same lanes: ALLCELL-FDZ-Q4 (G05; its filename was coined on 2026-09-22, and no historical dump is known); a certified C_H on shrinking charts (G08); the H5 F-map and the joint StationBox cancellation (G16, G17); H5-RIM flatness (G35); the D2 numerical constants (G58); Theorem B's TB-G6/G7 (G54).

## Five highest-value artifacts still worth locating

1. **CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip** (G13; feeds G01, G02, G06, G07, G10). This chat-delivered bundle holds rnu_t4.py, rnu_certify.py, certify_cells.jsonl, certify_queue.json, rnu_spine.py and the run traces of the historical Piece-1 certification. It is not in any git ref, the custody store, the recovered Dropbox texts or live Drive, and main PR #200 did not recover it. Look in the owner's chat exports of the 2026-09-17 session.
2. **The RN_UNIF_2026-09-16 folder of CL_ANTHROPIC_BUNDLE_2026-09-16_v3.zip** (G12, G14): rnu_env.py, rnu_white.py, rnu_diag.py, rnu_land.py, rnu_fdcheck.py, rnu_dcheck.py, rnu_run1.txt and the transcripts. The bundle's other natives are already on origin/main; these were never uploaded. Look in the chat exports of 2026-09-15 to 2026-09-17.
3. **Dropbox file `OKComputer_09132026Project_Gap_Closure.zip`** (24,581,995 bytes; uploaded to the owner's Dropbox on 2026-09-28; not in the 2026-09-27 inventory). The verifier's download was refused by the session's network policy (the egress proxy returned 403 for the Dropbox content-download host). No bytes were received. Its size matches no custody zip, so it is a distinct, unexamined object. Scan it for the nine R2 expected-byte digests (G38), the 9,700-byte Module I extraction b958c21d… (G140), and the RN-UNIF carrier names above.
4. **Dropbox file `OKComputer_Project_Gap_Closure.zip`** (23,251,839 bytes; uploaded 2026-09-28; same situation and same scan targets as item 3). To proceed, either the environment's allowed domains must include the Dropbox content-download host, or the owner places both zips where the session can read them.
5. **The 2026-09-22 local-workspace receipts** (G07, G11, G08): rnu_white_box_grad_bound_v1.py, RN_UNIF_WHITE_BOX_GRAD_BOUND_V1_RECEIPT.json, RN_UNIF_LOGQ_HESS_SKELETON_RECEIPT.json and RN_UNIF_LOGQ_CH_COMPOSE_CERT_GATE_V1_RECEIPT.json. They would define SYM-Fw-jet, back the q=2 "CERTIFIED" box claim and source the C_H probe values. They were never vendored into any repository.

Also unlocated, and lower in value or likelihood: the File 3 source PDF that should hold Table 4.1 / Appendix A (G47); the main#58 separating-cap proof file (sha prefix a1f500c8; its successor cap is in Math-); the GP-DATA-054 v1.1 source (G133); the MPV3 state carriers (G152); the rn-sector-review-master-20260921 packet, reported present in Dropbox but never saved (G33).

## Limits

- **Snapshot.** The gap list is the 2026-09-28 snapshot, and every class describes the record as of that date. The search corpus (all git refs of the program repositories, the DG-MIGRATION-20260919 custody store, and the Dropbox texts recovered on 2026-09-27) was built on 2026-09-28. Later GitHub records are noted in the "since snapshot" column. The only class they change is G46 (C→D), by explicit verifier override.
- **Search by models.** Language-model agents ran the search with regex and semantic queries over the corpus, plus read-only GitHub, limited live-Drive and live-Dropbox metadata queries. Several carriers travelled only as chat-delivered bundles, so "not found" does not prove "never existed". The two large Dropbox zips above and the contents of live Drive zips were not examined.
- **Verification adversarial but same-provider.** Every mandatory fragment was re-opened, quotes were checked verbatim and hashes were recomputed. But the verifiers belong to the same model family as the searchers, and the session running this recovery also wrote some of the records it audits (for example the PART3/PART4 upload reviews). Agreement between models is not evidence of correctness. Across the cited material, all reviewer lanes (Anthropic, xAI via Cursor, Moonshot Kimi, OpenAI) operate on one owner account, so none carries organizational-independence credit.
- **No scientific status changes.** Nothing here promotes, demotes, closes or reopens a theorem or register row. Reconstructions are labelled RECONSTRUCTED CANDIDATE — NOT ORIGINAL CARRIER and are never counted as recovered. The recommended next actions need owner or reviewer decisions.
- **Locators.** Drive file ids and Dropbox ids are withheld. Drive-only objects are named by their custody sha256 when the custody store holds them.
