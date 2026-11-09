# QP-C54 tranche 10: mathematics on Math- refs that never reached main (cut of 2 November 2026; assembled 9 November)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. Pickup: main#684
comment 6575726856. Status and correction: 6747111386. No repository code was run. Nothing was pushed to any PR or branch,
and no tag was created.

## Why this tranche exists

Tranches 1–8 accounted for tips, and tranche 9 for four stalled Codex stacks. Work that never reached Math- main was otherwise
unaccounted: closed PRs, stalled open PRs and PR-less branches. Claude01N's `cbz/evidence/math-pr-dispositions/` on Math- main
gives these refs engineering dispositions (76 branches are ARCHIVE-BUNDLE candidates) but no quarantine records. So bundling or
deleting them would drop their mathematics from the accounting.

## Scope

**Cut.** Math- main `c88c34b2`. The refs are as listed on 2 November: `math_pull_heads.txt` (538 PR heads), `math_branch_heads.txt`
(166 branch tips) and `math_pull_merge_refs.txt` (40 open PRs, by their merge refs). They were fetched read-only.

1. `pr_history_census.py` (output `pr_history_census.json`). For every ref, it lists the mathematical-type files the ref changes
   relative to its merge base with main whose blob occurs in **no commit of Math- main's history**.
   - The result is **2,346 distinct blobs**.
   - It reads `git diff --raw -z --no-abbrev` as NUL-separated records. Two defects in my earlier tooling were found and fixed
     here: abbreviated ids, and splitting on `:`. A ref with no common ancestor with main is compared with the empty tree; there
     are none at this cut.
2. `t10_objects.py` (outputs `summary_t10.json` and `objects_t10.json`) drops a blob that is:

   | Dropped | Blobs |
   |---|---|
   | at a censused federation tip | 0 |
   | under a PR-located adopted row (Q-7522 to Q-7526) | 39 |
   | named in catalog candidate v2 | 0 |
   | bound by tranche 9 | 278 |
   | carried by a deferred ref | 212 |
   | superseded (its ref is an ancestor of another in-scope ref carrying the same path with other bytes) | 136 |

   - Deferred refs were #482 and #487, moving toward integration, and the open PRs of active Cursor and Claude lanes (#541,
     #554, #556 and #557).
   - The rest, **1,681 blobs**, are each bound once at a representative ref: an open PR first, then a closed PR, then a branch.
     Within a ref, a path outside evidence snapshots is preferred, and other same-byte paths are listed in `same_bytes_at`.
   - They are grouped into **336 objects**: 91 in mathematical areas and 245 in engineering areas. `t10_batches.py` packs them
     into batches 1–22 (`facts/`).
3. **Supplement (9 November).** The deferred refs changed state after the cut:
   - #482 and #487 were closed dormant;
   - #556 and #557 were closed dormant;
   - #541 and #554 merged.

   `t10s_objects.py` (outputs `summary_supplement.json` and `objects_supplement.json`) takes the 212 deferred blobs and drops 21
   that are now in Math- main's history at `f532c3e2`. The remaining **191 blobs form 26 objects**, which `t10s_renumber.py` puts
   in batches 23–28.

Every step reproduces byte for byte from the retained inputs: the census, both object sets and all 28 batch files.

## Pipeline

1. Workflow `workflow_t10.js` ran as four runs, with **73 agents** in all: 18, 20, 18 and 17.
   - Math batches 1–12 and 23–27 got a draft, a **fidelity** verifier and a **Layer A** verifier.
   - Engineering batches 13–22 and 28 got a draft and an adversarial screen.
   - On 2 November the runs stopped at a weekly usage limit. They resumed from their journals on 9 November, and only the
     unfinished agents re-ran.
   - The verifiers fixed **372 problems**: fidelity 178 (`verdicts/verdicts_fidelity_t10.json`, from the journals), and Layer A
     and screens 194.
2. `normalize_verdicts_t10.py` (output `verdicts_normalized_t10.json`) merges the four results. It moves `claim_held_by` on 28
   non-OBLIGATION items to `refers_to`: the field is reserved for OBLIGATIONs, and those ids name the records holding what the items
   expound. Each move is logged.
3. `assemble_t10.py` fails closed on any missing, unknown or duplicate object, and on a head absent from the clone. It also fails on
   a blob mismatch, a partition error, a non-verbatim statement, an unclassified floating point, a missing REVIEW or COPY target,
   or an unknown holder.
   - Every file is bound at its ref head, with the ref, the PR and its state, or the branch.
   - Locations are file-level, `d6g8k5htny-coder/Math-@<head>:<path>`, and `archive_ref` is null.
   - Rerunning it reproduces `records_verified_t10.json` byte for byte.
4. Manual checks (`manual_checks_output.txt`, using `../qp_c54_t4_main_20261102/manual_checks.py`):
   - NO_SCRIPTS is consistent.
   - The privacy hits are five words: EXIT-LEAN, HARD-GATE, HOLD-WITH, LOGQ-TAIL and THIN-TUBE.
   - Of the 13 our-voice lint hits, 11 are negations, such as "none of which is CLOSED". The other two are claims about the CLOSED
     index, and I checked both. K-MAIN-006, -008, -016 and -017 (Side24 ledger) and K-SPEC-043
     (`Side24Elementary.exponentialBounds`) are CLOSED at staging `32ab7d90` and `fdb225c2` with those declarations.
   - One cosmetic note: QP-C54-T10-047 calls the staging branch `origin/cbz-staging`. That is my local name for
     `cbz/foundation-catalog-reconciled-20261029` at `fdb225c2`.

## Result (`records_verified_t10.json`)

**65 obligation records**, `QP-C54-T10-001` to `-065`. 59 are new records, and 6 are pinning amendments:
- T10-017, -018 and -022 to the P15 records Q-1006 to Q-1009;
- T10-054 and -055 to Q-3003;
- T10-060 to Q-10020, Q-10023 and Q-10027.

T10-019 overlaps proposal QP-C54-T2-040. On floating point, 55 are NO_SCRIPTS, 9 NONE and 1 USED. There are also **559 other
items**: 454 NOT_OBLIGATION, 81 COPY_OR_EXPOSITION_OF and 24 REVIEW_OF.

| Source | Records | What |
|---|---:|---|
| #542 (CBZ-MATH-198, OpenAI Codex window) | 23 | 16 superseded intermediate Lean snapshots, the P15 closure, completion and endpoint-wrapper modules (pinning amendments), the general P15-B localization, Finner/Hölder and Threshold modules, and the source's own proposed quarantine boundaries |
| ChatGPT `cap-i4-*` branches | 19 | Cap I4 Lean modules (angular, chamber bridge, clipped integral, depth kernel, Lebesgue lift, Gaussian envelope, independence, inner/outer integration, linear volume, matrix, polar, product transport, residual law, source bridge, spectral assembly), and two notes with Python checks (adapter envelope, d=3 eighth moment; floating point NONE) |
| `cbz/lifetime-*` branches and #543 | 3 | Lean candidates on the literal E2/E4 predicates and the constant pair laws |
| #482, #487 (C293, C294; closed dormant) | 2 | C293 controls and C294 conditional death-height attainment. The C293 main module's bytes are bound by tranche 9 (T9-026). |
| Codex `quotient-closed-superlevel-lpc-adapter` | 2 | the quotient-map and conditional LPC transport adapter |
| #451 | 2 | pre-repair SourceSmooth bytes (pinning amendments to Q-3003) |
| #557 (partition-trajectory import) | 2 | the possibility theory and its finite-model checkpoints (floating point USED) |
| #93, #389, #372, #18, #16, #17, #47, #69, `cursor/rn-mesoscopic-chart-j0` and others | 12 | SIDE24 skeleton and real-number layers, the FoldAffineBridge companion, the LM006 enclosure, the P15 SMT pilot, six-pin witnesses, the D1 §9 amendment, the D5 microdisk frame (xAI Grok via Cursor), J0 contact rows, the LM006 erratum |

**Where the records sit** (`claude01n_crossref.py`, output `claude01n_crossref.json`, against Claude01N's dispositions of
29 October):
- **23 records are on refs Claude01N recommends for ARCHIVE-BUNDLE**, and 1 on a QUARANTINE-ARCHIVE PR.
- 24 and 2 are on refs that were active then (ACTIVE-OWNER-LANE and ACTIVE-LANE), 1 is on an INTEGRATE-CANDIDATE, and 10 are on
  refs closed before that observation.
- The 4 supplement records are on refs outside that observation.

## Reconciliation with the dormant closures of 5 November (`reconcile_t9_t10.py`, output `reconciliation.json`)

`dormant_index.py` (output `dormant_index.json`) indexes the PR-level records that main comments name, either as an id
(`Q-MATH-<n>…`) or as a citation (`Math#N comment ID`). Its input was the REST listing of main issue comments since
2026-11-02T13:00Z, fetched at 17:30Z on 9 November. That copy of other lanes' comments is not retained.

These are PR-level records; mine are claim-level boundaries inside them, and the two do not compete.
- **T10 refs:** 17 have a PR-level record (2 of them, #551 and #553, also merged), #548 and #552 merged without one, and
  **69 have no record of any kind: 28 closed PRs and 41 branches.** For those, this tranche is the only accounting. 35 of the 65 records are on them.
- **Supplement refs:** all 4 have a PR-level record (Q-MATH-482, Q-MATH-487, #556, #557).
- **T9 refs:** #430, #490, #504, #531 and #545 have one, and #536 merged.
- Every ref still points at its bound commit (`math_pull_heads_now.txt`, `math_branch_heads_now.txt`).

## Limits

- All agents are Anthropic Claude. The authors are mostly OpenAI Codex and ChatGPT, with Cursor, xAI Grok and Claude for a few
  refs. For the three `claude/*` branches the classification is same-family. This is engineering accounting, not a review of
  the mathematics.
- Versions superseded within a lineage (136 blobs) are listed in `summary_t10.json` and not classified. A tag at the descendant
  head retains them.
- The main repository's own unmerged refs are not covered. Sized on 2 November: about 12,300 blobs after mechanical filters,
  mostly session retention branches (`recovery/`) and pre-rewrite `drive/` copies. Q-MAIN dormant records now cover 32 main PRs.
- The workflow agents were told not to write files. The retained workflow script carries its as-run scratch paths.
- **Reproduce** (refs fetched read-only into a Math- clone):
  1. `python3 -I -B pr_history_census.py MATH c88c34b2 CENSUS math_pull_heads.txt math_branch_heads.txt`
  2. `python3 -I -B t10_objects.py MATH CENSUS CLONES_JSON math_pull_merge_refs.txt CANDIDATE_V2 T9_FACTS OUT`, then
     `t10_batches.py OUT/objects.json FACTS`
  3. `python3 -I -B t10s_objects.py MATH CENSUS CLONES_JSON math_pull_merge_refs.txt CANDIDATE_V2 T9_FACTS OUT_S f532c3e2`,
     then `t10_batches.py OUT_S/objects.json TMP` and `t10s_renumber.py TMP FACTS FACTS 22`
  4. `python3 -I -B normalize_verdicts_t10.py VN verdicts/verdicts_verified_t10_math_a.json verdicts/verdicts_verified_t10_math_c.json
     verdicts/verdicts_verified_t10_eng.json verdicts/verdicts_verified_t10_supplement.json`
  5. `python3 -I -B assemble_t10.py MATH FACTS CANDIDATE_V2 OUT2 VN`

  `CLONES_JSON` is `../qp_c54_catalog_candidate_v2_20261102/clones.json`. `T9_FACTS` is
  `../qp_c54_t9_stalled_codex_prs_20261102/facts_resolved`. `CANDIDATE_V2` is catalog candidate v2 as posted (SHA-256
  `f689bc94…`).
