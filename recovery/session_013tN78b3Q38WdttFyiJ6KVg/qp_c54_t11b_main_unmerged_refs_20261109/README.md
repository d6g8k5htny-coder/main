# QP-C54 tranche 11b: mathematics on main refs that never reached main's history (cut of 2 November 2026; assembled 9 November)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. Pickup: main#684
comment 6748868324. Checkpoint: 6751848095. No repository code was run. Nothing was pushed to any PR or branch, and no tag was
created.

## Why this tranche exists

Tranche 10 accounted for Math- refs that never reached Math- main. This tranche is item (2) of the proposal in main#684 comment
6748785075. It applies T10's method to the **main** repository's own PR heads and branch tips.

## Scope

**Cut.** main `d2b0216e`. The refs are as listed on 2 November and were fetched read-only:
- `main_pull_heads.txt`, 689 PR heads;
- `main_branch_heads.txt`, 292 branch tips;
- `main_pull_merge_refs.txt`, the open PRs, by their merge refs.

Three PRs opened later (#780, #781, #783) are outside the cut.

1. `pr_history_census.py` is T10's script, run on main. For every ref, it lists the mathematical-type files the ref changes
   relative to its merge base with main whose blob occurs in **no commit of main's history**.
   - 333 refs have no common ancestor with main. These are heads from before main's history rewrite, and they are compared with
     the empty tree.
   - The result is 582,655 file entries: **12,817 distinct blobs**.
   - The full census (163 MB, SHA-256 `97e82d77…`) is not retained; step 1 regenerates it. `filter_census.py` reduces it to the
     entries the object builder uses: `main_history_census_math_areas.json`, with 55,761 entries and 1,453 blobs, keeping every ref
     row. The builder's output is byte-identical on either census.
   - **Excluded by rule.** The blob counts overlap, since one blob can sit in several directories:

     | Excluded | Blobs | Note |
     |---|---:|---|
     | `recovery/` | 6,571 | session retention; item (1), for the owning sessions |
     | `drive/` | 2,549 | item (3) |
     | `cbz/` | 1,633 | engineering, as in T4 |
     | `tests/`, `tools/` | 311, 210 | |
     | root files | 142 | |
     | `governance/`, `docs/site/`, `architecture/`, `incoming/`, `data/`, `.github/`, `docs/retro/` | 30, 24, 10, 7, 6, 5, 5 | |

     No `.sage`, `.ipynb`, `.m` or `.jl` file lies outside these directories.
2. `t11b_objects.py` (outputs `summary_t11b.json` and `objects_t11b.json`) drops a blob that is:

   | Dropped | Blobs |
   |---|---:|
   | at a censused federation tip | 4 |
   | under a PR-located catalog row | 31 |
   | named by catalog candidate v2 or a QP-C54 tranche file | 8 |
   | carried only by my own retention refs (my branch and its closed PR #441) | 0 |
   | superseded (its ref is an ancestor of another in-scope ref carrying the same path with other bytes) | 116 |

   - The scan of QP-C54 files skips this tranche's own directory, so a rerun after retention gives the same output.
   - The other **1,294 blobs** are each bound once at a representative ref: an open PR first, then a closed PR, then a branch.
     Within a ref, a path outside evidence snapshots is preferred, and other same-byte paths are listed in `same_bytes_at`.
   - They form **262 objects on 136 refs**: 107 PR heads and 29 branches. 187 of the objects sit on pre-rewrite heads.
   - 74 objects are in mathematical areas (434 files, 11 Lean) and 188 in other areas (860 files, 4 Lean). `t10_batches.py`
     packs them into math batches 1–12 and engineering batches 13–20 (`facts/`).

## Pipeline

1. Workflow `workflow_t11b.js` ran **52 agents**.
   - Each math batch got a draft, a **fidelity** verifier and a **Layer A** verifier.
   - Each engineering batch got a draft and an adversarial screen.
   - The verifiers fixed **363 problems**:
     - fidelity 136 (`verdicts/verdicts_fidelity_t11b.json`);
     - Layer A 116: 58 in `verdicts_verified_t11b_math_a.json` and 58 in `_math_b`;
     - screens 111 (`_eng`).
2. `normalize_verdicts_t10.py` (output `verdicts_normalized_t11b.json`) moves `claim_held_by` to `refers_to` on 3 COPY items. Each
   move is logged.
3. `fix_line_ranges.py` (output `verdicts_fixed_t11b.json`) repaired 2 statement ranges, at `docs/RN_BERNSTEIN_SHARP.md` 7-26 →
   7-25 and a `CL-DATA-051` jetbox script 3-18 → 3-19.
   - Each statement's text matched exactly once within ±3 lines.
   - Only the range moved; no text is ever changed.
4. `assemble_t11b.py` is `assemble_t10.py` with ids `QP-C54-T11B-NNN` and the repository set to main.
   - It has the same fail-closed checks.
   - Every file is bound at its ref head, with the ref and the PR and its state, or the branch.
   - Rerunning it reproduces `records_verified_t11b.json` byte for byte.
5. Manual checks (`manual_checks_output.txt`, from `../qp_c54_t4_main_20261102/manual_checks.py`):
   - NO_SCRIPTS is consistent.
   - The privacy hits are seven identifiers: C098-C102, CORR-CONG, HOLD-WITH, LOGQ-TAIL, OP01-OP15, OP02-OP05 and RUNG-OPEN.
   - I read all 8 our-voice lint hits:
     - Five are negations, such as "None of these is CLOSED" and "No uniform-in-r bound … is established".
     - T11B-063 quotes the source README under A7.
     - T11B-083 says the interval library "proves its enclosure lemmas in docstrings". That describes the source's informal
       docstring arguments, which the record's blocking reason calls informal and unverified.
     - T11B-109 "proved by `decide`" describes `AttemptDecide.lean`, whose declarations the source expects to fail.

## Result (`records_verified_t11b.json`)

**110 obligation records**, `QP-C54-T11B-001` to `-110`, all QUARANTINED.
- 106 are new records.
- 4 are pinning amendments: T11B-056 and -058 to Q-8340, T11B-057 to Q-8336, and T11B-059 to Q-2014.
- Floating point: 59 NO_SCRIPTS, 32 NONE and 19 USED.

There are also **415 other items**: 292 NOT_OBLIGATION, 74 COPY_OR_EXPOSITION_OF and 49 REVIEW_OF. Records sit on 30 of the 136
refs; the other 106 refs carry only other items.

| Source | Records | What |
|---|---:|---|
| #395 (closed unmerged, pre-rewrite) | 47 | the SIDE24/RN packet across `docs/`, `engine/carriers`, `engine/rn_engine` and `research/`: the LPW amplitude and constant chain, the RN Bernstein and sharp-variance successor, wedge and rectangle bounds, the density-determinant majorant, H3 bands and floors, Hermite–Gaussian envelopes, the exact-rational interval library, and the SIDE24 theorem package v1.0 |
| #7 (pre-rewrite) | 9 | P15 successor macro composition, P10-A substitution, P11-A/B/D obstruction covers and the P11-D endpoint erratum |
| #429 | 7 | exact rational controls for fold isolation, the finite Fourier critical count, the repeated-spectrum obstruction and the local-norm diagonal; the Gaussian conditioning extension; the all-frame one-contact floor. They cite Q-8429, whose location covers only the proposal files. |
| #603 | 4 | V22 band analysis and generic assembly, the generic-height mean, the joint height estimate |
| #606 (open) | 4 | three pinning amendments (Q-8336, Q-8340, Q-2014) and weighted cusp sampling |
| #8 (pre-rewrite) | 4 | full-height RN sector expansion, full-mark projection energy bound, annular-sector geometry, sharp-variance majorant input copy |
| #171, #238, #385 | 3 each | the RN inner-wedge bound, the Q0-C103 successor and the V3.4 ledger checks; the RN5 Hölder envelope reimplementation, the Hermite absolute-value envelope and the env_form envelope shape; the global H0 selector, analytic guards and falsification contract |
| #29, #664, #674, #728, `custody/dot-v19-gap2-originals-20261026` | 2 each | the JETMOD and RN-UNIF notes; the H0 falsifier design and prediction; the V22 two-height sector and its SymPy check; the elder tether countermodel and a whole-field Lean kernel; `VarianceBounds.lean` and the Gaussian moment certificate |
| 16 other refs | 1 each | including two `agent/*` branches holding conditional short-lifetime laws, and #457's validation report for the retired 2025 framework |

## Reconciliation (`reconcile_t11b.py`, output `reconciliation_t11b.json`)

The check is against main `5ebe5771` and T10's `dormant_index.json`, which indexes the Q-MAIN PR-level records named in main
comments up to 17:30Z on 9 November.
- **13 refs have a Q-MAIN record:** #343, #345, #346, #347, #349, #385, #595, #674, #690, #703, #724, #728 and #732. They hold 10
  records. These are PR-level records; mine are claim-level boundaries inside them, and the two do not compete.
- **123 refs have no record of any kind, and they hold 100 records.** For those refs, this tranche is the only accounting.
- **81 records sit on 80 pre-rewrite heads**, whose content exists nowhere in main's current history.
- **4 records are on branch-only refs**, which are lost with the branch unless tagged: `agent/observer-notes-20261001`,
  `agent/paper1-note-current-20261001` and `custody/dot-v19-gap2-originals-20261026`.
- Every ref still points at its bound commit except #720, which merged after the cut. #720 holds no records. Its one item, a
  NOT_OBLIGATION auditor document (`docs/cbz/G11_AUDITOR.md`), now has its bytes in main's history, and tranche 11a accounts for
  the tip's version.

**Other main-side lanes.** Claude S's MAIN-UNMERGED-MATH-ACCOUNTING (main#738 comments 6748297589 and 6749495978) is custody
discovery: where bytes live and how durable they are, with no dispositions. This tranche classifies. Their Drive-side post
(6751303676) is input to item (3). From metadata only, it compares the owner's Drive files with git by name and size, and finds
57 code files that exist only on Drive. Whether main's 2,549 `drive/` blobs are all on Drive is still open.

## Limits

- All agents are Anthropic Claude. For the two `claude/*` branches among the refs, the classification is same-family. This is
  engineering accounting, not a review of the mathematics.
- Root files and the excluded directories are out of scope, as in tranches 4 and 8. `recovery/` and `drive/` are items (1)
  and (3) of the proposal.
- The 116 versions superseded within a lineage are listed in `summary_t11b.json` and not classified. A tag at the descendant
  head retains them.
- The workflow agents were told not to write files. The retained workflow script carries its as-run scratch paths.
- **Reproduce** (refs fetched read-only into a main clone):
  1. `python3 -I -B pr_history_census.py MAIN d2b0216e CENSUS main_pull_heads.txt main_branch_heads.txt`, then optionally
     `filter_census.py CENSUS CENSUS_MATH`;
  2. `python3 -I -B t11b_objects.py MAIN CENSUS_MATH CLONES_JSON main_pull_merge_refs.txt CANDIDATE_V2 RETENTION_DIR OUT`, then
     `t10_batches.py OUT/objects.json FACTS`;
  3. `python3 -I -B normalize_verdicts_t10.py VN verdicts/verdicts_verified_t11b_math_a.json verdicts/verdicts_verified_t11b_math_b.json verdicts/verdicts_verified_t11b_eng.json`;
  4. `python3 -I -B fix_line_ranges.py MAIN facts VN VF`;
  5. `python3 -I -B assemble_t11b.py MAIN facts CANDIDATE_V2 OUT2 VF`;
  6. `python3 -I -B reconcile_t11b.py MAIN 5ebe5771 facts records_verified_t11b.json ../qp_c54_t10_unmerged_math_refs_20261102/dormant_index.json main_pull_heads_now.txt main_branch_heads_now.txt OUT3`.

  `CLONES_JSON` is `../qp_c54_catalog_candidate_v2_20261102/clones.json`. `RETENTION_DIR` is this session's retention directory
  (`..`). `CANDIDATE_V2` is catalog candidate v2 as posted (SHA-256 `f689bc94…`). Steps 3–6 were rerun at retention and
  reproduced their outputs byte for byte.
