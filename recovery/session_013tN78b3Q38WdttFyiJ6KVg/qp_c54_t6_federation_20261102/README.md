# QP-C54 tranche 6: mathematics in the eight federation repositories outside main and Math- (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. Pickup: main#684
comment 6567295735. No repository code was run.

## Scope

The CBZ catalog's mathematical scope is main and Math-. Its one federation row, adopted Q-9001, is an engineering regression region in trial (`tests/test_intent.py`, `README.md`, `scripts/` and `portable/` at `3738c859`). Q-9001 does not quarantine mathematics, but T6-007 and T6-008 sit inside its location, and catalog candidate v2 records this (erratum `E-R-OVL-T6-007`/`-008`). `fed_objects.py` (output `fed_objects.json`) lists every Markdown, TeX,
Python, JSON, Lean or text file in eight repositories, each at its remote tip:

- meta-framework `0f5894f8`, google-drive `ad08584b`, governance- `1b62ca2f`;
- Universal-Law-Workspace `202f9f5c`, query- `fb4c457b`, trial `3738c859`, sandbox `dce04a41`, d6g8k5htny-coder `d59de18e`.

Files are grouped into **92 directory objects**. Each file is marked `copy_of` when its Git blob equals a file at a
disposed main or Math- location. This replaces the filename-pattern census, which had missed, for example, the 33
`governance-/amendments/` reviews and most of `meta-framework/reviews/`.

## Pipeline

1. **Batches.**
   - 5 math-likely batches (`fbatch1.json`–`fbatch5.json`) get a draft, a fidelity pass and a Layer A pass.
   - 3 engineering-likely batches (`fbatch6.json`–`fbatch8.json`) get a draft and one combined adversarial screen that
     searches every object for mathematical claims.
   - The workflow is `workflow_t6.js`.
2. Each object is split into items. Each item is classified OBLIGATION, REVIEW_OF, COPY_OR_EXPOSITION_OF or
   NOT_OBLIGATION, with file:line evidence. **53 problems** were fixed (`verdicts_verified_t6.json`).
3. `assemble_t6.py` checks mechanically:
   - every object appears once, and its items partition its full file list;
   - quoted statements are verbatim at the object's own ref;
   - floating-point fields are classified, and REVIEW_OF / COPY items name their targets;
   - each `refers_to` target is resolved against main, Math- and the catalog;
   - the bindings come from Git;
   - byte-identical obligations in different repositories are merged into one record.

   Rerunning it on the retained verdicts reproduces `records_verified_t6.json` byte for byte.
4. Manual checks:
   - NO_SCRIPTS is consistent with each item's files;
   - an overclaim lint found 3 hits, all negations such as "cannot establish" or "certified false";
   - no e-mail address, token or device code appears. `trial`'s `GH_DEVICE_LOGIN.md` is only named, never quoted.

## Result (`records_verified_t6.json`)

175 items:

- 9 OBLIGATION items, which become 8 records `QP-C54-T6-001` to `-008` once the two identical palette
  `PROOF.md` items are merged;
- 36 REVIEW_OF;
- 26 COPY_OR_EXPOSITION_OF;
- 104 NOT_OBLIGATION.

| Record | Repository and object | What it states |
|---|---|---|
| T6-001 | google-drive `replicas/demanded-palette-consecutive-v1` **and** meta-framework `replicas/consecutive-palette-v1` (the two `PROOF.md` files are byte-identical, blob `1aa9db5b`. The `SOURCE.json` files differ, `e566be67` and `529c3281`, and each is its own NOT_OBLIGATION provenance item) | Exact demanded-palette optimum for consecutive capacity supports (P15-CONSECUTIVE-CAPACITY-PALETTES-20260924-v1); in neither main nor Math- |
| T6-002 | governance- `amendments/20260925-math18-d6-analytic-review.md` | Counterexample and erratum to the full-price (F10) endpoint: p > 1/2 must be narrowed to 1/2 < p < 1 |
| T6-003 | governance- `amendments/20260925-math19-finite-r-review.md` | The reviewer's sharpness remark on the S1 O(r) rate |
| T6-004 | governance- `amendments/20260925-mesoscopic-d2-chart-handoff.md` and `check_mesoscopic_d2_taylor.py` | d = 2 contact-chart Taylor rows, the J₀ divided-difference block and the r³ height row |
| T6-005 | sandbox `experiments/cone_v1` | Closed form 29/6 − √6 for a truncated Gaussian cone moment; the only evidence is a Monte Carlo probe. *Correction (`E-R-OVL-T6-005`, `E-R-T6-005-REASON`):* the same identity, with the same setup and the same remark on 29/6, is in Math-@760340e9 `coefficients/side24_v1/PROOF.md:59-75`, which adopted Q-10041 and Q-1001 hold. The record's sentence that no other object is its source is corrected, and the composer may reclassify the probe as a Monte Carlo check of that identity. |
| T6-006 | trial `experiments/p15_assumption_probe_20260925` | Hypothesis minimality for the P15 F5 curvature sign and F10 recurrence sign |
| T6-007 | trial `portable/DROPBOX_GAP_HUNT_20260927/NEW_RELEVANT/LPW_Return_05…` | LPW_CONSTANT decimal erratum and corrected fourth moment. **7 of its 8 files are byte-identical to main `audits/dropbox_import/2026-09-27/…/LPW_Return_05…`**. Reconciled in tranche 4 (`../qp_c54_t4_main_20261102/reconciliation_t6.json`): main's QP-C54-T4-004 is primary for the shared bytes, and T6-007 stays only for the trial-only `04_RETURN_TO_KIMI.md`. |
| T6-008 | trial `portable/review_packets/2026-09-27_pr160_lean_replay.md` | mpmath non-vacuity of the Γ(1/6) hypothesis window for PR #160 |

Floating point: NO_SCRIPTS 4, NONE 3, USED 1 (T6-005, the Monte Carlo probe).

**Referents.** Every COPY or REVIEW item names its target. The assembler also reports 207 referents that do not
resolve to a main or Math- path or record. Most are catalog keys or paths inside the item's own repository, or
PR and comment references. They are reported by design, not failed. 14 named targets exist in main or Math- but
have no disposition in the catalog as it stands:

- two main review directories that tranche 4 covers;
- `docs/retro` traces and `docs/RESEARCH_INDEX.md`, which are engineering or index files;
- files inside packets already proposed in tranche 5 or bound by Q-3010;
- `coefficients/side24_v1/coefficient.py`, beside Q-1001's `PROOF.md`.

## Limits

- All agents are Anthropic Claude. This is engineering accounting, not a review of the mathematics.
- trial's `portable` object holds about 5,970 files, mostly JSON batch briefs. Its classification rests on the
  `ls-tree` listing and on the example files the screen cites, such as `BATCH387_TIP_SYNC.json` and
  `BATCH357_RESEARCH_STACK_AUDIT.json`, not on reading every file.
- **Reproduce:** `python3 -I -B assemble_t6.py MAIN_CLONE MATH_CLONE . fed_objects.json CANDIDATE_CATALOG CLOSED_JSON verdicts_verified_t6.json OUT.json`.
  `CANDIDATE_CATALOG` is catalog candidate v1, built by `../qp_c54_catalog_candidate_20261102/build_candidate.py`
  (SHA-256 `0f956ace…`). `CLOSED_JSON` is staging `32ab7d90`'s `cbz/evidence/kernel/all-closed-declarations.json`
  (SHA-256 `c25b7e46…`). Each federation repository is read from its own clone, as named in `fed_objects.json`.
- Corrections accepted from the different-family check (main#738, comments 6568869813 and 6568931499): the item
  total (175), the `SOURCE.json` wording, the 207 unresolved referents, and this recipe. The `**` markup dropped
  inside quoted classification reasons is left as it is, because those reasons are not checked statements.
- One agent wrote two scratch files (`b8_items*.json`) outside this directory despite the no-write instruction. They are
  not used and not retained.
