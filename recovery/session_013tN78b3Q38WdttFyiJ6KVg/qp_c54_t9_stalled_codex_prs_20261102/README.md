# QP-C54 tranche 9: the unmerged mathematics of the stalled Codex Math- pull requests (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. Pickup: main#684
comment 6575251785 (inventory corrected in place at 11:12Z). No repository code was run. Nothing was pushed to any
pull-request branch, and no tag was created.

## Why this tranche exists

Claude S's routing (main#738 comment 6573494134) leaves four stalled Codex Math- stacks to an owner or CoS decision: carry them
(A) or quarantine-archive them (B). Their files are on no Math- branch tip, and no catalog row names any of them. Tranches 1–8
censused tips only, so an archive decision today would drop this work from the accounting. The only precedent for mathematics on
an unmerged pull request is the three main-PR records with durable archive tags (Q-8336, Q-8340, Q-8429).

## Scope

PR heads were fetched read-only into a Math- clone from `refs/pull/N/head`.

- `pr_inventory.py` lists, for each open Math- PR, the mathematical-type files it adds or changes relative to its merge base
  with Math- main `c88c34b2`, and whether each file's bytes are on main. Its output is `stalled.json` (as run); `closed.json` is
  the same inventory for the closed compact_source_cutoff predecessors.
- `t9_objects.py` groups the files of five stacks by directory into **46 objects** in 3 batches (`facts/pbatch1.json` to
  `pbatch3.json`, as the workflow saw them). A file carried by several PRs of a stack is bound at the latest PR that carries it.
  - compact_source_cutoff (batch 1, 7 objects) at #430's head. Every closed predecessor (#414, #415, #424, #425, #427) and #426
    is an ancestor of #430, and #426's files are byte-identical at the same paths in #430.
  - C299 `critical_ridge_20261020` (batch 2, 18 objects): #491 → #496 → #497 → #504 is linear, so #504's head binds it.
  - The determinant stack (#531, #532, #536), the downstream-gate graph #490 and the finite MomentTail CI binding #545 (batch 3,
    21 objects).
- *Supplement.* **#531 is not an ancestor of #532 or #536**, so "latest PR wins" bound 23 of #531's paths at #536's different
  bytes. `t9_supplement.py` writes `facts/pbatch4.json`: #531's own versions of those 23 files, in 5 objects bound at #531's head.
- Earlier versions in ancestor PRs (7 in the C299 stack, 11 at #532) are earlier commits of the same branch as the bound head.
  They are not accounted separately, as history is not on a tip; a tag at the top head retains them.

*Defect found after the run, and how it is handled.* `git diff --raw` prints abbreviated blob ids, so `pr_inventory.py` compared
abbreviated ids with main's full ids and never found a file on main. My first fix used `--full-index`, which does not apply to
`--raw` output, so it reproduced the defect; my first check of it passed only because both lists were abbreviated.
`pr_inventory_v2.py` uses `--no-abbrev`, and `stalled_v2.json` has 40-character ids.
- With full ids, **80 of the 463 object files are byte copies of files on the Math- main tip**: 77 Lean files across the 7
  compact_source_cutoff directories (11 in each: `CapFirstExit.lean` and ten `source_smooth_realization_20261011` modules), 2 in
  C299 `project/` and 1 determinant control.
- The agents had classified every one of the 80 as COPY_OR_EXPOSITION_OF a main file already, so no classification changes.
- `resolve_main_copies.py` (output `facts_resolved/`) moves them to `copies`, naming every main path that holds the same bytes.
  `restrict_verdicts_t9.py` (output `verdicts_resolved_t9.json`) restricts the verified items to the files that remain: 11 items
  lose every file and are dropped, 6 keep the rest, and no OBLIGATION is affected.
- `stack_checks.py` (output `stack_checks.json`) checks mechanically:
  1. the corrected ids are 40 characters, and the as-run list equals the corrected list plus the files on main;
  2. the ancestry of every stack;
  3. #426's 109 off-main files are byte-identical in #430;
  4. `pbatch4.json` equals the set of #531's own versions.

## Pipeline

1. Workflow `workflow_t9.js` (8 agents): batches 1–2 get a draft, a **fidelity** verifier and a **Layer A** verifier, and batch 3
   gets a draft and an adversarial screen. `workflow_t9_batch4.js` (2 agents) gives batch 4 a draft and a screen.
   - The verifiers fixed **99 problems**: fidelity 46, Layer A 37, screens 16 (12 and 4).
   - `verdicts_verified_t9.json` and `verdicts_verified_t9_batch4.json` are the workflows' return values. The fidelity passes are
     retained from the journal in `verdicts_fidelity_t9.json`. A row's `problems_fixed` lists its final stage only.
2. `restrict_verdicts_t9.py` also applies one mechanical normalization. Two COPY_OR_EXPOSITION_OF items listed `claim_held_by`,
   which is reserved for OBLIGATIONs; those ids name the records holding the claims the items expound, so they are moved to
   `refers_to`. The log lists both.
3. `assemble_t9.py` checks mechanically:
   - every PR head is present and every file's blob at its head matches the facts;
   - the items partition each object's remaining files;
   - OBLIGATION statements are verbatim at the head;
   - floating-point classes and REVIEW/COPY targets;
   - every `claim_held_by` id exists.

   It binds every file at its PR head (`pull_request`, `pull_ref`), with file-level locations `d6g8k5htny-coder/Math-@<head>:<path>`,
   and one `mechanical_copies` row per object with copies. Rerunning it on `facts_resolved/` and `verdicts_resolved_t9.json`
   reproduces `records_verified_t9.json` byte for byte.
4. Manual checks (`manual_checks_output.txt`, using `../qp_c54_t4_main_20261102/manual_checks.py`):
   - NO_SCRIPTS is consistent;
   - the privacy hit `EXIT-LEAN` is part of the word CAP-FIRST-EXIT-LEAN;
   - of three our-voice lint hits, one is topological ("closed, discrete and finite") and one says a quarantine is not yet
     closed. The third reports that the 66 determinant companion statements "are CLOSED in the index at that cut". I checked it:
     staging `32ab7d90`'s `all-closed-declarations.json` has exactly 66 CLOSED entries, K-DM-001 to K-DM-066, and all 66
     `git_object` values equal the blobs at #536's head.

## Result (`records_verified_t9.json`)

**27 obligation records**, `QP-C54-T9-001` to `-027`, all NEW_RECORD and all floating point NO_SCRIPTS, with 78 verbatim
statements. Plus 87 other items and 80 mechanical copies.

| PR | OBLIGATION | COPY | REVIEW | NOT_OBLIGATION | main copies |
|---|---|---|---|---|---|
| #430 compact_source_cutoff | 23 | 14 | – | 9 | 77 |
| #504 C299 | 4 | 2 | 3 | 17 | 2 |
| #531 (own versions and custody) | – | 1 | – | 13 | – |
| #536 determinant (with #532) | – | 2 | 2 | 21 | 1 |
| #490 downstream-gate graph | – | 1 | – | 1 | – |
| #545 MomentTail CI binding | – | – | – | 1 | – |

- **#430 (T9-001 to -023).** The baseline package's target theorems `compact_cutoff_of_geometry` and
  `actual_source_compact_cutoff`, stated with empty proof bodies (T9-001), and its 15 supporting modules (normal-form footprint,
  C4 regularity, component histories, binary merge, congruence branch, gluing, H0 elder span, spectral normalization, pushed
  height, quadratic cone, Morse coordinates, torus normal form, integral Taylor matrix, self-adjoint lift). Then each repair
  stage's own bytes: the F2 proof candidate and controls (T9-020, -021), the F4 and F5 controls (T9-019, -018), the line-138
  proof repair (T9-017), and the F6/F7 proper partial controls (T9-022, -023). The sources record these as NOT_RUN, NOT_COMPILED
  or failed hosted compilation, and no indexed receipt binds them.
- **#504 (T9-024 to -027).** The C299 negative partial-gradient ridge module (T9-025), its six superseded candidates (T9-024), a
  C293 finite critical-set module in the C299 project (T9-026), and the direct-ridge-route paper proposal V1 (T9-027).
- **The determinant stack, #490 and #545 hold no obligation of their own.** They are custody, gates, tests and evidence. The
  determinant package's 66 companion statements are already CLOSED (K-DM-001 to -066); its re-check is a REVIEW_OF item. #490's
  GRAPH.json additions register statements that live in other objects.

No record names a holder: none of these claims is held by an existing record.

## For the owner's (A)/(B) decision

These records make either choice safe for the accounting; they decide nothing.
- (A) carry: on integration each location becomes the same path on Math- main, and the records stay as they are.
- (B) archive: as for Q-8336, Q-8340 and Q-8429, the composer needs a durable annotated tag before any branch is deleted. Six
  heads cover everything here: #430 `1a7fc4be`, #504 `d65e4425`, #536 `0dd9f215`, **#531 `c4a89bb9`** (not an ancestor of
  #536), #490 `1016973a` and #545 `e1cf617b`. Each tag at a top head also retains every ancestor version.

## Limits

- All agents are Anthropic Claude; the authors are OpenAI Codex. This is engineering accounting, not a review of the mathematics.
- #482 and #487 are left out: their tip-syncs are read back (main#738 comment 6575161860) and they are moving toward
  integration. Other PR heads and branch tips are tranche 10 (main#684 comment 6575726856).
- The workflow agents were told not to write files; one wrote two scratch files (`b1_base.txt`, `b1_main.txt`) beside the
  inventory, outside the repository. They are not used and not retained. The retained workflow scripts carry their as-run
  scratch paths.
- **Reproduce** (PR heads fetched as `refs/remotes/pr/N`):
  1. `python3 -I -B pr_inventory_v2.py MATH_CLONE c88c34b2 stalled_v2.json 426 430 482 487 490 491 496 497 504 531 532 536 545`
  2. `python3 -I -B t9_objects.py stalled.json closed.json FACTS`, then `python3 -I -B t9_supplement.py stalled.json FACTS`
  3. `python3 -I -B stack_checks.py MATH_CLONE stalled.json stalled_v2.json FACTS OUT`
  4. `python3 -I -B resolve_main_copies.py MATH_CLONE c88c34b2 FACTS FACTS2`
  5. `python3 -I -B restrict_verdicts_t9.py FACTS2 VERDICTS2 verdicts_verified_t9.json verdicts_verified_t9_batch4.json`
  6. `python3 -I -B assemble_t9.py MATH_CLONE FACTS2 CANDIDATE_V2 OUT2 VERDICTS2`

  `CANDIDATE_V2` is catalog candidate v2 as posted (SHA-256 `f689bc94…`).
