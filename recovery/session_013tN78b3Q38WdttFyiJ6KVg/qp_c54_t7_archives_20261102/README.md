# QP-C54 tranche 7: the members of committed archives at the federation tips (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. Pickup: main#684
comment 6570778772. Archives are read from Git into memory only; nothing is extracted to disk and nothing is run.

## Scope

Tranches 1–6 classified tree files. They did not open committed archives.

`archive_inventory.py` (output `archive_inventory.json`) finds every `.zip`, `.tgz`, `.tar.gz`, `.tar` and `.gz` file at the
tips: main `d2b0216e` (8 zips and 1 `.json.gz`) and trial `3738c859` (1 `.tgz`). Math- `c88c34b2` and the other federation
repositories hold none.
- Each member gets its bytes, SHA-256 and Git blob id, computed as `git hash-object` would. Nested zips are opened one level
  down.
- **1,657 top-level members, plus 48 inside 2 nested zips.**
- A member is marked `copy_of` when its blob equals a file in main's or Math-'s tree. The disposed-location lookup uses
  catalog candidate v2 as built at `76eceeee` (SHA-256 `75ee024f…`).

`t7_objects.py` turns the archives into 11 objects in 5 batches (`facts/abatch1.json`–`abatch5.json`).
- Each object lists its members to classify: the first occurrence of each distinct blob, **696** in all.
- Its other entries are resolved mechanically (**1,009**): byte copies of a main or Math- tree file, copies of a member listed in
  another object, or the 2 nested archives.

`show_member.py` prints one member from memory. It was the agents' only way to read members.

## Pipeline

1. Workflow `workflow_t7.js`:
   - batch 1, the formal-gate receipts, gets a draft, a fidelity pass and a Layer A pass;
   - batches 2–5 get a draft and an adversarial screen that searches every member for mathematical claims.
   - 11 agents fixed **34 problems** (fidelity 5, Layer A 10, screens 19).
2. `verdicts_verified_t7.json` is the workflow's return value. It equals the final-stage results in the journal.
3. `assemble_t7.py` re-reads every archive and checks mechanically:
   - the archive's blob and SHA-256 equal the facts;
   - every member's bytes, SHA-256 and blob, recomputed from the archive, equal the facts;
   - the items partition each object's members;
   - OBLIGATION statements are verbatim at their lines of the member;
   - REVIEW and COPY items name their targets.

   Every row binds the archive file and lists each member it covers, with that member's bytes, SHA-256 and blob. Rerunning the
   assembler on the retained verdicts reproduces `records_verified_t7.json` byte for byte.
4. Manual checks (`manual_checks_output.txt`, using `../qp_c54_t4_main_20261102/manual_checks.py`):
   - no our-voice overclaim;
   - no e-mail address, token, signed link or login code in the output. The two device-code-shaped strings are the words
     `HARD-GATE` and `LOGQ-TAIL`.
   - Members that contain a login code or an e-mail address (trial `GH_DEVICE_LOGIN.md`, `LAND.md` and `OWNER_ONE_LINERS.md`,
     and a commit-identity capture in the lifecycle evidence) are named and never quoted.

## Result (`records_verified_t7.json`)

**0 obligations.** 53 items and 1,009 mechanical copies:

| Class | Items | Examples |
|---|---|---|
| REVIEW_OF | 3 | The `#print axioms` / `#check` audit scripts over the 31 Side24 targets (Main188) and the 13 ResearchFormalCoreR1 targets (Math96); the mechanical OPEN-status inventories in trial's research-stack audits |
| COPY_OR_EXPOSITION_OF | 8 | **Lean negative controls:** falsified copies of CLOSED `formal/UniversalLaw/Side24` files, each naming the changed declaration and the receipt line where Lean rejects it. For example, `wrong_image_constant.lean` changes `image_constant_value` from 21175738586478 to 21175738586479. Also the site snapshots' mathematical node pages, which render graph records, and the frozen atlas observation payload. |
| NOT_OBLIGATION | 42 | CI and build logs, axiom-gate probe fixtures (custom axiom, `native_decide`, `sorry`), gate mutants and their logs, lifecycle raw evidence, site index pages, trial's landing runbooks, restore plans, batch briefs and patch stack |

The trial `.tgz` is one archive split into two objects (`portable/` and `scripts/`). Each object has one row in
`mechanical_copies`, listing its copied members with `copy_of`.

## Limits

- All agents are Anthropic Claude. This is engineering accounting, not a review of the mathematics.
- Archives in Git history but not at a tip are out of scope. Claude S's R2 lane (main#684 comment 6568836103) searches code
  members across all history for implementations.
- `ev-source-history-bundle` is a git bundle whose pack contents were not decoded. It is classified from its receipts.
- **Reproduce:**
  1. `python3 -I -B archive_inventory.py CLONES_JSON CANDIDATE_V2 OUT`;
  2. `python3 -I -B t7_objects.py OUT FACTS_DIR`;
  3. `python3 -I -B assemble_t7.py CLONES_JSON facts verdicts_verified_t7.json OUT2`.

  `CLONES_JSON` is `../qp_c54_catalog_candidate_v2_20261102/clones.json`.
