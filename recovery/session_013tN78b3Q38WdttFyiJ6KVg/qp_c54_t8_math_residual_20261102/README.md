# QP-C54 tranche 8: the Math- directories that no disposition locates or binds (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. Pickup: main#684
comment 6571771733; checkpoint 6573957544. No repository code was run.

## Why this tranche exists

The Math- census behind tranches 1–3 matched file names against patterns. Directories whose files match none of them were never
censused. My own adversarial review of catalog candidate v2 noticed this (finding t4-04), with Math- `repairs/` as the example.

## Scope (`math_dirs.py`)

A directory-level census of Math- at `3b11bff6`, like T4's census of main but with `git ls-tree -z`, so non-ASCII paths are kept.
- It covers `.md`, `.tex`, `.py`, `.json`, `.lean`, `.txt`, `.sage`, `.ipynb`, `.m`, `.jl` and `.csv` files outside the
  engineering directories (`cbz/`, `tests/`, `tools/`, `.github/`, `.claude/`, `docs/site/`, `docs/retro/`, `docs/superpowers/`,
  `governance/`, `ctx/`, `history/`, `interface/`, `records/`, `architecture/`, `data/`, `formal/tests/`) and root files.
- It compares the tree with every row of catalog candidate v2 as built at `76eceeee` (SHA-256 `75ee024f…`): adopted, proposed,
  amendment, federation, accounting and pin rows, and the CLOSED declarations of staging `32ab7d90`.
- A directory is **covered** when one of these holds:
  - a location is the directory or an ancestor, in any form: a `repo@sha:path` string, a GitHub URL or a dict;
  - an adopted `REGISTER.json#claims/...` anchor resolves to it (a landing claim covers its layer-0 source; an
    `existing_lean_package` claim `lean:math-frontier-X` covers `frontiers/X`);
  - a disposed `PROOF.md`, or any disposed file location, lies in it or in an ancestor (a packet).
- A file counts as **bound** when a row binds it, a pin pins it, or a CLOSED declaration names it.

**Result: 45 directories (198 unbound files) with no covering location**, outside the engineering directories. The current tip
`c88c34b2` has exactly the same 45 (`math_dirs_c88_summary.json`).

`t8_objects.py` groups the 45 directories into **37 objects** in 5 batches (`facts/mbatch1.json`–`mbatch5.json`).
- `nodes/`, `results/` and `controls/` subdirectories are merged into their parent.
- Each object lists its files, and the records whose text mentions it, with the fields where the mention occurs (`mentioned_by`).

## Pipeline

1. Workflow `workflow_t8.js`. Every batch gets a draft, a **fidelity** verifier and a **Layer A** verifier: 15 agents. They fixed
   **142 problems** (fidelity 88, Layer A 54).
   - The item schema adds `claim_held_by`: existing records that already hold an item's claim at claim level, in their title,
     `source_claim` or boundary. A mention in quotes, `nonclaims` or prose does not count.
   - The fidelity verifier had to confirm every holder against the catalog.
2. `verdicts_verified_t8.json` is the workflow's return value and equals the journal's final stage. The fidelity passes are
   retained in `verdicts_fidelity_t8.json`. A row's `problems_fixed` lists the Layer A stage only.
3. `assemble_t8.py` checks mechanically:
   - the partition of each object's files;
   - verbatim statements at `3b11bff6`;
   - floating-point classes and REVIEW/COPY targets;
   - every `claim_held_by` id exists;
   - Git bindings.

   An OBLIGATION held only by adopted records becomes a **PINNING_AMENDMENT** to them. One held by a QP-C54 proposal stays a new
   record and gets `claim_overlaps` (SAME_CLAIM_AS_PROPOSAL) for the composer to merge. Rerunning the assembler on the retained
   verdicts reproduces `records_verified_t8.json` byte for byte.
4. Manual checks (`manual_checks_output.txt`):
   - NO_SCRIPTS is consistent;
   - one our-voice hit remains, "that module is closed", which reports the CLOSED index state of the files named;
   - the privacy scan is clean.

## Result (`records_verified_t8.json`)

88 items:

| Class | Count | What it means |
|---|---|---|
| OBLIGATION | **20** | `QP-C54-T8-001` to `-020`, in 14 objects. 18 are new records, and 2 are pinning amendments to adopted Q-1041 and Q-1032 (the exact finite controls of the d5 local-collar proofs). 61 verbatim statements. Floating point: USED 8, NONE 8, NO_SCRIPTS 4. |
| COPY_OR_EXPOSITION_OF | 31 | Chain and frontier indexes, custody READMEs and manifests, each naming what it restates |
| NOT_OBLIGATION | 30 | Harnesses, replay scripts, manifests and plans, each with a cited reason |
| REVIEW_OF | 7 | Review directories that state no result of their own |

The records include:
- the `lifetime_dd_model_d_20261013` certificates v1 and v2 of note DD, with their C10 sliver-gap controls and C1–C9 controls
  (T8-003 to -008, floating point USED);
- the C3-R controls (T8-002);
- the SARD-G A2 v1 parameter interface (T8-009);
- two Grok cycle-4 jet computations (T8-010, -011);
- the exact Laurent reciprocal in `repairs/c026_reciprocal_20260927` (T8-012) and the H5 rim angle-175 falsifier in
  `repairs/h5_rim_contract_20260926` (T8-013);
- a signed-weight clipping counterexample in Math- `formal/` (T8-001);
- review remarks with results of their own (T8-014, -015, -018, -019, -020).

**Overlap with proposals.** Five records name a QP-C54 proposal that already holds their claim, and each carries `claim_overlaps`:
- T8-002 → QP-C54-019;
- T8-003 and T8-006 → QP-C54-021;
- T8-010 → QP-C54-T2-046;
- T8-013 → QP-C54-T2-044.

The verifiers confirmed each holder against the proposal's title or boundary. The composer merges each pair on adoption.

## Limits

- All agents are Anthropic Claude. This is engineering accounting, not a review of the mathematics.
- `main_dirs_check.json` is the same census run on main at `d2b0216e`. It is cited by tranche 4's README: only CSV data and
  manifests are unbound there.
- The workflow's agents wrote helper files in a scratch directory outside the repository, despite the no-write instruction. They
  are not used and not retained.
- **Reproduce:**
  1. Rebuild candidate v2 as of `76eceeee` (`75ee024f…`).
  2. `python3 -I -B math_dirs.py MATH_CLONE d6g8k5htny-coder/Math- 3b11bff6 CAND CLOSED OUT`.
  3. `python3 -I -B t8_objects.py OUT CAND FACTS`.
  4. `python3 -I -B assemble_t8.py MATH_CLONE facts CAND verdicts_verified_t8.json OUT2`.

  `CLOSED` is staging `32ab7d90`'s `cbz/evidence/kernel/all-closed-declarations.json`.
