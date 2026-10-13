# Recovery reading copy: session `session_01Cz7WZybv8znP64SpPj6sWY` scratch work (2026-10-13)

Published under `recovery/` (a recovery input with explicit retention, see below; zero scientific authority) in answer to the
owner's recovery request on main#307 ([6136156988](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136156988),
expanded in [6136210973](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136210973); implementation decision
[6136231457](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136231457)): preserve the still-local project
material of this Anthropic Claude session so that no derivation, probe or tool has to be repeated after an interruption.

**Everything here is scratch: candidates, probes, mutants, logs, drafts, failed attempts and working notes.** Nothing is a
receipt, a review verdict or evidence about any mathematical claim; no label of any claim, Prop or packet is changed by
this copy; organizational-independence credit 0; scientific effect NONE. Where a piece of work landed, the landed object
is the record and this copy is its provenance: the Lean statement packet and proof packet on Math- main (`8acc4856`,
Math-#441 `aef861eb`); the main#358 renderer and round-5/6 binders in branch `claude/handoff-lean-register-tools-20261013`
at `b88efe53`; the museum review [151/6135519604](https://github.com/d6g8k5htny-coder/main/pull/151#issuecomment-6135519604);
the frontier section in draft Math-#445 (`9fb0a4a0`).

## Layout

`scratch/<original relative path>` keeps each file where it sat below the session's scratch root, so the notes, logs and
scripts reference each other as they did. `MANIFEST.json` lists every published file with `area`, `kind`, byte count and
SHA-256 of the published copy and of the exact as-run bytes (`as_run_*`); `scrubbed: true` means the only change is the
replacement of the private scratch root by `<SCRATCH>`, the session UUID by `<SESSION-UUID>` or the owner's relay
address by `<OWNER-RELAY-EMAIL>`. `EXCLUDED.json` names what was not copied and why.

| area | files | bytes | scrubbed | contents |
|---|---|---|---|---|
| `renderer_versions_1_9` | 147 | 7,422,313 | 33 | `render_spec.py` … `render_spec6.py` (lvs/spec_doc, with coverage/roadmap/timing notes) and `render_spec7/8/9.py` with their round logs: every renderer of `docs/LEAN_VERIFICATION_SPECIFICATION.md` before `render_spec10.py` (which is in the handoff branch). Older versions and fix-round candidates; superseded by render_spec10 at main#358 head `b4615124`. |
| `binders_rounds_1_9` | 16 | 318,358 | 10 | Register/document binders of the early rounds (`register_update2/5/6`) with their summaries; superseded by later rounds, kept for provenance. |
| `binders_rounds_10_14_with_logs` | 350 | 18,369,854 | 125 | Rounds 10–14 (proof-packet rounds 5–6 pending and binding tiers; the spec renderer round): scripts (as-run versions also published scrubbed in `b88efe53`), `.v1/.v2` candidates, dry runs, verify dirs, check/test logs, committed-doc copies used as render bases. Landed results: Math-#439, Math-#441. |
| `round4_triage_probes` | 268 | 2,229,042 | 44 | The round-4 triage of the 82 Props without a theorem after round 3: `triage_table.json` and the per-Prop probe `.lean` files with their elaboration outputs (`.out`). Basis of the frontier section in Math-#445. |
| `proof_packet_rounds_5_6_verification` | 342 | 12,943,211 | 12 | Working files of proof-packet rounds 5–6: commit messages, PR bodies, handoff drafts, run scripts, local receipt JSON (`r6commit`), verifier scripts and outputs (`r6*`, `verify_r5*`, `verif_r5d`, `vr9*`, `verify13d`, `verify_reg11`), Side24/P15 scratch (`side24`, `p15`, `dv2`). Local records only; hosted receipts are in the register. |
| `museum_review_probes` | 1152 | 14,757,238 | 107 | The main#151 museum review: five lens probe directories and six second-reader directories under `museum/probes/` (probe scripts, mutant generators, fixture pin lists and base64 pinned inputs, outputs, `ALL_OUTPUT.txt`), `museum/synth/` (35-finding synthesis, isolated re-run outputs, the posted review and payloads), `a2/` (the PR-151 body with the 19 identities). The extracted `cd32303` / `main_*` trees are excluded (reproducible by `git show`). |
| `torus_companion_c254_c26x` | 2024 | 23,905,354 | 216 | Torus companion work on Math-#428: C254 compile-fix briefs and fixtures, C262 gate patches/controls/replays (scratch copies of trees excluded), C263/C264 compile and repair notes, C264 controls and their integration receipts, C26x fixes and registration replay, #419/#428 push records, verification scripts and logs. |
| `lvs_misc_notes_and_fixes` | 182 | 2,372,381 | 21 | PR-body binders for main#358 / Math-#434 / Math-#428, crawl worklists, elaboration probes, lifetime/field work notes, fixer probes (field-lifetime, infra, RN), governance notes, guide-row and infra fixtures. |
| `frontier_round_and_handoff_tooling` | 52 | 1,785,354 | 1 | Tonight: the frontier generator `lvs/r7/frontier15.py`, coverage JSON, PR body and payloads (Math-#445); the handoff build (`handoff/`: publication builder output, manifests, demo renders, check logs, posted comment drafts). |
| `lvs_top_level_notes` | 57 | 1,568,511 | 14 | Loose notes and records directly under `lvs/`: design notes, PR/commit message drafts, claim/release texts, census scripts and outputs, CI job log, inventories. |

Totals: 4,590 files, 85,671,616 bytes.

## Not copied (185 entries, see `EXCLUDED.json`)

- repository/tree copy: 103
- excluded directory: 47
- build artifact: 35

Repository and tree copies (clones, worktrees, extracted trees, mutant copies of whole packets) are reproducible from the
commits their neighbouring notes and logs name (`git show <commit>:<path>`); the Lean `.lake` caches and `.olean` build
products are reproducible from the pinned toolchain (Lean 4.34.1, Mathlib `d13f23b7`). No credential-shaped bytes were
found by the intake guard's own patterns, and no file was excluded on that ground.

## How to read a round

Each `register_update<n>/` holds the binder scripts (`update_register<n>.py`, `render_docs<n>.py`, `edit_proof_docs<n>.py`),
their `update_summary<n>.json`, the check and test logs (`*_check.log`, `*_tests.log`, `*.exit`) and, where a round had
fix rounds, the `.v1/.v2` candidates. The round-4 triage probes elaborate against the statement packet at the round-4 head
(each probe file names it). The museum probe directories each have a `run_all.sh` or equivalent and an aggregate output.

## Retention (explicit, per main#307 6136476552)

This branch and this commit are a **recovery input**, not a disposable scratch copy: retain the payload branch and
commit while any required work relies on them, until a verified durable replacement contains the actual bytes and
the index packet (`incoming/claude-01cz-recovery-index-20261013`, draft main#386) is updated to point at it. An
index alone is not a replacement backup; a locator or report is not the payload. No deletion is requested or
authorised by this README. Incremental checkpoints, if any, are further commits on this branch with the manifest
regenerated, never rewrites of history. The directory was moved here from `sandbox/recovery_01cz_20261013/`
(commit `037436b4`, whose README carried the sandbox wording "wipeable, never to be linked from a permanent record");
that wording is withdrawn for this material and this move is the reconciliation. The payload bytes and `MANIFEST.json`
are unchanged by the move; only this README differs from `037436b4`.

## Checkpoint 2 (2026-10-13, after 02:00Z)

Incremental commit, no rewrite: 39 files added and 1 changed since `ee7491ba` (none removed), bringing the
copy to 4,629 files, 86,169,223 bytes. New area `recovery_tooling` (`scratch/recovery/`: the recovery build script, its
profile, the posted delivery texts and the index-packet drafts); `scratch/lvs/r7/` gains the frontier round's design note,
notice, checkpoint and PR texts. `MANIFEST.json` and `EXCLUDED.json` are regenerated over the whole copy (same scheme,
published and as-run digests per file; the scrub now also replaces the owner's relay local part by a placeholder). Changed
files are files this session kept editing after the first copy; their earlier bytes remain in `037436b4` / `ee7491ba`.
