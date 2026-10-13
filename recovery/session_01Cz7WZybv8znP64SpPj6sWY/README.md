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

## Checkpoint 3 (2026-10-13, about 02:30Z)

Incremental commit, no rewrite: 8 files added and 2 changed since `097d9450` (none removed); now 4,637 files,
86,192,343 bytes. Adds the round-7 feasibility probe `scratch/lvs/r7/probes/Side24B_probe.lean` (4,724 B, sha256
`f630a7d1…`, no theorem, no `sorry`) with its elaboration output (exit 0 at Lean 4.34.1 / Mathlib `d13f23b7`), the
round-7 first-slice proposal and its addendum as posted on main#307, and the recovery texts of checkpoint 2.

## Checkpoint 4 (2026-10-13, about 02:50Z): the round-7 design-review packet, under HOLD

Incremental commit, no rewrite: 15 files added and 2 changed (`MANIFEST.json`, `EXCLUDED.json`) since `e8349bed`
(none removed); now 4,652 files, 86,391,390 bytes. The coordinator's correction on main#307 (6136949323) put the
round-7 first slice on HOLD before anything was pushed to Math-: no packet, register, gate or docs change exists
anywhere. What is preserved instead is the exact object offered to the routed nonauthor design read,
`scratch/lvs/r7/hold/`: `DESIGN_REVIEW_PACKET.md` (the declaration-by-declaration mapping of the draft to the pinned
sources and the caveats found: `sphereMeasure` still free beyond support and area, frame invariance a consistency
burden not a field, nondegeneracy carried as two positivity fields, Mathlib's `multivariateGaussian` degenerating to
a Dirac mass off positive semidefinite covariances); `Side24_append.lean` (the draft block: 9 auxiliary definitions,
the successor structure `PeriodicCoefficientInterface2`, the successor Props `CoefficientEnclosure2`,
`ReferenceComparison2`, four-label docstrings with verbatim anchors); `Side24_draft.lean` (the full draft module,
prefix byte-identical to Side24.lean at Math- `8acc4856`; `lake env lean` at Lean 4.34.1 / Mathlib `d13f23b7` exit 0,
no output, `elab_draft.log`); `scan_draft.txt` (the statement gate's own `scan_module` on the draft: 78 declarations =
the committed 66 + 12 new, every new registered-kind declaration with the four labels and a verbatim anchor);
`register_rows_proposed.json` (the three statement rows and the proposed `not_established`/`next_step` sentences,
NOT applied); the posted hold acknowledgement. Also added: the round-7 pickup and scope-correction texts and payloads
as posted, the statement-gate test runner copy `stmt_test_check.py`, and the checkpoint-3 texts. Everything stays a
draft: `specified` throughout, no theorem, no inhabitant, no label move, no acceptance; the review and the routing
are the coordinator's.

## Checkpoint 5 (2026-10-13, about 03:25Z): the corrected round-7 design packet

Incremental commit, no rewrite: 3 files added and 7 changed since `52eb485f` (none removed); now 4,655 files,
86,407,828 bytes. The coordinator's bounded design read of checkpoint 4 (main#307 6137276515) found the slice eligible
in principle after five corrections and a stacked placement on Math-#445; the corrections are applied to the draft in
`scratch/lvs/r7/hold/` (Dirac-fallback wording; `ReferenceComparison2` anchored to the quantitative display (4) line
with its tag; the angular-measure remark restricted to the inequalities; the pre-existing inventory stated as 26 Props,
2 structures, 38 auxiliaries; the frontier-row correction noted for the stacked docs), and the corrected draft module
re-elaborates at the pin with exit 0 (`elab_draft.log`) and passes the gate's `scan_module` again (`scan_draft.txt`);
`DESIGN_REVIEW_PACKET.md` gains a "Corrections applied" section. `confirm_placement.md` is the pre-post draft of the
author's confirmation (its two placeholders, this checkpoint's commit and the elaboration line, were filled at posting
time); the checkpoint-4 texts are added. Still a draft under hold: no Math- ref, packet, register, gate or docs change
exists anywhere; activation is the coordinator's.

## Checkpoint 6 (2026-10-13, about 03:40Z): round 7 published as draft Math-#447

Incremental commit, no rewrite: 15 files added and 7 changed since `78ba7ac6` (none removed); now 4,670 files,
86,718,859 bytes. The coordinator activated the four-file stacked draft (main#307 6137545127) with one more wording
correction, applied before publication (`scratch/lvs/r7/hold/` refreshed: Dirac/degenerate wording exact, the Schur
identification's invertible-pin-block hypothesis named at `pinCov_posDef`). Published: Math- branch
`claude/lean-statements-round7-side24b-20261013`, head `077a168ba9fe3a0888c2d1988f85511c30ca9484` on #445's `9fb0a4a0`,
draft PR Math-#447 (base `claude/lean-statements-frontier-20261013`). Preserved here: the binder `scratch/lvs/r7/bind_round7.py`
with its summary, the exact commit as a patch (`round7_077a168b_on_9fb0a4a0.patch`), the gate logs (source mode on the
dirty and on the committed CLEAN worktree, exit 0), the test logs (97 OK in both modes), the PR body as posted and the
create payload, the activation report posted on main#307. The hosted runs on `077a168b` were in progress at this
checkpoint; no hosted outcome is recorded here. Still: every new declaration `specified`; no theorem; no label move.

## Checkpoint 7 (2026-10-13, about 03:50Z): the hosted runs of Math-#447 bound

Incremental commit, no rewrite, after checkpoint 6 (`1259ee34`): adds the read-only run records for Math-#447 head
`077a168b` (`scratch/lvs/r7/runs447.json`: runs 38342494931 statements, 38342494925 proofs, 38342495166 downstream
replay, all pull_request attempt 1, all `success`; artifact names, ids, sizes and GitHub digests; bytes not downloaded),
the collector and text generator, the binding sentence as placed in the PR body (`pr_body_round7_bound.md`) and the
binding post on main#307, and the fetched PR records. The actual tested merge `9ad2639c` (= `refs/pull/447/merge`,
parents `9fb0a4a0` and `077a168b`) has exactly the head's tree `7534b393`, verified locally. A green run is a record of a
computation about the formal objects; nothing here is a review, an alignment, an acceptance or a label move.

## Checkpoint 8 (2026-10-13, about 03:55Z): the Lifetime successor-interface feasibility probe

Incremental commit, no rewrite, after checkpoint 7 (`b3f24cf0`): adds `scratch/lvs/r7/probes/Lifetime2_probe.lean` with
its elaboration output (exit 0 at Lean 4.34.1 / Mathlib `d13f23b7`, `#check`s only, no theorem, no `sorry`): a scratch
probe, not a packet file, of the design note's §2 successor `PairLawInterface2` for the three interface-blocked Lifetime
Props (`ParentUniformSelectionA`, `PlanarElderLowerE2`, `PlanarTwoSidedRateE4`): the `2(d+1)` pin observations, their
jet covariance matrix from `periodicCovariance`, the regression mean and Schur-complement covariance, the typed
determinant weight, and a structure extending `PairLawInterface` whose fields give the regression law `Q` on `G`'s
sample space with its finite-dimensional Gaussian regression characterisation, positive definiteness of the observation
covariance, the pins almost surely, positivity of the normalizer, and `p` pinned to the `Q^W` probability of `I`'s elder
event, with `Q^W` defined from `Q` and the weight. Input for a later proposal only; nothing is proposed, claimed or
written to any packet by this checkpoint.

## Checkpoint 9 (2026-10-13, about 04:06Z): the #447 amendment, the Lifetime proposal and the RN scoping note

Incremental commit, no rewrite, after checkpoint 8 (`0ed6ced1`): adds the amendment of Math-#447 after dot's final source
review (AMEND_SCOPED, two currentness-text corrections): the commit as a patch (`round7_amend_215726a2.patch`, new head
`215726a2` on `077a168b`), the gate logs (dirty and committed CLEAN worktree, exit 0), test logs (97 OK both modes), the
elaboration log, the reply on #447 and the note on main#307, the PR body as updated for the new head, and the patched
wording of the earlier binding post; the Lifetime successor proposal as posted (main#307 6137849133) and the RN successor
scoping note (`scratch/lvs/r7/rn_successor_scoping.md`, read-only design input, not a proposal). Hosted runs on
`215726a2` were in progress at this checkpoint; no hosted outcome for that head is recorded here.

## Checkpoint 10 (2026-10-13, about 04:13Z): the amended head's hosted runs bound

Incremental commit, no rewrite, after checkpoint 9 (`791e5c3e`): adds the read-only run records for Math-#447 head
`215726a2` (`scratch/lvs/r7/runs447_215726a2.json`: runs 38344645102 statements, 38344645161 proofs, 38344645372
downstream replay, all pull_request attempt 1, all `success`; artifact names, ids, sizes and GitHub digests; bytes not
downloaded), the generalized collector and text generator, the binding as placed in the PR body and posted on #447 and
main#307. The tested merge `15e8fa2f` (= `refs/pull/447/merge`, parents `9fb0a4a0` and `215726a2`) has exactly the head's
tree `f98bbf51`, verified locally. A green run is a record of a computation about the formal objects; nothing here is a
review, an alignment, an acceptance or a label move.

## Checkpoint 11 (2026-10-13, about 04:17Z): the RN successor-interface feasibility probe

Incremental commit, no rewrite, after checkpoint 10 (`06b5784f`): adds `scratch/lvs/r7/probes/RN2_probe.lean` with its
elaboration output (exit 0 at Lean 4.34.1 / Mathlib `d13f23b7`, `#check`s only, no theorem, no `sorry`): a scratch probe,
not a packet file, of the design note's §1 successor `PinnedLawInterface2` for the 19 interface-blocked RN Props, fields
1–4 and 6 (the kernel `K_L` defined locally as Lifetime.lean has it, the `2(d+1)` pin observations and their jet
covariances, regression mean and Schur-complement covariance, positive definiteness of the observation covariance, the
Gaussian regression characterisation of the given law `Q`, the given weight `W` tied to the typed determinant weight, the
normalizer floor of [LP] (5.5) on compact windows as an assumption, a.s. finiteness of the critical set and measurability
of the counts), with two illustrative successor Props by structure projection. Field 5 (Kac-Rice identities and the
defined contact kernel) is left to a second probe. Input for a later proposal only; nothing is proposed, claimed or
written to any packet by this checkpoint.

## Checkpoint 12 (2026-10-13, about 04:27Z): the RN Kac-Rice contact-kernel probe

Incremental commit, no rewrite, after checkpoint 11 (`ba43a043`): adds `scratch/lvs/r7/probes/RN2b_probe.lean` with its
elaboration output (exit 0 at Lean 4.34.1 / Mathlib `d13f23b7`, `#check`s only, no theorem, no `sorry`): the second part of
the RN successor design, defining the Kac-Rice contact kernel from the regression law's Gaussian parameters (two Schur
steps, a Gaussian density on a finite index type, a Sylvester-inertia predicate for matrices, the typed weight on Hessian
coordinates, `multivariateGaussian`) and stating the Kac-Rice identities for the window and all-height mean counts as
fields; the RN scoping note gains an addendum listing the reader items. Design input only; nothing proposed, claimed or
written to any packet; the posted texts of the PASS_SCOPED round and the #445 reader aid are added as well.

## Checkpoint 13 (2026-10-13, about 04:53Z): #445 amended, #447 reconciled, both bound

Incremental commit, no rewrite, after checkpoint 12 (`5302a6de`): adds the #445 amendment work (Codex's classification
review 6138311227, amendments 445-A to 445-D: `scratch/lvs/r7/amend445.py`, the commit `0c506fea` as a patch, gate logs),
the #447 reconciliation (`bind_round7b.py`, the binder variant that reproduces the amended #445 docs byte-for-byte from the
committed register and then applies the round-7 transformation; merge head `06f03d06`, gate and test logs), the read-only
run records of both new heads (`runs447_0c506fea.json`, `runs447_06f03d06.json`; six green pull_request runs, tested
merges `640093a3` and `3582fc6a` with the heads' trees), the parametrized binding generator `bind_text_pr.py`, the PR
bodies as bound, and every posted text (ACK, replies, bindings, board notes). Every declaration stays `specified`; no
label moves; nothing marked ready or merged.

## Checkpoint 14 (2026-10-13, about 05:00Z): the Lifetime slice design-review packet (draft, not activated)

Incremental commit, no rewrite, after checkpoint 13 (`8890ee69`): adds `scratch/lvs/r7/hold_lifetime/`, the exact object a
design read of the proposed Lifetime successor slice (main#307 6137849133) would take: the draft block to be appended to
`Specifications/Lifetime.lean` (13 auxiliaries, `PairLawInterface2`, three successor Props, four-label docstrings with
verbatim anchors), the full draft module (prefix byte-identical to the committed module; `lake env lean` at the pin exit 0,
no output), the statement gate's `scan_module` output (55 = 38 + 17, labels and anchors verified), the proposed register
rows (NOT applied) and `DESIGN_REVIEW_PACKET_LIFETIME.md` (declaration-by-declaration source mapping and seven caveats).
Nothing is written to any packet; activation is the coordinator's. Also adds the closing-checkpoint draft.

## Checkpoint 15 (2026-10-13, about 05:04Z): closing checkpoint of the evening

Incremental commit, no rewrite, after checkpoint 14 (`3cfb5e6e`): adds the texts posted at the close of this lane's evening
work: the acknowledgement of Codex's PASS_SCOPED_SOURCE readback of Math-#445 at `0c506fea` (the optional B-row wording rides
the next documents push, none made for it alone), the pointer to the Lifetime slice's design-review packet, and the closing
checkpoint on main#307 (6138625278). State at close: Math-#445 `0c506fea` and Math-#447 `06f03d06` green, bound and
review-complete on their current heads, waiting on an integrator (#445 first, with the bounded reconciliation read of
#447's regenerated documents); Lifetime slice proposed with its packet, awaiting routing; RN probes preserved as design
input; main#386 frozen under GROK-H03's read. Nothing merged or marked ready by this lane; every new declaration `specified`.

## Checkpoint 16 (2026-10-13, about 05:32Z): the round-7 Side24 slice landed

Incremental commit, no rewrite, after checkpoint 15 (`b931f777`): adds the run collection and the binding texts for the
integrator's carried head `f6ba2e6a` of Math-#447 (`runs447_f6ba2e6a.json`, `hosted_runs_line_f6ba2e6a.md`, the PR-body
refresh script with its output `pr447_body_carry.md`, and the post-merge read-back as posted at 447/6138936108). State:
Math-#445 merged as `6b5b9a5d` (05:15Z) and Math-#447 merged as `7f5f1c82` (05:27:53Z), both by the distinct integrator;
the landed tree is `d2f5bd70`, the module and the register are the reviewed blobs (SHA-256 `3164ccfd…`, `64674ac4…`).
Lifetime slice proposed with its packet, awaiting routing; RN probes preserved as design input; the main#386 read offered to
GROK-H03 without a response yet. Nothing merged or marked ready by this lane; every new declaration `specified`; nothing
proved; no inhabitant constructed; no label moved.
