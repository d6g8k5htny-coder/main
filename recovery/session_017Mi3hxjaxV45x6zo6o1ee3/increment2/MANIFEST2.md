# Recovery increment 2 — Anthropic Claude session `session_017Mi3hxjaxV45x6zo6o1ee3`

Generated 2026-10-13T02:08:00Z. Follows checkpoint 1 (commit `d579bcc05aa8173986fdb2634fbcf75df7e9f53b`, verified by the coordinator in main#307 comment 6136304987) and answers its three asks: map every local file to an exact remote artifact or upload it, preserve local Lean logs with labels, register the checkpoint. Performer: Anthropic Claude, Claude Code cloud session `session_017Mi3hxjaxV45x6zo6o1ee3`, delegated for Dylan Roy. Organizational-independence credit 0. Scientific effect NONE. Everything here is scratch: drafts, probes, local diagnostic outputs, logs. Nothing is a receipt, a review record or a landed object.

## 1. The map (`RECOVERY_MAP.tsv`)

One row per local file of the session's scratch area (plus one row per unwalked directory). Dispositions:

| disposition | rows | meaning |
|---|---:|---|
| MATCHED_SESSION_COMMIT | 7055 | byte-identical to a blob in a commit carrying this session's trailer; column `locator_or_reason` gives repo, commit, path and PR. All 191 Math- and 2 meta-framework session commits are on their default branches; 14 of the 15 main session commits are (the fifteenth is checkpoint 1 on the retention branch) |
| MATCHED_REPO_HISTORY | 7652 | byte-identical to a blob somewhere in the history of Math-, meta-framework or main (every branch and pull-request head fetched on 13 October 01:55Z); path given |
| UPLOADED_CHECKPOINT1 | 0 | same SHA-256 as a file published in checkpoint 1 and not otherwise matched (0 here: the checkpoint-1 originals match commit d579bcc0 directly and appear under MATCHED_SESSION_COMMIT with `on_default_branch` False) |
| UPLOADED_INCREMENT2 | 2289 | published in this increment under `scratch/<original relative path>` or `local_lean_evidence/` |
| EXCLUDED | 572 | not uploaded; the reason and, where one exists, an exact remote locator are in the row and in `EXCLUDED.json` |

A match is a byte identity (git blob id), not a name match: an older draft of a landed file is unmatched and therefore uploaded, with `related_landed_paths` in `MANIFEST2.json` naming landed files of the same name where they exist.

## 2. What this increment publishes

2291 files, 13,764,570 bytes, each listed in `MANIFEST2.json` with published SHA-256, as-run SHA-256, byte counts and a scrubbed flag, and in `SHA256SUMS2.txt`. Text files had the container's private scratch path replaced by `<SCRATCH>` and its UUID by `<SESSION-UUID>` (456 files changed by that alone); nothing else was edited.

### 2a. Local Lean logs and receipts (`local_lean_evidence/`), labeled

- `torus_filtration_20261011_replay/`: the untracked outputs of the local `companions/torus_filtration_20261011/replay.sh` run of 12 October 22:52–22:57Z at Math- `40440ed8` (Lean sources, build/census/axiom/type logs, negative-control logs, `source.json`, `receipt.json`). **Local, non-authoritative; `receipt.json` invalid as a receipt.** See its `LABEL.md`.
- `math_formal_partial_gate/`: the untracked outputs of the partial local Math- `formal/` gate run of 12 October 22:45–22:47Z at `40440ed8` with the leanchecker step replaced by a logged SKIP after the full gate was OOM-killed. **Not a gate pass; `receipt.json` invalid as a receipt.** See its `LABEL.md`.
- `scratch/gate-run/`, `scratch/gate-run2/`: the two 28 September mutation runs of main `frontiers/downstream_gate_20260925` (REPORT.json, outputs, per-mutant stdout/stderr); mutant trees excluded as regenerable. Software replay only; promotion permission false in both reports.

### 2b. Areas (`scratch/<directory>`)

| area | files | bytes | what it is |
|---|---:|---:|---|
| `(root)` | 802 | 5,870,054 | working files of every round, 26 September – 13 October: PR bodies and body-patch payloads, comment drafts and readbacks, probe and exact-check scripts, successive versions of edited records (for example `audit_new*.md` / `audit_rb*.md` are the successive texts of the edited audit comment main#229 6004355577, whose earlier versions GitHub does not keep), small logs |
| `torus_filtration_20261011_replay` | 22 | 599,286 | LOCAL Lean replay outputs of the torus-filtration companion (see LABEL.md): non-authoritative, receipt invalid |
| `rtfOEb` | 64 | 596,314 | copy of the downstream-gate packet with `downstream-gate-results` and `log.txt` (28 September); mutant trees excluded |
| `rt` | 64 | 586,606 | copy of the downstream-gate packet with `downstream-gate-results` (28 September); mutant trees excluded |
| `gate-run2` | 51 | 533,302 | second mutation run of the same packet, 28 September 23:35Z |
| `gate-run` | 53 | 500,343 | main `frontiers/downstream_gate_20260925` mutation run of 28 September 23:01Z (REPORT.json names the source hashes; mutant trees excluded as regenerable) |
| `v22` | 10 | 494,766 | draft of Lean packet v2.2 (`trace.txt` is the local Lean build trace); landed in Math-#264 |
| `r178` | 21 | 353,888 | probe scripts, logs and JSON outputs behind the certified tail-constant work (Math-#178 thread, Math-#190) |
| `pr105` | 54 | 317,651 | replay probe scripts and stderr captures for the review work on the PR named by the directory |
| `pinout` | 27 | 250,865 | mutation-run outputs (per-mutant stdout/stderr and summaries) |
| `v19` | 9 | 205,769 | draft of Lean packet v1.9; landed in Math-#259 |
| `verdicts` | 21 | 201,894 | verdict texts as drafted for posting (the posted comments are authoritative) |
| `rw7` | 20 | 199,087 | review-round probe scripts and logs (rw series) |
| `math_formal_partial_gate` | 15 | 185,363 | LOCAL partial Math- formal gate outputs (see LABEL.md): not a gate pass, receipt invalid |
| `takeover` | 23 | 179,191 | 12 October takeover-round notes and posted texts (API dumps excluded) |
| `ev331` | 292 | 162,990 | evidence replay invocations for Math-#331 (process.json, stdout.bin, stderr.bin per invocation) |
| `rw5` | 18 | 162,808 | review-round probe scripts and logs (rw series) |
| `pr282` | 35 | 152,663 | main#282 custodian working texts (drafts, templates, resolve script) |
| `iw_out` | 43 | 152,421 | mutation-run outputs (per-mutant stdout/stderr and summaries) |
| `rw6` | 23 | 140,694 | review-round probe scripts and logs (rw series) |
| `rd6` | 12 | 112,243 | read-round probe scripts and outputs (rd series) |
| `v21` | 7 | 111,008 | draft of Lean packet v2.1; landed in Math-#262 |
| `mf15` | 30 | 110,851 | meta-framework#15 decision working texts and parity-probe outputs (local repository copy excluded) |
| `l331` | 14 | 110,044 | replay stdout/stderr for the engineering read of Math-#331 |
| `l356` | 4 | 100,816 | replay stdout/stderr for the read of Math-#356 |
| `rw4` | 13 | 86,565 | review-round probe scripts and logs (rw series) |
| `pr383` | 28 | 75,019 | main#383 author working texts |
| `v15` | 14 | 72,453 | draft of Lean packet v1.5 before the K-stack amendments; landed in Math-#253 |
| `rw3` | 7 | 66,094 | review-round probe scripts and logs (rw series); the posted reviews are the records |
| `s307` | 2 | 59,351 | hub #307 posting helpers |
| `rd5` | 9 | 50,694 | read-round probe scripts and logs (rd series) |
| `v17` | 8 | 46,679 | draft of Lean packet v1.7; landed in Math-#257 |
| `dl320` | 17 | 43,241 | files downloaded from Math-#320 for a read (non-identical to any fetched ref; kept) |
| `v13` | 5 | 42,772 | draft of Lean packet v1.3 (cap first-exit stack); superseded by the landed packets of Math-#253–#266 |
| `v16` | 8 | 41,390 | draft of Lean packet v1.6; landed in Math-#255 |
| `v21m` | 6 | 40,880 | amendment working set for v2.1 |
| `v22m` | 6 | 38,920 | amendment working set for v2.2 |
| `dl328` | 12 | 37,558 | files downloaded from Math-#328 for a read, with local test outputs |
| `v20m` | 8 | 34,892 | amendment working set for v2.0 |
| `v23m` | 6 | 33,703 | amendment working set for v2.3 |
| `r169` | 3 | 31,690 | probe scripts for the nonauthor Slice A/B/C reviews of Math-#169 |
| `dry` | 3 | 29,238 | dry-run outputs |
| `rd4cf` | 7 | 26,544 | read-round probe scripts and outputs (rd series) |
| `v19m` | 6 | 26,536 | amendment working set for v1.9 (K-stack AMENDs) |
| `lean25` | 4 | 25,462 | Lean sources of the v2.5 draft; landed in Math-#285 |
| `v18` | 5 | 24,580 | draft of Lean packet v1.8; landed in Math-#258 |
| `lean26` | 2 | 21,401 | Lean sources of the v2.6 draft; landed in Math-#295 |
| `d5_collar_20260928` | 4 | 20,729 | draft of the D5 collar packet (28 September) |
| `c5051` | 4 | 20,410 | probe scripts (C50/C51 lemma work on Math-#233) |
| `mf27` | 8 | 19,919 | meta-framework#27 read working texts |
| `collision_packet` | 2 | 19,461 | draft PROOF.md and SOURCE_FILES.json of the witness-collision packet (28 September) |
| `elder_dimension_lift_claude_20260928` | 2 | 17,845 | draft of the elder dimension lift note (28 September) |
| `rd3cf` | 2 | 17,544 | read-round probe scripts and outputs (rd series) |
| `a022` | 6 | 16,949 | mutant control files and workflow script of a 28–30 September packet |
| `rcap` | 6 | 16,937 | working files of this session; see RECOVERY_MAP.tsv |
| `kreplies` | 8 | 16,285 | replies to the K-stack reviews of Math-#257–#260 (text and payload) |
| `leancap` | 2 | 15,077 | Lean sources of the cap first-exit packet draft |
| `pr132` | 38 | 14,908 | replay stdout captures for the nonauthor review of Math-#132 |
| `r185` | 2 | 13,782 | probe scripts for the Slice A review of Math-#185 |
| `pr125` | 46 | 13,688 | replay stdout captures for the nonauthor review of Math-#125 |
| `remote_pair_law_claude_20260928` | 2 | 13,083 | draft of the remote pair law note (28 September) |
| `v20` | 1 | 12,755 | draft of Lean packet v2.0; landed in Math-#260 |
| `m444` | 8 | 12,610 | Math-#444 review working texts |
| `r181` | 2 | 12,255 | probe scripts for the Slice B review of Math-#181 |
| `m443` | 8 | 11,599 | Math-#443 read working texts |
| `pr128` | 38 | 10,868 | replay stdout captures for the nonauthor review of Math-#128 |
| `ev305_opt` | 7 | 10,494 | evidence replay of Math-#305 (optimized mode) |
| `ev305_normal` | 7 | 10,395 | evidence replay of Math-#305 (normal mode) |
| `pr384` | 6 | 9,826 | main#384 notes |
| `v23` | 1 | 7,149 | draft of Lean packet v2.3; landed in Math-#266 |
| `toolchain` | 3 | 7,009 | toolchain-readiness notes (owner approval relay, executor confirmation, PR body) |
| `c235` | 1 | 6,886 | probe script for the Math-#235 read |
| `red282` | 7 | 6,410 | main#282 rerun working texts |
| `rd4` | 7 | 6,276 | read-round probe scripts and logs (rd series) |
| `rcsout` | 19 | 5,746 | replay outputs for the Math-#135 R-slice review |
| `rcsout-o` | 19 | 5,746 | replay outputs for the Math-#135 R-slice review (-O) |
| `p15_tail_replay_out` | 17 | 5,013 | replay outputs for the P15 sharp even tail review |
| `p146m` | 15 | 4,874 | mutant replay outputs for the Math-#146 review |
| `p146m-O` | 15 | 4,874 | mutant replay outputs for the Math-#146 review (-O) |
| `p146out` | 15 | 4,874 | replay outputs for the Math-#146 review |
| `p146out-O` | 15 | 4,874 | replay outputs for the Math-#146 review (-O) |
| `landing_fix` | 1 | 3,981 | patch used for a landing fix |
| `p157_verify_out.json` | 19 | 3,760 | replay outputs for the Math-#157 Slice A review (directory despite its name) |
| `a41` | 1 | 2,943 | review texts and API readbacks of early-October reads |
| `replies213` | 3 | 2,902 | replies on Math-#213 |
| `collar` | 1 | 2,809 | collar probe script |
| `synth` | 4 | 857 | synthesis note, results and certified d=2 script |
| `rwin` | 2 | 523 | window-check log and stderr |
| `l365` | 2 | 208 | stderr captures from the main#365 dry-run builds |

Directory names follow the pull request, packet or round they served; where a note above is generic, the files themselves and `RECOVERY_MAP.tsv` carry the detail. Posted texts (verdicts, readbacks, bodies) are drafts: the GitHub comment or body is the authoritative record.

## 3. Excluded (`EXCLUDED.json`)

| files | reason |
|---:|---|
| 240 | mutation run output tree (regenerable from frontiers/downstream_gate_20260925 at the hashes in gate-run/REPORT.json) |
| 96 | mutation run output tree (regenerable, see gate-run2/REPORT.json) |
| 68 | GitHub API response saved during work; regenerable from the API |
| 48 | mutation run output tree (each mutant is the packet with one mutation applied; regenerable from the packet named in rt/downstream-gate-results/REPORT.json) |
| 48 | mutation run output tree (regenerable, see rtfOEb/downstream-gate-results/REPORT.json) |
| 32 | Lean build products (generated IR and setup files; regenerable; also matched by the repository .gitignore rule .lake/) |
| 14 | mutant copies of main packets used for local mutation checks (regenerable) |
| 6 | downloaded CSV exports of the shared R17 registers (the registers themselves are the source; linked from docs/WORKSPACE.md) |
| 4 | copy of the Mathlib cache tool source (upstream Mathlib; regenerable) |
| 2 | HTML rendering of the third-party document above |
| 2 | owner-supplied third-party document (the owner holds the original); not project work product |
| 2 | text extraction of the third-party document above; contains its authors' e-mail addresses |
| 2 | binary or checkout pointer (no extension); regenerable |
| 1 | privacy screen: third-party e-mail address (1); Drive locator other than the public WORKSPACE links |
| 1 | clone of Math- (regenerable; its untracked local Lean evidence is published separately below) |
| 1 | clone of meta-framework (regenerable) |
| 1 | Python virtual environment (third-party packages) |
| 1 | third-party Python package copies (pypdf) |
| 1 | main#365 private delivery material: owner-only Drive locators and permission observations; public identities on main#365 (bundle 19034 B afd1fff8...) |
| 1 | this increment's build directory |
| 1 | checkpoint-1 staging copy (published as commit d579bcc05aa8173986fdb2634fbcf75df7e9f53b) |

GitHub API responses are listed with the URL of the first and last record they hold, so each can be re-fetched. The main#365 private material (`d365/`) stays excluded: its public identities are on main#365 and its bytes are in the owner-only Drive location recorded in the custody note.

## 4. Verify

```
cd recovery/session_017Mi3hxjaxV45x6zo6o1ee3/increment2 && sha256sum -c SHA256SUMS2.txt
```
`MANIFEST2.json` → `files[].original_sha256` is the as-run identity of each scrubbed file; an unscrubbed file has equal published and original hashes.

## 5. Retention, registration, obligations

- Retention: branch `claude/nice-maxwell-7i7ok0` of `main` is a retention branch, not for merge; retain this branch and its commits (checkpoint 1 `d579bcc05aa8173986fdb2634fbcf75df7e9f53b` and this increment) while any work relies on them, until a verified durable replacement holds the actual bytes and the shared index is updated; an index alone is not a replacement. No deletion is requested.
- Registration: one screened Work Events row for the existing R17 register is requested from the appending lane in the hub post that announces this commit; this session has no native append route and makes no whole-file register write.
- Obligations held by this session: none open beyond the standing offers in checkpoint 1 section D. Math-#444 delta read follows if routed.
- Blockers: none. Platform limits unchanged (no scheduling from this session; hosted job logs not fetchable).
