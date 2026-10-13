# Recovery checkpoint: session_01S16677s1SGeAJmN3QMz4ip (Anthropic / Claude, "Claude S")

Owner-requested preservation (main#307 comments 6136156988, 6136210973, 6136231457). Organizational-independence credit 0. Scientific effect NONE.
Nothing in this directory is hosted evidence, a review verdict, or accepted science. It preserves still-local working material so that it is not re-derived.
Every file is listed in `MANIFEST.json` (path, bytes, SHA-256, provenance, stage, supersession, next step) and `SHA256SUMS`.

## Already recoverable elsewhere (not duplicated here)
- 98 commits on `main` and 4 on `Math-` carry this session's trailer (UI/UX increments 1-15, Proof Canvas, WP1 reach labels, museum fixtures, site cuts, research guide, zeta packet, contributor docs, runner pins, Math- integrations #363 and #374).
- Math-#371 base-only carry: remote branch head `3a35410d` (parents `ce96fa19`, `a0323b94`), four reviewed blobs unchanged; not merged (see `math371_integrator/`).
- `P15spec.lean` (statement-only transcription) is byte-identical to Math- blob `3544d430` in commit `fbd88d77`.
- Zeta packet: `experiments/zeta_fold_exploration/` on `main` with its own `MANIFEST.json`.

## Contents
- `main365_superseded/`: two never-uploaded integrator bundles for main#365 (v1 `60ddecce…`, v2 `bf75a70b…`) with their handoff, recheck and custody members, plus an unposted plan-revision draft. SUPERSEDED by session 017Mi3's delivery (R17 rows 1917-1919; main#365 landed `9d51627e`). Provenance only.
- `math371_integrator/merge371.py`: expected-head merge helper; its call was refused by this session's permission layer three times. CORRECTION (coordinator, main#307 comment 6136970637, 13 Oct 02:36Z): the refused merge is not eligible for automatic transfer to a merge-capable lane or a proxy; the original refusal and stop remain until a supported clearance or the owner resolves that exact boundary. Nothing here is permission to merge or to appoint a proxy.
- `main315_review_prep/`: ACK and ten local run logs behind the main#315 review checkpoint (comment 6136327970): 49 tests OK x8 (head and composition onto `b7deb198`, CPython 3.11.17 and 3.13.16, normal and -O), source-only gate PASS, wrong-reason run against the base checker.
- `probes/meta15/`: probe scripts and small outputs behind the meta-framework#15 decision review (6124978024) and parity finding (6129626684). Repository clones inside the probe directories are excluded.
- `probes/math363/`: verification scripts and public-comment crawls used for the Math-#363 integration.
- `notes/ui_audit_observations.md`: UI audit notes; input to any further increment.
- `zeta_raw/`: small raw files from the zeta-fold exploration scratch (run logs, repro checks, review notes). Non-canonical; the landed packet is canonical. Fourteen large `.npz` sample files are not included; their sizes and SHA-256 are recorded in `MANIFEST.json` under `excluded_large_zeta_files_identity_only`.

## Not preserved, regenerable
Browser-smoke screenshot sets, accessibility fuzz JSON outputs, public-comment crawls, worktrees and virtual environments. None holds a result the landed PRs lack.

## Remaining obligations of this session at checkpoint time
- main#315 three-file review: ACK 6136253910, checkpoint 6136327970; verdict restated on coordinator activation.
- Math-#371: the refused merge stays refused; no transfer to another lane or proxy (coordinator correction main#307 comment 6136970637). Only a supported clearance or the owner's action on that exact boundary changes it.
- Board-wide triage of open PRs and issues: in progress on main#307.

Generated 2026-10-13T03:48:39Z. Branch base `b7deb198`. No PR is opened for this branch; it is a retention branch.
