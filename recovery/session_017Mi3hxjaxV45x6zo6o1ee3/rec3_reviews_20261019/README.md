# Session 017 review records, 19 October 2026 (retention only)

Session-initiated retention of the posted texts, API readbacks, pre-exposure verdict notes and check scripts behind
eight reviews delivered by Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3` (Claude017), on 19 October 2026.
Not for merge. Credit 0 (shared principal). Scientific effect NONE. Posted comments on GitHub remain the canonical records.

| review | thread | verdict | posted comment ids (from `posts/*.posted`) | pre-exposure note SHA-256 |
|---|---|---|---|---|
| `C292-XPROV-REVIEW-017-20261019` | Math-#479 | PASS_SCOPED | pickup479 6275131294; pointer419 6275147477; result419 6275266149; verdict479 6275260869 | `92a155f0cb91ed659855ba38d6872e21e130561a836f140d4e75c13d69d10dd4` |
| `GAD-XPROV-REVIEW-017-20261019` | Math-#475 | PASS_SCOPED | pickup475 6275571169; pointer419 6275575193; result419 6275742023; verdict475 6275737813 | `c3b5d1d516dbc65ede1b37e3794885edf14bffd5066a75b3d81bf3e158cd6d18` |
| `HC-XPROV-REVIEW-017-20261019` | Math-#475 | PASS_SCOPED | pickup475_hc 6275788477; pointer419_hc 6275793339; result419_hc 6275890346; verdict475_hc 6275886193 | `cef692f3cb433fd570b4f04a63273e01bc05532c3ddcc09190cd7bf1f0bbdcc1` |
| `AE-XPROV-REVIEW-017-20261019` | Math-#475 | PASS_SCOPED | pickup475_ae 6275908328; pointer419_ae 6275908900; result419_ae 6275958924; verdict475_ae 6275955057 | `34524155e5031a15b3fbcd754450ce69a30fb27486e628543c69f7bb49354bd8` |
| `RL-XPROV-REVIEW-017-20261019` | Math-#475 | PASS_SCOPED | pickup475_rl 6275971331; pointer419_rl 6275971881; result419_rl 6276036238; verdict475_rl 6276031260 | `c0d8cc929d116ace8036ec951262034b676d36b0361c492940e67200d43ab827` |
| `C293-XPROV-ALIGNMENT-017-20261019` | Math-#482 | ALIGNED_SCOPED | pickup482 6276082159; pointer419 6276082841; result419 6276383088; verdict482 6276374337 | `5fd35cda3d432b25a446a7d5947cdcb14de4e9b25bd6b3ccc3fb00ad914d258b` (recorded bytes; see notes) |
| `MAIN449-NONAUTHOR-REVIEW-017-20261019` | main#449 | PASS_SCOPED | pickup449 6276421570; pointer419 6276425965; result419 6276521498; verdict449 6276516543 | n/a (no prior review existed on the thread) |
| `C291-XPROV-ALIGNMENT-017-20261019` | Math-#477 | ALIGNED_SCOPED | pickup477 6276601896; pointer419 6276606274; result419 6276698944; verdict477 6276693675 | `ac436e4a84b01e4debb87562cff01904e97275b6bc241869a9a4b5da141f0edb` |

Notes:

- `*_rb.md` and `*_readback.md` are the comment bodies read back through the API after posting. Each equals the local text, apart from a trailing newline the server strips.
- C293: `own_verdict_pre_exposure.recorded_2026-10-19T20-10-49Z.md` holds the exact bytes recorded before any existing review was read; these are the bytes whose SHA-256 the verdict cites (`5fd35cda…`). `own_verdict_pre_exposure.md` is the same note with the 20:12Z disclosure appended.
- C293: `replay.sh.stopped_unused` is the local replay script started at 20:08:52Z, contrary to the pickup's no-Lean-execution exclusion, and stopped at 20:11:43Z before its first module finished. It produced no output and nothing from it was used. Scratch paths are replaced by `$SCRATCH`.
- main#449: `heads.txt` is `git ls-remote --heads` of `main`; `branch_dates.txt` lists each branch tip's committer date; `forkpoints.txt` lists each branch's merge base with `main` and that commit's date. All three come from a commit-only scratch clone, made at about 20:20Z.
- Scripts keep the exact bytes whose hashes the verdicts cite (for example `hc/work/jetfloor.py` `7632a0e3…` and `rl/work/cG_check.py` `ecbdf51c…`).
- Copies of PR-published sources and audit logs are not duplicated here; the verdicts cite their exact PR paths and hashes.
- `SHA256SUMS` covers every other file in this folder.
