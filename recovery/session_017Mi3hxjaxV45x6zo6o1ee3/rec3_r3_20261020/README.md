# Session 017 R3 records and guidance correction, 20 October 2026 (retention only)

Session-initiated retention for Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3` (Claude017). It covers what happened after recovery increment 7 (`61f635b3`).
- Not for merge.
- Credit 0. Scientific effect NONE.
- The posted GitHub comments remain the canonical records.

## main#451 R3 (`m451/`)
| post | thread | comment id |
|---|---|---|
| `pickup451`: acceptance and exact scope (Dylan's assignment) | main#451 | 6279493098 |
| `pointer419` | main#419 | 6279497076 |
| `route451`: standing-route exclusion check and three conditions | main#451 | 6279743473 |
| `hold451`: merge refused by this session's auto-mode safety check; nothing reached GitHub | main#451 | 6279758562 |
| `hold419` | main#419 | 6279761539 |

**State when this was committed:**
- R3 is on HOLD.
- Dylan later approved the integration in session as a case-specific engineering and evidence-preservation change. That approval does not approve the mathematics or close R2.
- The session-permission step is pending. It goes through the supported process: Dylan switches the mode, then approves the exact action.
- Terminal time is 01:30Z on 20 October.
- R2 is open, and the packet README's §9 stays NOT_PERFORMED.

**Evidence (`m451/evidence/`):**
- Ruleset, check-run and refreshed-head commit snapshots. `SOURCES.md` links the PR, its commits and the third-party decision records, which are not copied here.
- The Codex summary and the usage-limit comment.
- Hosted log excerpts: the Ready-event `public-intake`, `verify` with checked commit `180ff565`, and `landing-checks`.
- Local landing and navigation outputs: `problems: []` (local runs, not hosted receipts).
- The 192-path list, the reviewed blob map at `8f1d846d`, and the update-branch response.
- C8's 23:31Z owner-decision record. The standing route, the main#446 hold and the #450 decision are linked in `SOURCES.md`.
- There is no pre-merge snapshot: the refused command never ran.

## Guidance correction (`guidance/`)
`GUIDANCE_CORRECTION.md` withdraws the overgeneralized "no integration unless dot allocates it", and explains the correction once. Historical records are unchanged.

## Notes
- **Readbacks.** `*_rb.md` files are API readbacks. Each equals the local text, apart from the trailing newline.
- **Checksums.** `SHA256SUMS` covers every other file in this folder.
