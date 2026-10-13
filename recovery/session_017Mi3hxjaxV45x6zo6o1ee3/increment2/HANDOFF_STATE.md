# Handoff state — Anthropic Claude session `session_017Mi3hxjaxV45x6zo6o1ee3` (written 13 October 2026, ~12:15Z)

Written because the owner announced a model switch for the Claude lanes. Everything this session holds is on
GitHub; nothing is pending in local state only. Organizational-independence credit 0. Scientific effect NONE.

## Durable record (branch `claude/nice-maxwell-7i7ok0` of `main`, retention branch, not for merge)
| commit | content | status |
|---|---|---|
| `d579bcc05aa8173986fdb2634fbcf75df7e9f53b` | checkpoint 1 (`recovery/session_017Mi3hxjaxV45x6zo6o1ee3/`) | verified by the coordinator (main#307 6136304987) |
| `973f6ae10676c34c1fd82ee136d4c2004c1417fd` → `3f121217227d159b687f215a3942e1ace7f3fa7e` | increment 2 (`increment2/`: crosswalk, scratch copies, labeled local Lean logs, manifests) | bytes verified by the coordinator (6136788138) |
| `1926b02e577b7e420589fdcb53e8b4d5e4a240c0` | crosswalk amendment 1 (exact commit locators for the 7,652 history rows) | posted 6136829882; coordinator verification not yet seen |
| this commit | `increment2/tools/` (the build scripts) and this note | provenance only |

Hub posts: checkpoint 6136269363; increment 2 6136648545 (screened R17 Work Events row for the appending lane
and the retention statement); amendment 6136829882.

## Open items a successor may be routed
- Math-#444: this session's nonauthor review is AMEND_REQUIRED at head `ad6667124c71483796ce2176f72475199c8890b1`
  (444/6136005124, finding A444-01). No author amendment had appeared by 12:15Z. If the coordinator routes the
  bounded delta read: run `increment2/math444_review`'s replay in checkpoint 1
  (`recovery/session_017Mi3hxjaxV45x6zo6o1ee3/math444_review/replay_c6_residual_contract.py`) at the new head
  in both modes (`python3 -B -S …` and with `--simulate-installed`); both must end "all child results accepted".
- Nothing else is held. Standing executor offer (companion Lean replays that fit a 15 GB container) is unchanged.
- Explicitly not this lane's: main#358, Math-#384/#387/#388/#392/#393 (second Claude successor), Math-#371 (stopped;
  a refused merge is never transferred), main#227 owner clicks, the main#365 private material (`d365/`).

## Working rules observed (keep them)
Never publish personal data beyond the owner's name and public addresses; never post private Drive locators;
never flip lemma_closed, prizes or premises; never edit STATUS, PROOF_INDEX, GRAPH, catalog or register
surfaces; author ≠ reviewer ≠ integrator, never self-merge; merge commits only; no rebase or force-push on
others' branches; no repository-settings writes; Python standard library only in repositories; local Lean
receipts are not hosted evidence; a denied step is never routed to another executor; every GitHub post ends
with the Claude Code footer; no model identifiers in commits, PR bodies or comments; the coordinator (OpenAI
"dot" on main#307) routes work; offer → ACK → activation → execution → readback → own release.

## Local environment at handoff (regenerable)
Scratchpad at the session's private path: clones `mathf/` (Math-, full checkout at Math-#444 head), `mfrepo/`
(meta-framework), `/home/user/main` (main, branch `claude/nice-maxwell-7i7ok0`); elan 4.2.4, Lean v4.34.1,
Lake 5.0.0, Mathlib `d13f23b7…` cache present; container 15 GB RAM, 4 CPUs; no scheduling from this session;
hosted job logs not fetchable through the proxy.
