## Change and purpose

This records the owner's decision that the 2026-09-27 Cursor stop is **no longer in effect**. Agents read `AGENTS.md` first, and it still described the stop as retained. Scope: **governance text only**.

- **Source.** Dylan Roy gave the instruction directly in Claude Code session `017Mi3hx…` on 2026-10-05: *"Is no longer in effect per Dylan Roy aka me. Let all agents know my decision."* He then gave a blessing for the follow-on decisions about agent roles.
- **Announcement.** The decision went to every lane in [main#229 6002718035](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6002718035).

Changes:
- **New** `governance/OWNER_DECISION_20261005_CURSOR.md`, the dated record. It holds:
  - the owner's words, verbatim;
  - the decision: Cursor agents work on request for one bounded task each, with no self-assigned changes and no permanent windows, timers or loops; the Grok Bot roles are unchanged; trains while runners are scarce; one claim per object;
  - what stays unchanged.
- **Four lines repointed** to the decision. Each still described the stop as retained:
  - `AGENTS.md` lines 7–8;
  - `governance/OP-WORKFLOW-20260930.md` line 8;
  - `docs/BRAND_STYLE.md` line 28;
  - `docs/site/AGENTS.md` line 19.
- **Unchanged:** `governance/OWNER_STOP_20260927.md`, kept as historical evidence.

## Current disposition

- **Head:** `d65221c` on `claude/nice-maxwell-7i7ok0`. Base: `main` `5388b60`.
- **Author:** Anthropic Claude, session `017Mi3hx…`. **The author will not merge.** A non-Claude integrator is requested.
- **Claim:** main#229 6002718035.
- **Reviews:** none yet. A nonauthor read is welcome. The diff is 5 files: +55/−6.
- **Unresolved findings:** none.

## Verification

All run locally at `d65221c`:
- `python3 -B -S tools/workspace_landing_check.py`: exit 0, `problems: []` (16 pages, 125 local links).
- `python3 -B -S -m unittest discover -s tests -p 'test_*.py'`: 352 tests OK. Same under `-B -O -S`: OK.
- `python3 -B -S tools/navigation_check.py`: exit 0, `problems: []` (29 pages, 157 local links).
- `node --test tests/test_brand_identity.mjs`: 12 pass, 0 fail.
- No test pins the replaced wording (checked with `grep`). The new file's relative links resolve (`OWNER_STOP_20260927.md`, `OP-PRIVACY-20260927.md`).

## Evidence and limits

- Scientific effect is **NONE**. No STATUS, PROOF_INDEX, GRAPH, catalog, register, flag, premise or prize changes.
- The record relays an owner instruction given in a Claude session. The owner can confirm it or amend it here.
- Same GitHub account as every lane, so organizational-independence credit is 0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
