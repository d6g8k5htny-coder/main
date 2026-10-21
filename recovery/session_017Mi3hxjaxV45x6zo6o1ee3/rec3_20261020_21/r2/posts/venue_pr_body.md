## Change and purpose

This draft PR is a **venue for #451's R2**: the full replay, by a provider other than Anthropic, of the 59 README §8 commands of `experiments/zeta_fold_exploration` (slice C).
- It adds one stub, `reviews/zeta_fold_slice_c_r2_replay_20261020/README.md`.
- The stub states the request, the source identities, what the finished record must hold, and its limits.
- **Scope:** an evidence-record venue only. Scientific effect NONE.

Why a new PR: a Cursor cloud agent starts from a pull request's head branch. #451's branch was deleted after its merge, so the earlier request there failed at startup ([451/6281525014](https://github.com/d6g8k5htny-coder/main/pull/451#issuecomment-6281525014)). Cursor agents also can't post on issues.

**Owner approval.** Dylan Roy chose "Allow a venue PR" in Claude Code session `session_017Mi3hxjaxV45x6zo6o1ee3` (20 October 2026, about 02:25Z), with these conditions, verbatim:

> Use one verified non-Anthropic execution worker for R2, check existing claims first, and preserve all 59 commands, environment details, output hashes and discrepancies. Keep README §9 unchanged until the complete replay is reviewed. No merge or scientific acceptance is authorized by the venue approval.

**Linked records**
- Ledger: [451/6281117591](https://github.com/d6g8k5htny-coder/main/pull/451#issuecomment-6281117591).
- Coordinator's offer: [451/6281445636](https://github.com/d6g8k5htny-coder/main/pull/451#issuecomment-6281445636).
- Board: [#419](https://github.com/d6g8k5htny-coder/main/issues/419).

## Current disposition

- **Head and owner of this PR:** `d68d3547`, opened by Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3`. It is a venue; nothing here is to be integrated.
- **Claims:** none on R2 at 02:30Z (#451 and #419 checked). One Cursor agent will be summoned here for the replay. It must verify a non-Anthropic provider and post its own pickup before running.
- **Reviews:** none. The finished replay record needs its own review before any README §9 change.
- **Unresolved findings:** none.

## Verification

Run locally on this head:
- `python3 -B -S tools/workspace_landing_check.py`: no problems.
- `python3 -B -S tests/test_research_delivery.py`: 28 OK.
- `python3 -B -S -m unittest discover -s tests -p test_workspace_landing.py`: 15 OK.
- `python3 -B -S tools/navigation_check.py`: no problems.

The packet tree `bca06db86581b2ff878fa321c4040c7b8f7bd55d` is the same at `0a7e5145`, `73a3cfcd` and this base `5cab7663`. Hosted checks run on this PR as usual.

## Evidence and limits

- This PR changes no packet file, no README §9 and no status surface.
- Partial chunks are progress records, not R2. A run by an Anthropic performer earns no R2 credit.
- The venue approval authorizes no merge and no scientific acceptance.
- Organizational-independence credit 0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
