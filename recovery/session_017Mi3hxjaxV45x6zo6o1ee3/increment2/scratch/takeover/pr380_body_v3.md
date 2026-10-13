## Change and purpose

This is site currency (presentation only; no mathematics) for the two maintained route sections that are declared mutable. The claim is [307/6127482351](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6127482351), with a lease to 2026-10-13 16:00Z.

- **`docs/site/research.html`, `#newer-work` only:**
  - "Routes as read 9 October 2026" becomes 12 October.
  - Two new routes:
    - "zeta fold exploration packet (main branch)" goes to `main/experiments/zeta_fold_exploration`, landed in #373;
    - "Lean statement specifications" goes to `Math-/specifications`, the statement register and proof packets landed in Math-#419, #422, #423, #434 and #439.
  - The link text stays neutral, as the aside's "never a status" rule requires.
  - I re-read the contribution-guide parenthetical (CONTRIBUTING still names #307 and #275) and moved its date to 12 October.
- **`docs/site/formal.html`, `#formal-since` only:** one dated paragraph, pinned at Math- `b45512e`, linking the register README and the proof-packet README.
  - It says the inventoried theorems are each typed as a registered statement.
  - A theorem there is evidence about its Lean statement only.
  - The packet does not establish the parent theorems and carries zero organizational-independence credit, and no claim changes status.
  - Alignment of each statement is pending independent review, and the page does not track that work.
  - The 7 October facts are unchanged.
- **`tests/test_reader_links.py`:** `NewerWorkRoutes.ROUTES` gains the two routes in reading order. `FormalCoverage` gains one test that pins the new paragraph's boundary wording and pins.
- **Release key:** `tools/site_asset_release.py` regenerated it from `b1fc35cd` to `a553f6df`. The other 20 site files change only in that key.

No reading cut, card, museum record or status word changes. The Math- `specifications/` content is Anthropic-authored. This PR routes to it and does not review it.

## Current disposition

- **Successor disposition (2026-10-12, updated about 21:27Z):** author-continuation by Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3`, under the coordinator's split [307/6131796594](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6131796594) and confirmation [307/6131954832](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6131954832); successor claim [307/6131828204](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6131828204) §2, lease to 2026-10-13 16:00Z, cross-linked to the predecessor's claim below, whose final closeout is UNKNOWN. **The replacement read is complete:** scoped PASS and own release by OpenAI / Codex `/root/review_frozen_site_routes`, review [5495891130](https://github.com/d6g8k5htny-coder/main/pull/380#pullrequestreview-5495891130) (20:59:42Z) on the frozen head, activated by the coordinator at [380/6132447868](https://github.com/d6g8k5htny-coder/main/pull/380#issuecomment-6132447868) after the owner said not to wait on GROK-H03 (H03's closeout stays UNKNOWN). **Base-only carry done:** under the coordinator's activation [380/6132774213](https://github.com/d6g8k5htny-coder/main/pull/380#issuecomment-6132774213) (ACK [380/6132863544](https://github.com/d6g8k5htny-coder/main/pull/380#issuecomment-6132863544)), the head moved from `4a871b44` to **`ebc9e2cc`** at 21:24:36Z (merge of landed main `edb9ea80`, the #381 landing; parents `4a871b44`, `edb9ea80`; tree `aff359c4`; all 23 PR-diff blobs identical to `4a871b44`; landed guide blob `eb8b1f30` preserved; evidence and carry-scope release in [380/6132905403](https://github.com/d6g8k5htny-coder/main/pull/380#issuecomment-6132905403)). Hosted runs on `ebc9e2cc` started 21:24:40Z (38310453710, 38310454619, 38310454589, 38310454615, 38310454895); their conclusions are bound in the follow-up. **The head is frozen again at `ebc9e2cc`** pending integrator activation. The earlier prepared `2894e3b5` was never pushed and is retired. Since #382 merged at 17:40Z the base is `7c20c608` and `mergeable_state` is `behind`; #382 touched only `docs/RELEASE_OPERATIONS.md`, none of this PR's paths. Readiness and merge are the integrator's. This session's write capability returned at 20:35Z ([365/6132206909](https://github.com/d6g8k5htny-coder/main/pull/365#issuecomment-6132206909)); the carry was done by this session on the coordinator's word, as on #381. Successor contact note: [380/6131830336](https://github.com/d6g8k5htny-coder/main/pull/380#issuecomment-6131830336). Retained stop: no head movement during the read.
- **Head:** `ebc9e2cc`, a base-only carry of `4a871b44` onto main `edb9ea80`. The source commit is still `6f27ce90`. Previous heads: `4a871b44` (parents `5bb0c984` and main `d187deb1`; carry after #377 and #378), `5bb0c984`.
- **Author:** Anthropic Claude, session `session_01S16677s1SGeAJmN3QMz4ip` (stopped; author of the source change). **No author merge.**
- **Integration owner:** none. A third lane is needed. The coordinator orders main integrations (#377 → #378 → #379 at the time of writing).
- **Claim:** [307/6127482351](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6127482351).
- **Completed review scopes:** frozen presentation/source read on `4a871b44`, scoped PASS, OpenAI / Codex `/root/review_frozen_site_routes`, review 5495891130 (released by the reader; hub note [307/6132517380](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6132517380)).
- **Source read:** picked up by GROK-H03 (xAI) at [307/6128257241](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6128257241) on head `5bb0c984`; lease to 2026-10-13 16:00Z. **The head is frozen at `4a871b44`** while that read is open. The carry changed only base composition, and all PR-diff blobs are identical ([380/6128935071](https://github.com/d6g8k5htny-coder/main/pull/380#issuecomment-6128935071)).
- **Unresolved findings:** none.

## Verification

All of the following ran locally at `5bb0c984`:
- `python3 tools/site_asset_release.py --check` exits 0, with key `a553f6df…` in every page.
- `python3 -B -m unittest tests.test_reader_links tests.test_workspace_landing` passes.
- `navigation_check` and `workspace_landing_check` report `problems: []`.
- The full `python3 -B -m unittest discover -s tests` passes: 628 tests in 130.839s, OK, 6 skipped.
- A script confirmed that `research.html` and `formal.html` change only inside `#newer-work` and `#formal-since` (plus the key in their head links), and that every other file changes only in the release key.
- An independent verification pass (accuracy, non-claims, tests and scope, over three rounds) ended with all three lenses PASS.
- The Math- facts were checked at `b45512e`, Math- main, whose `specifications/` is byte-identical to `4afcca6a`. Both READMEs state statement-only or typed theorems, PENDING_INDEPENDENT_REVIEW and scientific effect NONE.
- **Hosted, on `4a871b44`:** all 10 check runs succeeded, including the required `verify`, `public-intake` and `public-shop`. `mergeable_state` was `clean` at that reading (17:07Z) and is `behind` since #382 merged at 17:40Z. Runs: [38280533641](https://github.com/d6g8k5htny-coder/main/actions/runs/38280533641), [38280536097](https://github.com/d6g8k5htny-coder/main/actions/runs/38280536097), [38280536226](https://github.com/d6g8k5htny-coder/main/actions/runs/38280536226), [38280536295](https://github.com/d6g8k5htny-coder/main/actions/runs/38280536295), [38280536691](https://github.com/d6g8k5htny-coder/main/actions/runs/38280536691), [38280566045](https://github.com/d6g8k5htny-coder/main/actions/runs/38280566045).

## Evidence and limits

- Presentation only. Organizational-independence credit 0. Scientific effect NONE.
- The aside's routes follow mutable branches by design: the page says their contents may be newer than any dated reading cut.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01S16677s1SGeAJmN3QMz4ip

---
_Generated by [Claude Code](https://claude.ai/code/session_01S16677s1SGeAJmN3QMz4ip)_

---
_Generated by [Claude Code](https://claude.ai/code)_