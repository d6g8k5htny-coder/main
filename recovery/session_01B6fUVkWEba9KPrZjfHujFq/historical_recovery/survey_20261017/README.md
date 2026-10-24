# Recovery-routing survey, 17 October 2026

This survey checks, for each of the 18 sections of
[`../results_20261013/TOP_CANDIDATES.md`](../results_20261013/TOP_CANDIDATES.md), three things as of 17 October 2026, 22:15 UTC:

- whether any lane other than this one has cited or integrated the recovered carrier since 13 October;
- whether the section's recommended next step is still open;
- who currently owns the related node or task, according to native records only.

`SURVEY.json` holds each investigator's row together with the adversarial verifier's check of that row. Where they disagree, the verifier's corrected value is the one to use.

**Method.** Six read-only investigators covered three sections each, and three adversarial verifiers re-ran every search. Searches used `git log -S`/`-G` and `git grep` on `origin/main` of `main` (`9e7bacb8`) and `Math-` (`dc7aca2f`), plus the hardening branch, and `gh api` reads of `main` issues, PRs and comments since 13 October. Math- was read through its local git clone only, and no open-PR content was read. No writes were made anywhere.

**Limits**
- The verifiers belong to the same model family as the investigators.
- Owners are named only where a native record states ownership.
- Automated site inventories, such as `docs/site/current.json`, do not count as citations.
- Scientific effect: NONE. Nothing here promotes, demotes or reviews a mathematical claim.
