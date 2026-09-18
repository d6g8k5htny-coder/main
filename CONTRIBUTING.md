# Contributing

This repository is the git home of a research program that previously lived
entirely in a Google Drive shared drive. Read, in order:

1. `README.md` — what the research is and where it stands.
2. `governance/README.md` and `governance/GIT_ADAPTATION.md` — the rules, and how
   they map onto branches, pull requests and manifests.
3. `registers/README.md` — the coupled registers (review queue, frozen objects,
   quarantine index, work events, closure log, …).

## Workflow (the git form of DRAFT → CANDIDATE_VERIFIED → READY_FOR_REVIEW)

1. Draft in `sandbox/<task-id>/`. No authority, no manifest churn.
2. Freeze a candidate: move it to its destination (`research/…`, `packages/…`),
   add it to that directory's `_MANIFEST.jsonl` (bytes + SHA-256), and append one
   row to `registers/json/work_events.json` (append-only) and, if it is a review
   obligation, to `registers/json/review_queue.json`.
3. Open a **draft** pull request. The PR is the claim; its branch is the lease.
4. Run `python3 tools/registers_import.py --check`, `python3 tools/registers_check.py`,
   `python3 tools/verify_manifests.py` and `python3 -m pytest -q` locally; CI runs
   the same.
5. Mark the PR ready for review. A review records four separate dimensions:
   exact-object correctness, scope/dependencies, reviewer authorship/exposure,
   organizational independence. Same-provider review earns zero independence
   credit; author/coauthor checks are internal verification, not peer review.
6. Merge. Never rewrite history on `main`; never delete evidence (move it to
   `quarantine/` with a rollback record instead).

## Never

* Edit a frozen body in place. Issue a numbered successor.
* Compose the 2D upper track with the 3D lifetime track.
* Treat a display, Monte Carlo estimate or fitted exponent as a theorem constant.
* Treat a session CLOSE, a smoke test or registration as lemma closure.
* Cite anything under `legacy/`, `quarantine/` or the DO_NOT_OPEN vault as evidence.
