# CBZ-0 decision ledger

Append-only. Format: `[timestamp] [phase] [layer] action — rationale — reversible? — evidence`.

- [2026-10-29T16:50Z] [P1] [C] Started the Crystalline Baseline Protocol with read-only discovery of all ten in-scope repositories — the directive orders Phase 1 first — reversible: yes (no writes) — `audits/cbz0_20261029/INVENTORY.md`.
- [2026-10-29T16:55Z] [P1] [C] Declined `git reset --hard` on the existing Math- checkout (refused by the session's safety layer as irreversible local destruction) and used a detached worktree on `origin/main` instead — same result, nothing at risk — reversible: yes — this entry.
- [2026-10-29T17:00Z] [P1] [B] Rule: a grep for `sorry`/`admit` is not a Layer A audit, because 17 of 17 raw hits in Math- were prose in comments and docstrings. Layer A source checks must lex Lean comments and literals — reversible: yes — `audits/cbz0_20261029/TOOLING.md`.
- [2026-10-29T17:00Z] [P1] [C] Built `lean_layer_a_scan` (tooling-gap rank 1: unblocks verification of 153 Math- Lean files outside the gated `formal/` package) with a mutation-tested self-test, per §2.3 — reversible: yes — Math- PR.
- [2026-10-29T17:05Z] [P1] [B] Rule: every Layer A auditor ships with a harness control (the unmutated tool must pass the same harness) next to its mutation kills, so an import or harness failure cannot count as a kill — reversible: yes — `tests/test_lean_layer_a_scan.py`.
- [2026-10-29T17:05Z] [P1] [C] Added `layer-a-lean-scan.yml`, triggered on any `**/*.lean` change, because the full regression suite does not trigger on Lean files and a new `sorry` in a frontier directory would otherwise pass unseen — reversible: yes — Math- PR.
- [2026-10-29T17:06Z] [P1] [C] Recorded G2–G9 as PARTIAL, NOT RUN or NO TOOL where that is what was observed, instead of inferring green from older runs (A7, §10 "cached status does not count") — reversible: yes — `audits/cbz0_20261029/INVENTORY.md` §3.
- [2026-10-29T17:06Z] [P2] [C] Deferred deleting branches and closing PRs: 69 PRs and 225 October branches include work other agents updated today (e.g. Math-#526 at 16:34Z), and remote branch deletion is in §9.2's irreversible class (one verification cycle first). The 242 fully merged branches are the first, lossless prune batch — reversible: n/a (not executed) — `audits/cbz0_20261029/INVENTORY.md` §4.
