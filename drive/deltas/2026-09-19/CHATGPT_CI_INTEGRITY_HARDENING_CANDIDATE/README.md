# ChatGPT CI-integrity-hardening candidate (2026-09-19) — stored as data, NOT applied

Three Drive objects created 2026-09-19 14:59 UTC in folder
`1glz4_QXkwq7NV7ub07eGYEjseYtcNr0K` (owner `th3realdyll@gmail.com`), announced
on pull request #2 at 15:00 UTC by the GitHub account `th3realdyll-afk` (not the
repository owner's account) as an "Owner-authorized isolated engineering
candidate from ChatGPT — NOT DEPLOYED, no merge requested".

| file | identity |
|---|---|
| `ci_integrity_hardening.patch` (31,374 B) | SHA-256 `7f6f1719df6659e18485e73d6ab558a02341c08b9e2b35bbb00d3ad8a3503356` — equals the digest the PR comment and `VERIFICATION.json` declare |
| `HANDOFF.md` (6,602 B) | `19c0761c…` (first computed here) |
| `VERIFICATION.json` (4,506 B) | `b55f582c…` (first computed here) |

## What the candidate is, in its own words

`HANDOFF.md`: "Status: locally tested; not pushed, not deployed, not a
mathematical review." Task `CG-CI-20260919-6c6f47b`, pinned base
`6c6f47b7884651b2330ab1380155f667caad0178`. A five-file patch: `.github/workflows/ci.yml`
(the nine guarded checker steps made mandatory, a new strict manifest guard,
`permissions: contents: read`, `persist-credentials: false`, a 30-minute limit,
`fetch-depth: 0`), `.github/workflows/research.yml` (the lane-check failure
mask removed, receipt upload failing when no files are found, final
governed-path check including untracked and ignored files), a new
`tools/manifest_integrity_check.py`, and two test files. `VERIFICATION.json`:
"146 scoped tests passed", "10/10 deliberate regressions detected", Python
3.13.5 runtime, and, under NOT_RUN: the Python 3.11 runtime, the full
repository test suite, the full manifest corpus, remote CI, independent review.
The patch touches no file under `research/`, `registers/`, `claims/`, `drive/`
or `engine/`.

## What this repository did with it

Stored the three files byte-exact, here, as a dated Drive delta. **Nothing was
applied.** `git apply --check` against the branch head of 2026-09-19 fails on
`.github/workflows/ci.yml` (the file has changed since the pinned base), so
the candidate would need reconciliation on a separate branch in any case, which
its own `HANDOFF.md` prescribes ("Use a real, clean checkout … Do not reset,
force-push, merge or edit Claude's branch … commit only these five files on the
separate branch"). Whether to integrate it is the repository owner's decision:
the announcing account is not the owner's, its "owner-authorized" statement is
not verifiable from this repository, and `governance/GIT_ADAPTATION.md` records
that CI permission and gate changes are not something a relayed instruction can
authorize. The candidate's ideas overlap work already on the branch: every
guarded CI step has used `if/then/else` since 2026-09-18 (`docs/FINDINGS_2026-09-18.md`
§1.5), and the failure mask this candidate names in `research.yml` is a real
remaining defect recorded here for the owner.

What this does not establish: that the candidate is correct, safe or wanted;
that its tests pass on this tree; or anything about any mathematical status.
