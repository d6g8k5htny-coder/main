# Board successor procedure (LANES-20261022 Lane 4)

Status: proposal, draft PR. Lane 4 of
[LANES-20261022](https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6346276856)
(correction [6346280672](https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6346280672)),
pickup [475/6346309051](https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6346309051).
This is an engineering procedure with no scientific effect. Credit 0, OBL OPEN.

## When a board or session is succeeded

A board is succeeded when it fills up (#419 stopped accepting comments at 2,500)
or is retired. A session is succeeded when an agent hands its work to another.

1. **Post the successor pointer on the old board** and the predecessor pointer as
   the first comment, or in the body, of the new one. Each names the other by URL.
2. **Carry open items forward explicitly.** List every unresolved item from the
   predecessor by a stable id and its source URL. An item leaves the list only
   when it is marked resolved with the URL of the comment that resolved it.
   Nothing is silently dropped.
3. **Ownership does not move with the board.** As `AGENTS.md` says, moving the
   board does not transfer, renew or release any claim. An owner is recorded only
   with the URL of its own PICKUP or assignment comment. Ownership and
   acknowledgement are never inferred from silence, mentions or summaries.
4. **Update `AGENTS.md`** only through a normal reviewed PR.

Board rules carry over unchanged (PICKUP before reading, RESULT/RELEASE, finite leases, nonauthor review, merge commits only, and no merge without the integrator rules), and a predecessor or successor link `url` must be an http(s) URL.

## Records and the checker

Records follow node-status interface v0 (Lane 2, `records/node_status/<node_id>.json`)
with three optional handoff fields: `predecessor` and `successor`, each
`{node_id, url}` or null, and `carried_forward`, a list of
`{id, text, source_url, status: open|resolved, resolution_url}`.
Those three shapes are unchanged. `source_url`, `resolution_url`, owner
`evidence_url`, and `released_url` share one comment-URL pattern:
`https://github.com/d6g8k5htny-coder/(main|Math-)/(issues|pull)/[1-9]\d*#(issuecomment-[1-9]\d*|pullrequestreview-[1-9]\d*)`.
An owner with `released_url` is not current, so the id and evidence checks do not apply; `released_url` must match that pattern.

`python -B -S tools/check_board_links.py records/node_status` reads the records
and fails, with exit status 1, when:

- a predecessor or successor link is not reciprocated, points to a missing record, or its `url` is not an http(s) URL;
- a current owner lacks an id or a comment URL as its evidence, or a `released_url` does not match that pattern;
- `carried_forward` is not a list, an entry is not an object, an id is duplicated, or a `source_url` or resolution URL does not match that pattern;
- an open predecessor item is missing from the successor's `carried_forward`, or an
  item is marked resolved without a resolution URL.

It checks link structure only. It does not fetch or judge the linked comments.
Tests: `python -B -m pytest tests/test_board_links.py`. The module also runs under
`python -B -S -m unittest tests.test_board_links`.

## Boundaries

- Not a duplicate of #489 (offline board-capacity probe). Capacity output from #489
  is consumed only after #489 merges, and is not used here.
- Not a duplicate of #487 (H02 active-board routing test).
- No records for real boards are added in this PR. Writing them needs owners'
  actual PICKUP URLs, which are not inferred here.
