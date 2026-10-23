# ctx: agent 1, GB-8a (organizational, NOT A RESULT)

Lane GB-8a per main#475 6369972075 / docs/retro/RETRO-MATH-BIND-01 (#527). credit 0 · OBL OPEN · no flags · sci NONE.

## What was done
- Added 3 per-file v2 records, kind `file`, all pinned at the node_source commit
  `d6g8k5htny-coder/Math-@dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde` (the same form as #525/#526):

| node | path | blob | sha256 | GRAPH.json fingerprint |
|---|---|---|---|---|
| eng.d0-packet-allowlist-fix | frontiers/downstream_gate_20260925/patches/d0_packet_allowlist.patch | f09b248b10d6a8f229bd21bb3a53a97ca16525f2 | 84b7ad72…ff4a | sha256, MATCHES |
| math.rn-selector-region-crosswalk | frontiers/downstream_gate_20260925/SELECTOR_REGION.json | ebffdb0969c7d3484f8fd1f4352ea290346f0318 | 53fa0ec8…390d | semantic token `rn-selector-region-crosswalk-20260925-v1` (not a hash; not comparable) |
| math.d5-pin-neighborhood-first-moment | reviews/d5_reconciliation_20260929/RECONCILIATION.md | 4062d76bdf3c60bc73f04295b88c245bcee395de | 14342b7e…3432 | null |

- Paths are exactly each node's `source` in GRAPH.json@dbe6c0a1; blobs from `git rev-parse dbe6c0a1:<path>`; sha256 over those blob bytes. All three blobs are identical at the frozen Math- cut 9f3fb691.
- The first two files sit inside the already-bound `eng.hard-gate` tree; per GB-5 a file under a tree needs its own record.
- Binding is not review, acceptance or status promotion. `math.d5-pin-neighborhood-first-moment` keeps its GRAPH classification; nothing is promoted by this record.

## Sources consulted
- GitHub Math- history (local clone, objects verified with git). Google Drive not needed: every unit resolved to an in-repo blob at the pin. Dropbox: no access (no connector), not attempted.
- No Codex-stalled PRs were touched or bound.

## Context from prior agent-1 lanes
- GB-COV-01 (6367145303, corrected 6368905569): checker counts trees with matching sha as bound.
- GB-7e (#526): 12 d5 components bound; `math.d5-component.remote-window-proof` deliberately left unbound (shares FC-10 candidate file for `math.rn-fixed-remote-window`).
- Box test env: `pytest tests` has 10 environmental failures in tests/casefold/test_source_isolation_casefold.py on clean main; unrelated.
