# `01.1_SADDLE_EXIT_BOUNDARIES` — reading copy

Drive folder id `1cMCkfzm5WeWW3V27LZGh2ubVUypWHFZz`, 2 inventory items: this folder
(tree-only, indexed in the lane-root `_MANIFEST.jsonl`) and one native Google Doc,
stored here as a text export.

**Nothing here is byte-exact and nothing here can be made byte-exact.** The object is a
native Google Doc; `drive/inventory.jsonl` declares no `sha256` for it and no register
tab declares a body digest for it, so the digest in `_MANIFEST.jsonl` was computed here
from this repository's own export and can be compared with nothing the Drive publishes.

| file | Drive id | exactness | bytes stored | inventory bytes |
|---|---|---|---:|---:|
| `00_LEAF_CARD — 01.1 Saddle-Exit Boundaries (CL-NAV-036.2).export.txt` | `12fWLb4T87tb2easBWGjYjUJ4heDs5R0ApXP7vx0IzPs` | `exact: false` — reading copy | 2,639 | 3,044 |

The two byte figures differ because the inventory cell is the Drive-reported size of a
native Doc, which is neither a payload length nor a digest.

## The status banner, verbatim

The card's header line reads "ARTIFACT: CL-NAV-036.2-v1.0 · AUTHOR: Claude (Anthropic),
CL-* instance · CREATED: 2026-07-21 · CLASS: NAV — leaf charter + live-state card ·
CANONICAL IMPACT: NONE · AUTHORITY: none", and it closes "Non-authority header guards
against stale-card citation."

Under the heading the card spells "What must not be claimed" it says: "The endpoint
blocks are local, degree-four-model results with an exact-field transfer criterion —
they are not, alone, the P0.2 theorem. P0.2 remains OPEN."

Its live-state list records a candidate, not a result — "Candidate closure: GP-DER-047's
adaptive certificate (REG_r, s ≤ −C_cap(U), cone/strip/capture chain) subsumes the
endpoint blocks into the full P0.2 candidate — CANDIDATE / NONCANONICAL (CM-044,
CM-047)." — and an identifier hazard: "Identifier collision warning (GP-CLS-001): a
second, mathematically distinct document also carries ID GP-DER-044-v1.0", with the
instruction "Always cite by full title + Drive ID." It also records a counting
correction: "Classification discipline: reports must separate ESCAPE, M-CAPTURE, and
STALL — a T_max-censored stall was previously miscounted as an escape (GP-AUD-012 /
CM-022)."

## What this does not establish

Mirroring is not review, replay, endorsement, promotion or acceptance. Every status word
above is the card's own and none was decided here. The reading copy is not the object.
No object the card points to is held in this repository: neither `GP-DER-044-v1.0` under
either of its two colliding ids, nor `GP-DER-045`, nor `GP-DER-047`, and this port does
not resolve the collision the card warns about. P0.2 is open in
`docs/OPEN_PROBLEMS.md` §D and in `claims/graph.json`, and is exactly as open after this
port as before it.
