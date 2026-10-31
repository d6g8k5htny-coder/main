# Retroactive source resolution for the 126 "no source" ledger units

**Object:** RETRO-SOURCE-RESOLUTION-01CZ-20261031. Credit 0 · scientific effect NONE · OBL OPEN.
Data: [`RETRO_SOURCE_RESOLUTION_01CZ_20261031.json`](RETRO_SOURCE_RESOLUTION_01CZ_20261031.json).
Resolver: [`retro_source_resolve_01cz.py`](retro_source_resolve_01cz.py). It is read-only and
standard-library only, and its sha256 is recorded in the JSON.

The math coverage ledger (`data/hypergraph/math_coverage_ledger_20261023.md`; hypergraph at main
`4712fc13`) marks 126 of its 346 units `has_source = false`. This pass asks two narrower
questions of each of those units, at Math- `19a55750` and main `4712fc13`:

1. Does the unit itself contain mathematical text?
2. Does every source the unit declares, by path and Git blob, resolve to those exact bytes?

## Results

| | Units |
|---|---:|
| Contain in-unit mathematical text (`.md`/`.tex` other than README, or a single `.md`/`.lean` file) | **105** |
| Contain no mathematical text: code, config, evidence folders, experiments | 20 |
| Unit absent at Math- main | 1 |

The one absent unit is `reviews/soft_fold_protocol_20261006`. It exists only on the unmerged
branches `cbz/lm006-*-20261029`.

Of the 520 declared `{path, git_blob}` references in those units:

| Resolution | References |
|---|---:|
| Exact blob at the declared path on main | 484 |
| Exact blob in that path's history, on some fetched branch | 29 |
| **Unresolved** | **7** |

So `has_source = false` does not mean "no mathematical source" for most of these units. For 105
of them, the source is in the unit, typically as `NOTE.md`, `LOCAL_PAIRING.md` or
`RECONCILIATION.md`. The ledger's source classifier seems to look for proof-named files. Whether
to change the column is the ledger owner's decision; this pass does not edit it.

## The seven unresolved references

**Six in `frontiers/window_multiplicity_laws_20260928`, on the elder-rule chain.** That unit
holds `ELDER_LOWER_AND_DENSITY_GAP.md`, the source the Lifetime spec cites for the E4 elder
lower bound. Its `PUBLICATION.json` records:

- `"code_publication": {"state": "BLOCKED_UNRESOLVED", …}` for `algebra.py`: 6,163 bytes,
  sha256 `16ed713d…3350`, Git blob `00140088`. The publishing tool refused the upload, and the
  file was never committed.
- Five evidence files, also never committed: `FINITE_REPLAY_SUMMARY.json` (`1b5c9e41`),
  `PACKET_VALIDATION.json` (`2807b887`), `PACKET_NEGATIVE_CONTROLS.json` (`81116b1b`),
  `RED.stderr` (`ee5187f3`) and `ELDER_RED.stderr` (`21dc9450`).
- `"reproducible_computation_from_this_github_packet": false`, which the packet already says.

I searched for these bytes and found none:

- `git log --all --find-object` over every fetched Math- branch, for `00140088` and `1b5c9e41`;
- a Google Drive title search for `algebra.py`;
- two Dropbox searches, `algebra.py` and `window_multiplicity_laws algebra.py`. All hits were
  unrelated files, dated before the packet's 28 September date.

**The elder lower-bound packet's algebra check is therefore not reproducible from any store this
session can read.** Its original author, an OpenAI lane, may still hold it. Recovering it, or
re-deriving the check from `ELDER_LOWER_AND_DENSITY_GAP.md` as a clearly labelled successor, is
open work.

**One in `imports/hardening_ebedb780`:** `reviews/records/REV-P15-B.md` at blob `0579a10c`
resolves neither on main nor in that path's history.

## What this does not establish

- It establishes identity only: correctness, review, acceptance and status are all untouched.
  A resolved blob means the bytes are the declared bytes, not that they are right.
- The resolver reads only JSON references that carry a 40-hex `git_blob`. References by sha256
  alone, by prose, or by an abbreviated blob are not checked.
- It searched the history of the declared path only. A blob that moved to another path counts
  as unresolved.
- It does not edit the ledger, the hypergraph or any exported register.
- It is Anthropic/Claude work (session 01CZ) at zero organizational-independence credit.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
