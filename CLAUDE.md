# Working in this repository

This is the git home of the q0 / SIDE24 mathematics research program. Its value is not the
code — it is that **every status label in it is honest**. A repository of research claims that
quietly inflates one of them is worth less than no repository at all.

Read [`README.md`](README.md), then [`docs/RESEARCH_MAP.md`](docs/RESEARCH_MAP.md), then
[`docs/OPEN_PROBLEMS.md`](docs/OPEN_PROBLEMS.md), then
[`governance/GIT_ADAPTATION.md`](governance/GIT_ADAPTATION.md).

## The rules that are not negotiable

1. **Never promote, close, discharge or reclassify a claim, premise or obligation.** Status
   labels are transcribed from the source registers, never decided here. The five validity
   premises of Theorem D1 v2.2(2) are OPEN. `D3-LEMMA-RN-UNIF` is not closed. The exact
   licensing predicate, and only that, moves a gate — and only an operator applies it.

2. **State what your work does not establish.** Every module, report, receipt and review record
   carries this explicitly. It is not a disclaimer; it is the most load-bearing field.

3. **High precision is not certification.** An mpmath computation at 200 digits, a Monte Carlo
   estimate, a fitted exponent, a dense sampling, a display, a probe cell, a candidate constant,
   a smoke test, a session CLOSE and a registration are each *not* a certificate. The sources
   say so themselves, repeatedly. Where you compute in floats, label the path NON-CERTIFYING in
   the code and in any output. Certified bounds go through `research/interval/`, whose contract
   is that containment is unconditional and tightness is best-effort.

4. **Never compose the 2D upper/lower tracks with the 3D lifetime track.**
   `ERRATA_AND_CLARIFICATIONS_2026-09-13` withdraws exactly that composition, and
   `tools/claims_check.py` fails the build on it.

5. **No original prize problem is solved.** Every prize claim carries
   `original_prize_closed: false`. The prize reconnaissance track is HOLD, not for submission,
   and must not enter the q0 dependency graph in either direction.

6. **Organizational independence cannot be manufactured.** A same-provider reviewer earns zero
   independence credit. Record the technical verdict and the zero credit separately, and say
   that the independence-requiring gate remains open. Aging never approves anything.

7. **Never edit exported source data.** `registers/source/`, `registers/json/`,
   `registers/csv/`, `drive/inventory.jsonl` and `drive/source_map/` are exports.
   `tools/registers_import.py --check` regenerates them and fails CI on any drift. Found a
   defect in the source? Record it in `registers/KNOWN_FINDINGS.json` and propose a repair —
   do not apply one.

8. **Never edit a frozen body in place.** Numbered successors only, and no permanent deletion.

9. **Never open `99_DO_NOT_OPEN`** (Drive id `1VTiBaRlBvHGptiXqoEli4E5630nO7mNb`) for
   authority, proofs, certificates or "latest" status unless the operator names a vault id for
   forensic recovery. Metadata only.

10. **Nothing under `legacy/`, `quarantine/` or the vault may be cited as evidence.** Drive
    paths beginning `02_LEGACY_Q0_ARCHIVE` carry zero evidentiary authority.

Dylan Roy is the single final authority for canonical promotion, external release, permanent
deletion and machine-root replacement. This repository changes none of that.

## Layout

| path | what it holds |
|---|---|
| `governance/` | operator protocols and how each Drive construct maps onto git |
| `registers/` | the 42 coupled register tabs as JSON and CSV, plus the source export |
| `drive/` | the source map: `inventory.jsonl` (4,456 items) and the accessibility CSVs; `deltas/` (dated, byte-exact copies of what changed on the Drive after the snapshot, with `PATH_CHANGES.jsonl`); `mirrors/` (byte-exact copies of selected lane objects, one `_MANIFEST.jsonl` per directory, README quoting the source banners) |
| `claims/` | the machine-checked claim graph and its firewalls |
| `engine/` | the active layer: lanes, bound carriers, the runner, receipts; `bridge/` holds the work-order and run-receipt schemas of the proposed (not deployed) Drive–GitHub execution contract and nothing that enforces it |
| `research/` | the mathematics: certified intervals and the per-lane drivers |
| `reviews/` | nonauthor technical review records, all at zero independence credit |
| `recovery/` | recovered accessibility exceptions, with provenance |
| `quarantine/` | non-authoritative material and the logical-exclusion list |
| `packages/`, `sandbox/`, `legacy/` | submission packages, drafts, zero-authority material |
| `tools/` | the checkers CI runs |
| `tests/` | CI-enforced invariants and negative controls |

## Engineering conventions

- Python 3.11, **standard library only**. `pytest` is the single test dependency. `mpmath` and
  `numpy` are not guaranteed present: guard the import and `pytest.skip` when absent.
- Exact rational arithmetic (`fractions.Fraction`) wherever a bound is claimed. Never float.
- Certified enclosures via `research/interval/`. Read its public API; do not reimplement it.
- House style: see `research/rn/moment_envelope.py` and `tools/claims_check.py`. Checkers print
  a one-line summary and exit nonzero on failure.
- **Negative controls are the deliverable, not decoration.** Every checker and every claimed
  bound needs a test that fails when the check is weakened or an inequality flipped. Writing
  them found a real bug in `tools/claims_check.py` itself: a default argument bound the graph
  path at import time, so every mutation test was silently re-checking the good graph. Resolve
  module globals at call time and invoke checkers through their CLI flags in tests.

## Before you commit

```bash
python3 tools/registers_import.py --check   # exported registers still match the source export
python3 tools/registers_check.py            # structural invariants, modulo documented findings
python3 tools/claims_check.py               # the claim-graph firewalls
python3 tools/quarantine_check.py           # logical quarantine is enforced
python3 tools/verify_manifests.py           # every SHA-256 and byte count
python3 -m pytest -q                        # unit tests and negative controls
```

`tools/` also holds `carriers_verify.py`, `lanes_check.py`, `receipts_check.py`,
`reviews_check.py`, `recovery_check.py`, `collision_proposal_check.py`, `slack_check.py` and
`frozen_check.py` and `bridge_check.py` as the active layer lands them; CI runs whichever exist.

## A note on what "active" means here

`engine/` lets the repository run the program's own computations and record what happened. A
green run is a run. A receipt is a record of a computation, not evidence. Neither a receipt, a
passing test, a reproduced bound nor a bound carrier may raise any claim's grade —
`tools/claims_check.py` enforces that as a firewall, not a convention. The ranking that
`engine/next_action.py` prints is a work-scheduling heuristic and carries no mathematical
authority whatsoever.
