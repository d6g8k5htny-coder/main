# `engine/` — the work layer

This directory is where the repository schedules and runs work. It holds three
kinds of thing, and the difference between them is the whole point:

| | what it is | what it is worth as evidence |
|---|---|---|
| **lanes** (`lanes/*.json`) | work items: one per open-problem section, carrying the source's own next exact action, falsifier and caveats | **none** |
| **receipts** | records that a computation ran, with its inputs, outputs and hashes | **none** |
| **carriers** (`carriers/`) | the source code and data a lane consumes, content-addressed | provenance only |

> **Neither a lane nor a receipt is evidence.** A lane is a note about what the
> sources say should be done next. A receipt is a note about what a program did.
> Neither is a proof, a certificate, a review verdict or a status. **A green run
> is a run, not a proof.**
>
> **The one thing that can change a mathematical status is an operator decision
> under the protocols in [`governance/`](../governance/)** — OP-PROT-019/R17 for
> entry, claims and review, OP-PROT-012 for the autonomy classes, OP-GDN-002 for
> the coupled-advancement invariant and its required transition record,
> OP-CNS-001 for preservation and closure discipline. The owner (Dylan Roy)
> remains the single final authority. Nothing in this directory changes that, and
> nothing in this directory is a substitute for it.

Not a status, and never to be read as one: a lane's `repo_state`, a passing
`pytest`, a passing checker, a receipt, a registration, a session CLOSE, a smoke
test, a display, a Monte Carlo estimate, a fitted exponent, or a
high-precision floating-point number. Where a computation in this repository
uses floats it must be labelled **NON-CERTIFYING** in the code, the docstring and
the receipt. Where a bound is claimed, the arithmetic is exact
(`fractions.Fraction`).

## What is here

```
lanes/<KEY>.json      one work item per section of docs/OPEN_PROBLEMS.md
next_action.py        the dispatcher: ranks the lanes and says why
carriers/             carrier source and data (owned and documented elsewhere)
rn_engine/            mirrored engine sources (owned and documented elsewhere)
```

`carriers/` and `rn_engine/` are not described by this file beyond their names.
Read their own documentation; where they have none, this README makes no claim
about their contents, and in particular makes no claim that anything mirrored
there is byte-identical to its Drive object. (`governance/README.md` records a
case where a mirrored reading copy had the declared byte count and a different
SHA-256.)

The checker for this layer is [`tools/lanes_check.py`](../tools/lanes_check.py);
its negative controls are [`tests/test_lanes.py`](../tests/test_lanes.py).

## Using the dispatcher

```bash
python3 engine/next_action.py              # ranked list, with the ranking rule
python3 engine/next_action.py --lane A5    # one lane in full
python3 engine/next_action.py --json       # machine consumption
python3 tools/lanes_check.py               # the invariants; non-zero on failure
```

The ranking is by claims blocked (from the claim graph's transitive structure),
then whether the source names a falsifier, then whether the lane's inputs are
bound in this repository, then `repo_state`, then lane key. The program prints
that rule, and prints that **a ranking is a work-scheduling heuristic carrying no
mathematical authority whatsoever**, on every run. Disagreeing with the ranking
costs nothing: it reorders work, it does not reorder mathematics.

## The lane schema

Every field that restates a source is transcribed from it, and quotes the
source's own words where `docs/OPEN_PROBLEMS.md` quotes them.

| field | meaning |
|---|---|
| `key`, `title` | section key and title, from the document's heading |
| `source_section` | document, exact heading line, level, parent heading |
| `blocks` | ids in `claims/graph.json` this lane directly blocks; the dispatcher computes the transitive claim set |
| `status` | **verbatim from the source**, or `null` with `status_absent_reason` when the source assigns none. Never decided here |
| `status_source` | where the word came from, quoted, including which layer |
| `next_exact_action` | the source's own "next exact action" language |
| `falsifier` | the source's own falsifier, or `null` where it names none |
| `inputs` | `carrier_id`s from `engine/carriers/MANIFEST.json`; `[]` with `inputs_note` while that manifest does not exist |
| `not_a_substitute` | the source's own caveats, quoted with their source |
| `repo_state` | `none` / `scaffolded` / `partial` / `running` — **code in this repository only, not a mathematical status** |
| `artifacts` | repository paths that work this lane |
| `does_not_establish` | what this lane does not establish. Required on every lane |

Two layers are recorded separately in `claims/graph.json` for the D1 premises —
the frozen v2.2 body and the register note plus addenda — and they are **not
collapsed**. A lane carries the weaker of them and says so in `status_source`.

## What the checker enforces

`tools/lanes_check.py` exits non-zero when:

1. the lane set and the document's section set disagree in either direction. The
   section list is **parsed from the document's headings**, not hardcoded: a
   level-2 section with level-3 subsections is a container represented by those
   subsections, and every other section is a lane. Add a section to
   `docs/OPEN_PROBLEMS.md` and the checker demands a lane for it;
2. a `blocks` id is not in `claims/graph.json`;
3. an `inputs` carrier id is not in the carrier manifest (or inputs are declared
   while no manifest exists);
4. a `status` is not a value the registers, or the claim graph that transcribes
   the same sources, actually use;
5. **a lane declares a status stronger than the same object's status in
   `claims/graph.json`.** This is the firewall. A work ledger is an easy place to
   promote something quietly — a lane reading `CLOSED` where the frozen layer
   reads `OPEN` would launder a status change through a file nobody reviews as
   mathematics. The checker refuses it by name: `PROMOTION REFUSED`;
6. `repo_state` is missing, out of vocabulary, used as a status, nested anywhere
   inside a lane, or present in any structure that carries evidentiary meaning
   (`claims/graph.json`, `registers/json/`, `reviews/records/`,
   `quarantine/EXCLUSIONS.json`, `governance/PROVENANCE.json`);
7. lane D's 24 sub-items stop matching `registers/json/review_queue.json`
   verbatim.

`tests/test_lanes.py` runs the checker and then breaks a **copy** of the ledger
in eighteen ways — promoting a lane above its claim-graph status, collapsing the
two D1 layers, inventing a `blocks` id, deleting a documented lane, adding an
undocumented one, editing a review route's transcribed status, using a
`repo_state` value as a status — and asserts each one is rejected. It also
asserts that declaring every lane `running` changes no status, no blocked-claim
set and no verdict.

## Adding or changing a lane

* Change `docs/OPEN_PROBLEMS.md` first if the work item itself changed; the lane
  follows the document, never the other way round.
* Quote the source. If the source names no status, write `null` and say why in
  `status_absent_reason` rather than choosing one.
* If you believe a status should change, that belief goes to the operator under
  `governance/`. It does not go in a lane file, and no amount of green CI makes
  it go there.
* Never edit `registers/source/`, `registers/json/`, `registers/csv/`,
  `drive/inventory.jsonl` or `drive/source_map/`: that is exported source data
  and CI fails if it drifts. Lane F exists because defects in that export are
  **reported, not repaired**.

## Reusable operations and trials (`operations/`)

The 2026-09-18 register export added a `reusable_operations` tab (fifteen
operations OP01–OP15, every one `Utility: UNMEASURED`, `Novelty: NOT_ASSESSED`)
and an empty `operation_trials` ledger. [`operations/`](operations/README.md)
transcribes the fifteen cell for cell into `REGISTRY.json`, marks — separately,
under `git_side` — the four whose `Action / output` cell displays an identity
exact arithmetic can reproduce from the cell alone (OP02's power ledger, OP03's
pushforward Jacobian, OP04's Hölder exponents, OP05's rational witness), runs
those over `fractions.Fraction` with `operations/trial.py`, and records each
run in the ledger's own eighteen-column shape under `operations/trials/`,
append-only. **A trial is a record of a computation, not evidence**: it
reproduces displayed algebra and says nothing about premises, scope, sources,
usefulness or novelty, and it moves no status. `tools/operations_check.py`
enforces that the registry is the register's words (never stronger on Utility
or Novelty, in a cell or in any sentence the repository wrote), that every
trial is in the ledger's shape, names a readable catalog version, carries the
register's `Do not infer`, Utility and Novelty cells verbatim, uses no word
from the status-word list and no usefulness word without the register's
`UNMEASURED` / `NOT_ASSESSED` (word lists, not a reading of the sentence), has
`Split`, `Arm` and every cost column agreeing with its verdict, has its
recorded arithmetic re-run here with the same steps field for field, is
`NOT_RUN` for a non-checkable operation and never for a checkable one, carries
the runner's fixed `does_not_establish` sentences verbatim (the load-bearing
field is pinned, not free), is dated by the runner's clock — never later than
the checker's clock nor, once committed, later than the commit that added it —
and is unchanged since `git HEAD`; the registry's own authority and
`does_not_establish` sentences and its provenance blocks are pinned the same
way. `tests/test_operations.py` holds the negative controls, including the
mutations three adversarial rounds found slipping past earlier versions of the
checker (the third found four records stamped with a supplied time seventy-eight
minutes after they were written; they were withdrawn and re-run from the clock).

## What this directory does not establish

Nothing mathematical. `engine/` schedules work and records that programs ran. It
proves no theorem, certifies no bound, discharges no premise, closes no
obligation, performs no review and grants no independence credit. It does not
compose the 2D upper or lower tracks with the 3D lifetime track — that
composition is withdrawn by `ERRATA_AND_CLARIFICATIONS_2026-09-13` and forbidden
by the `FW-2D-3D-COMPOSITION` firewall in `claims/graph.json`. No original prize
problem is solved anywhere in this repository, and nothing here should be read as
implying otherwise.
