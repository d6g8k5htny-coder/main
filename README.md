# q0 Research Program — git home

Git-side home of a multi-model mathematics research program that until now
lived entirely in a Google Drive shared drive. This repository carries the
program's **architecture** — its registers, governance protocols, package
structure, quarantine discipline and content-addressed source map — plus the
machine-readable inventory of all 4,456 Drive items, so that the same work can
be run, reviewed and verified with ordinary git and CI.

> **Status discipline.** Nothing here promotes, closes or reclassifies any
> mathematical claim. Every status label is carried verbatim from the source
> registers as exported on **2026-09-17**. Packaging is not premise discharge;
> a display is not a certified enclosure; a draft is not an operator promotion;
> registration is not review.

## What the research is

Two independent tracks plus a platform.

**1. The q0 / SIDE24 spine (main track).** For a periodized Gaussian field on a
side-24 torus, let `q(r)` be the typed pair-Palm probability that a marked
six-pin configuration at separation `r` behaves correctly. The program's central
object is the failure rate `1 − q(r)` and its cubic law.

| Object | Statement | Status as written |
|---|---|---|
| Q0-C101 qualitative rate | `0 ≤ 1 − q(r, 6/5) ≤ C_Q0 · r³` for `0 < r ≤ 0.025` (as the Canon writes it), `q → 1`; no typing division, no decimals | live-root theorem in the frozen Q0 core; "No numerical value of C_Q0 is certified" |
| Theorem B (Q0-C104) | `ν_B(ℓ) = C*·ℓ^(−1/3)(1 + o(1))`, conditional-MS form | **unconditional `PROVEN-HERE` RETRACTED** (GP-AUD-187, 2026-07-24: "THEOREM NOT KILLED / NO PROMOTION"); candidate; exact Jacobian proved; conditional B0 proved under A1–A5; seven repair gates TB-G1…G7 open; no numerical `C*` |
| Theorem D1 v2.2(1) | certified rung at `r = 0.05` | certified under named hypotheses H5-RIM / H5-AXIS and D3-LEMMA-RN-UNIF at the rung |
| Theorem D1 v2.2(2) | all-small-r `1 − q(r, 6/5) ≤ C r³` | **CONDITIONAL** on five named validity premises (see below) |
| SIDE24 3D (AO48-OPR-045) | compact-positive-mark 3D upper / lifetime result | ratified on the 3D track only |
| LPW qualitative | `∃ c, r₀ > 0 : 1 − q ≥ c r³` (2D six-pin typed pair-Palm) | accepted at review scope; not a two-sided law |

The Canon's four-file core is `Q0_MASTER.md`, `Q0_LEDGER.md`, `q0_machine.json`,
`q0_verify.py`; canonical promotion enters only through `Q0_LEDGER.md` by the
owner's decision.

**Standing firewall:** the 2D upper track and the 3D lifetime track must never be
composed into a two-sided law. `ERRATA_AND_CLARIFICATIONS_2026-09-13` withdraws
exactly that composition.

**2. The prize reconnaissance track (independent).** Restricted-scope work on
discrete Talagrand / convexity (PR-TAL-001…019), Erdős problems 3, 39, 142 and
169, Riemann and Collatz probes, plus the P14/P15 structural-localization
results. Every carrier is author-side, external review pending, historical
novelty unestablished. **Original prize problems solved: 0.** This track is
HOLD / not for submission and must not be merged into the q0 packages.

**3. The research platform.** Coupled registers, evidence-gated governance,
verifiers, regression corpora, a Lean/formal system lane, and the accessibility
source map. This is the part that generalizes, and it is what this repository
makes reusable.

## The five open validity premises

These block unconditional promotion of Theorem D1 v2.2(2). The frozen v2.2 body
still lists all five; the register note and the v2.3 DRAFT propose reductions
that take effect only at the next issuance. **Both layers are recorded; they are
not collapsed.**

| # | Premise | Frozen v2.2 / HOLD | Register note + addenda (effective at v2.3) |
|---|---|---|---|
| 1 | `OBL-D1-PROMOTE` | OPEN (chart + sub-obligations) | normalizer sub-part **DISCHARGED** by the H3 band floor; **chart side still OPEN** |
| 2 | `D3-LEMMA-RN-UNIF` | NOT closed | still OPEN; receipts carry `lemma_closed: false` |
| 3 | `PERC-DECAY` | OPEN | **RESTATED** as Θ(r³); engine complete and frozen; `PD-CONN` named input still open |
| 4 | `OBL-B1-BRANCH(loop\|B1)` | OPEN | demoted to **REFINEMENT** for v2.3 |
| 5 | `B4.loc` dam-line tube certificate | OPEN (asserted-not-established identification) | **CLOSED 2026-09-15** (B4LOC-R1); wrap/remote reconciliation still open |

Chart-side sub-obligations of premise 1: `OBL-H5-JETMOD` (certified 24-jet band
enclosure — OPEN, display only), `OBL-H5-ZBAND` (lo side rides the frozen H3
floor; **hi side OPEN**), `OBL-H5-REMOTE-THRESHOLD` (OPEN, rides premise 2).

## Repository layout

```
governance/     operator protocols (reading copies — see PROVENANCE.json) + git mapping
registers/      the 42 coupled register tabs as JSON and CSV, plus the source export
drive/          complete source map: inventory.jsonl (4,456 items) + accessibility CSVs
claims/         the machine-checked claim graph and its firewalls
engine/         the active layer: lanes, bound carriers, the recovered RN engine, runner
research/       the mathematics: certified intervals and the per-lane drivers
reviews/        R17 §4 nonauthor technical reviews, all at zero independence credit
recovery/       recovered accessibility exceptions, with provenance and what is missing
packages/       PKG-01..05 peer-review submission packages and the HOLD sibling
quarantine/     non-authoritative material and the logical-exclusion list
sandbox/        drafts, no authority
legacy/         zero evidentiary authority / inspiration only
tools/          the checkers CI runs
tests/          CI-enforced invariants and negative controls
docs/           research map, open problems, contribution plan, findings, ported reports
```

**Of the seven artifacts under `governance/` and `docs/` that mirror a Drive
object, three are byte-identical to it and four are not.** The three are
proved by a full SHA-256 match to a digest the corpus declares; the four have
no payload digest anywhere in the corpus, so their exactness is unverifiable —
and two of them have known content divergences. Until 2026-09-18 the count was
zero: `OP-PROT-019-v1.1_R17.md` had the same byte count as the object the
register names and a different digest, so a byte-count check confirmed the
wrong bytes. `governance/PROVENANCE.json` records every digest and outcome and
`tools/provenance_check.py` refuses to let any file call a non-exact copy
verbatim.

Start with [`docs/RESEARCH_MAP.md`](docs/RESEARCH_MAP.md), then
[`docs/OPEN_PROBLEMS.md`](docs/OPEN_PROBLEMS.md), then
[`governance/GIT_ADAPTATION.md`](governance/GIT_ADAPTATION.md). What the
active layer has found so far, defects in this repository's own work first, is
in [`docs/FINDINGS_2026-09-18.md`](docs/FINDINGS_2026-09-18.md).

## What CI enforces

```bash
python3 tools/registers_import.py --check   # registers still match the source export
python3 tools/registers_check.py            # structural invariants, modulo documented findings
python3 tools/provenance_check.py           # no reading copy is presented as the object
python3 tools/claims_check.py               # claim-graph: no claim rests on an open premise
python3 tools/quarantine_check.py           # no excluded payload appears in any manifest
python3 tools/verify_manifests.py           # every SHA-256 / byte count in every manifest
python3 tools/reviews_check.py              # zero independence credit, no gate moved
python3 tools/recovery_check.py             # candidates never stored as recoveries
python3 tools/collision_proposal_check.py   # additive only, exported registers untouched
python3 tools/lanes_check.py                # no lane's status is stronger than the claim graph's
python3 tools/carriers_verify.py            # every bound carrier blob matches its manifest digest
python3 tools/receipts_check.py             # receipts: schema, append-only, none claims a status change
python3 tools/slack_check.py                # bound-slack registry bookkeeping; utility never correctness
python3 tools/frozen_check.py               # frozen-object register vs source-map digests, offline; classes B/C not comparable
python3 -m pytest -q                        # negative controls throughout
```

Every checker prints a one-line summary and exits nonzero on failure. Every
claimed bound and every checker has tests that fail when it is weakened —
those negative controls have found real defects in this repository's own
verifiers three times, which is what they are for.

## Provenance

| | |
|---|---|
| Drive items covered | 4,456 (3,714 files, 742 folders), 319 MB |
| Archive carriers / members | 77 carriers, 11,649 member occurrences, 4,020 distinct payloads |
| Source-map snapshot | 2026-09-17 accessibility publication |
| Register export | GP-REG-032-v1.2, 42 tabs, exported 2026-09-17 |
| Control plane at export | AI-DRIVE-AUTONOMY-R17 (OP-PROT-019-v1.1) |
| Owner / final authority | Dylan Roy |

The owner remains the single final authority for canonical promotion, external
release, permanent deletion and machine-root replacement. This repository
changes none of that.
