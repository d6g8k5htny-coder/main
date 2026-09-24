# q0 Research Program — git home

Current retrieval: [Drive search navigator](https://docs.google.com/spreadsheets/d/1x9aKuv9pQGD6kwwa3ZmehSyEnVX2ZlhhcY2OehjMCqI/edit)
and [keyword/SHA guide](docs/DRIVE_SEARCH_GUIDE.md). The active research packages
have 2,944 verified stored files/exports; held subtrees remain metadata-only.
The [interval-family candidate](docs/RN_INTERVAL_FAMILIES.md) extends exact
Gaussian moment bounds without promoting any scientific status.
The [SIDE24 point-law candidate](docs/RN_SIDE24.md) connects the normalized
field kernel and nine-pin interval conditioning to those moment witnesses at
one exact spatial point, over the complete mark interval.
The [density/window extension](docs/RN_SIDE24_DENSITY.md) completes the
pointwise RN integrand bound under an explicitly imported H3 floor.
The [spatial extension](docs/RN_SIDE24_CELL.md) gives a complete local rectangle
integral upper below `6e-12`. The [parallel mathematical replays](docs/PARALLEL_MATH.md)
reconstruct that fixed-r H3 floor, improve the LPW covariance modulus and bound
one H5 jet over an exact radius band. The full near-annulus bound remains open.

The [H3 radius-band and RN N6 continuation](docs/H3_RN_N6.md) proves fixed-axis
`1.7 r² <= Z(r) <= 3.49 r²` for every `0 < r <= 1/20` and improves four
local RN squares, with full-annulus and all-angle obligations retained.

The [finite-width RN inner wedge](docs/RN_INNER_WEDGE.md) adds a complete
scoped cover at `r=1/20`: twelve auxiliary strips, zero pending cells, and
an exact area-weighted integral below `33/256000000000`, conditional on the
imported fixed-radius H3 floor. The remaining annulus and all-radius problem
stay open.

The [twelve-project mathematical continuation](docs/TWELVE_PROJECT_MATH.md)
adds uniform six-pin covariance bounds, a conditional LPW coefficient,
two gradient-jet bands, and exact persistence/contact, weighted-event,
Lambda, Gaussian-tube and integer-count lemmas. Each result retains its
stated hypotheses. In particular, elder pairing and selected-branch
adjacency are different events; the LPW coefficient does not identify them.

[Reproducible execution](docs/RESEARCH_EXECUTION.md) adds one verification
command, exact RN certificate replay and derived dependency/diagnostic
checkpoints. These retain the existing source and scientific-status boundaries.
[Bounded recurring continuation](docs/RESEARCH_AUTOMATION.md) advances one
verified delivery at a time and coordinates through the live register.

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

## Navigation

Reading routes on this hardening checkout. The status tables above stay in
force. These links do not invent a source of truth and do not move a claim flag.

| I want to… | Go here |
|---|---|
| See where this branch sits relative to `main`, Drive, and Math- | [Workspace](docs/WORKSPACE.md) |
| Follow a topic without treating a link as acceptance | [Research index](docs/RESEARCH_INDEX.md) |
| Replay a check or an external Math- candidate | [Reproduction](docs/REPRODUCE.md) |
| Stay on the program map and open problems | [Research map](docs/RESEARCH_MAP.md), [open problems](docs/OPEN_PROBLEMS.md) |

`lemma_closed`, `prizes_solved`, `discharges_OBL_H5_JETMOD`, and
`certified_C_H` stay false. SIDE24 carriers marked ABSENT stay ABSENT.
Quarantine is not a source of truth. `inventable_attempt_accepted` stays
false. A navigation check is engineering hygiene, not discharge. This page
lives on `chatgpt/drive-github-hardening-20260919`, not on `main`.

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
| SIDE24 3D (AO48-OPR-045) | for the normalized periodized Bargmann–Fock field on the side-24 three-torus, uniformly on compact (b,κ) subsets: `sup(1 − p_r) ≤ C r³`, and with it `ν₃,₂₄(ℓ) = c₃,₂₄ ℓ^(−1/3)(1 + o(1))` with a closed-form constant | register status **RATIFIED-AT-STATED-SCOPE** (operator_decisions row AO48-OPR-045, 2026-08-02; a record authored by the AO48 line relaying the operator's one-line sign-off, with carried dependencies and reopening conditions stated); 3D track only, never composed with the 2D tracks |
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

**3. The research platform.** The coupled registers (exported and checked here),
the evidence-gated governance protocols, the accessibility source map, and the
Drive's platform lane: metrics and defect interception, the Fresh Start 2.0
"LEAN VERIFIED RESEARCH SYSTEM" (eight Core capsules of Markdown statements,
sympy hostile tests and PDFs — it holds no Lean sources), and two sandbox
verification prototypes at "LIVE MIGRATION HOLD". This repository carries the
registers, the source map and its own checkers; the lane's verifiers, control
plane and capsule statuses are indexed, not carried, run or reused.

## The five open validity premises

These block unconditional promotion of Theorem D1 v2.2(2). The frozen v2.2 body
still lists all five; the register note and the v2.3 DRAFT propose reductions
that take effect only at the next issuance. **Both layers are recorded; they are
not collapsed.**

| # | Premise | Frozen v2.2 / HOLD | Register note + addenda (effective at v2.3) |
|---|---|---|---|
| 1 | `OBL-D1-PROMOTE` | OPEN (chart + sub-obligations) | normalizer sub-part **DISCHARGED** by the H3 band floor; **chart side still OPEN**, "and now also the uniform-band extension of the B4LOC/B2-far/far-route certificates" (register note §5: the obligation is reduced on one side and enlarged on the other) |
| 2 | `D3-LEMMA-RN-UNIF` | NOT closed | still OPEN; receipts carry `lemma_closed: false` |
| 3 | `PERC-DECAY` | OPEN | listed **CLOSED "in its RESTATED form"** (Θ(r³) with certified constants; register note §5); `PD-CONN` moved to the note's **REFINEMENT/constants register** ("named; constants-not-order"); engine complete and frozen |
| 4 | `OBL-B1-BRANCH(loop\|B1)` | OPEN | demoted to **REFINEMENT** for v2.3 |
| 5 | `B4.loc` dam-line tube certificate | OPEN (asserted-not-established identification) | **CLOSED 2026-09-15** (B4LOC-R1); wrap/remote reconciliation **ADJUDICATED YES** — "B4.rem is CLOSED by B4LOC-R1 for the O(r³) validity grade" (register note §2, §5) |

Chart-side sub-obligations of premise 1: `OBL-H5-JETMOD` (certified 24-jet band
enclosure — OPEN, display only), `OBL-H5-ZBAND` (lo side rides the frozen H3
floor; **hi side OPEN**), `OBL-H5-REMOTE-THRESHOLD` (OPEN, rides premise 2).

## Repository layout

```
governance/     operator protocols (reading copies — see PROVENANCE.json) + git mapping
registers/      the 44 coupled register tabs as JSON and CSV, plus the source exports
drive/          complete source map: inventory.jsonl (4,456 items) + accessibility CSVs;
                deltas/ (what changed after the snapshot, byte-exact, dated) and
                mirrors/ (byte-exact copies of selected lane objects, manifest-verified)
claims/         the machine-checked claim graph and its firewalls
engine/         the active layer: lanes, bound carriers, the recovered RN engine, runner;
                bridge/ (work-order and run-receipt schemas of the PROPOSED, NOT DEPLOYED
                Drive–GitHub execution contract; holds no orders and no receipts)
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
python3 tools/vault_hygiene_check.py        # vault listing matches inventory; vault ids stay unstored
python3 tools/verify_manifests.py           # every SHA-256 / byte count in every manifest
python3 tools/reviews_check.py              # zero independence credit, no gate moved
python3 tools/recovery_check.py             # candidates never stored as recoveries
python3 tools/collision_proposal_check.py   # additive only, exported registers untouched
python3 tools/lanes_check.py                # no lane's status is stronger than the claim graph's
python3 tools/carriers_verify.py            # every bound carrier blob matches its manifest digest
python3 tools/receipts_check.py             # receipts: schema, append-only, none claims a status change
python3 tools/slack_check.py                # bound-slack registry bookkeeping; utility never correctness
python3 tools/frozen_check.py               # frozen-object register vs source-map digests, offline; classes B/C not comparable
python3 tools/bridge_check.py               # execution-bridge orders/receipts: schema, digests, git freeze; no enforcement
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
| Archive carriers / members | 77 carriers — recomputable from the committed files: the 76 inventory rows `ARCHIVE_INDEXED` plus the `BINARY_UNRENDERED` `S2-DATA-002-v1.0_result_carrier.zip`, which `Archive_Members.csv` lists members for (72 of the 76 have member rows; the other four are single-file `.gz` uploads); 11,649 member occurrences; 2,974 distinct member payload digests (the five `READ_FAILED` members carry an empty digest cell; until 2026-09-19 that cell was counted as a 2,975th); 4,020 distinct payloads over the whole source map |
| Source-map snapshot | 2026-09-17 accessibility publication |
| Register export | GP-REG-032-v1.2, 44 tabs, xlsx export of 2026-09-18 (the 2026-09-17 markdown rendering is retained beside it; it truncated seven tabs) |
| Control plane at export | AI-DRIVE-AUTONOMY-R17 (OP-PROT-019-v1.1) |
| Owner / final authority | Dylan Roy |

The owner remains the single final authority for canonical promotion, external
release, permanent deletion and machine-root replacement. This repository
changes none of that.

### 2026-09-19 research-transfer continuation

The [2026-09-19 coverage bundle](drive/deltas/2026-09-19/DG-MIGRATION-20260919/README.md)
accounts for 4,112 stored research files, 758 folder records and 54 held
files. Exact inventory bytes, native exports and newly observed raw snapshots
are distinguished, hashed and reconstructible; the original inventory remains
unchanged. `python tools/drive_coverage.py --json` verifies the declared scope.

A [two-axis Hermite–Gaussian envelope candidate](docs/HERMITE_GAUSSIAN_ENVELOPE.md)
adds exact interval evaluation and 66 reproducible derivative cases through
order 10. It is author-produced and unreviewed; it changes no claim or
obligation status and supplies no full 24-jet or six-pin band result.

### 2026-09-20 membership, native fidelity and RN continuation

The [additive reconciliation](drive/deltas/2026-09-20/DG-RECON-20260920/README.md)
accounts for 4,116 stored file identities, 758 folders and the same 54 holds.
All 744 permitted folders were relisted, with no missing prior observed
membership. Four prior delivery files were added and the live register export
was refreshed. Native checks cover 1,930 single-tab Docs and 39 workbooks with
308 tabs and 14,338 formula cells. Native formula strings and tab names are
preserved alongside the exports, including the observed XLSX transformations.
`tools/drive_reconcile.py` and `tools/native_export_check.py` reproduce the
coverage checks; the latter reparses containers with `--verify-containers`.

The [RN affine-moment candidate](docs/RN_AFFINE_MOMENTS.md) implements exact
correlated Gaussian determinant moments and uniform Bernstein mark bounds,
with source/dimension/order/law/domain applicability checks and falsification
tests. Its finite reuse evaluation discloses four development duplicates
and makes no held-out or general usefulness claim. The recovered
[research context](docs/context/RESEARCH_CONTEXT_20260919_v2.md) preserves
chronology and scientific boundaries; no raw conversation dumps are published.
These additions do not identify an RN field law or change any scientific status.
