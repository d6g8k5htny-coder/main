# Downstream closure crosswalk — historical RN route versus 2026-09-24 successors

**Date:** 2026-09-25  
**Disposition:** dependency classification only. This document changes no scientific
Boolean, frozen object, register row, or historical theorem statement.

**Placement:** `docs/DOWNSTREAM_RN_CROSSWALK_20260925.md`. It does **not** live under
`docs/math_status/`. That packet directory is closed (`EXPECTED_NAMES`); an extra file
there fails `tools/math_status_check.py` without discharging or promoting anything.
This note is navigation and classification only. It also does **not** edit
`docs/OPEN_PROBLEMS.md`, which is byte-pinned by `tools/twelve_project_check.py`.

## Purpose

The hardening branch still correctly records the historical D3-LEMMA-RN-UNIF route as
OPEN: ENV-RESCOV → ALLCELL-FDZ-Q4 → LOGQ-TAIL → SYM-Fw-jet → CH-LIFT, with Piece-2,
OBL-H5-JETMOD and FREEZE downstream. Historical objects `rnu_env.py`,
`CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip`, and `allcell_fdz_enclosures.json` are recorded
ABSENT. This crosswalk prevents newer author-side proofs from being mistaken for those
absent carriers or, conversely, old route-specific walls from being silently treated as
dependencies of every newer draft. The SIDE24 prep triad on this branch stays
ABSENT as a source of truth. This crosswalk does not invent a carrier for it.

This page is the human D0 classification companion to the in-development Math-
[downstream hard gate](https://github.com/d6g8k5htny-coder/Math-/pull/8)
(`DOWNSTREAM-HARD-GATE-20260925-v1`, main [#86](https://github.com/d6g8k5htny-coder/main/issues/86) /
[#90](https://github.com/d6g8k5htny-coder/main/issues/90)). Linking that draft has
scientific effect NONE. As of Math- head `5887a2f8`, the gate still has a load-bearing
own-node eligibility hole: dependency terminality alone can allow
`AUTHOR_SIDE_CANDIDATE` → `controlling`. Do **not** read this crosswalk as evidence
that Math- #8 already enforces every [#90](https://github.com/d6g8k5htny-coder/main/issues/90)
semantic.

## Classification rule

A newer draft can avoid importing a historical-route requirement only when its own
statement does not use that predicate. Avoiding the import does not set the old
predicate TRUE and does not discharge it.
Distinguish historical-route requirements, mathematical interfaces rederived by a
successor, and carrier/replay identities required only for the historical certificate.

Terminal integrity labels used by the Math- gate (not theorem acceptance):
`PROVED_REVIEWED`, `SUPERSEDED_NONBLOCKING`, `REFUTED`, `BLOCKED_ABSENT`. Green CI,
hashes, same-author review, and navigation success are never mathematical discharge.

## Crosswalk

| Historical object / predicate | Math- gate node | Gate classification | Active reading |
|---|---|---|---|
| `rnu_env.py` | `hist.rnu_env.py` | `BLOCKED_ABSENT` | ABSENT; historical-route blocker only |
| `CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip` | `hist.CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip` | `BLOCKED_ABSENT` | ABSENT; historical-route blocker only |
| `allcell_fdz_enclosures.json` | `hist.allcell_fdz_enclosures.json` | `BLOCKED_ABSENT` | ABSENT; historical-route blocker only |
| ENV-RESCOV | `hist.ENV-RESCOV` | `OPEN_HISTORICAL` | OPEN for historical D3. A newer author-side draft may restate a qualitative interface; that restatement does not set this predicate TRUE and does not discharge D3-LEMMA-RN-UNIF |
| ALLCELL-FDZ-Q4 | `hist.ALLCELL-FDZ-Q4` | `OPEN_HISTORICAL` | OPEN for historical D3. The fixed-remote author-side candidate is a different statement and does not discharge this predicate |
| LOGQ-TAIL | `hist.LOGQ-TAIL` | `OPEN_HISTORICAL` | OPEN historically. A named successor draft may use a different tail statement; that does not replace or discharge this predicate |
| SYM-Fw-jet | `hist.SYM-Fw-jet` | `OPEN_HISTORICAL` | OPEN historically. An analogous interface written in a successor draft is not a discharge of this predicate |
| CH-LIFT | `hist.CH-LIFT` | `OPEN_ACTIVE` | OPEN and still mathematically relevant for shrinking charts |
| Piece-2 annulus | `hist.Piece-2-annulus` | `OPEN_ACTIVE` | OPEN; active mathematical blocker for shrinking/pin-collision regions |
| OBL-H5-JETMOD | `hist.OBL-H5-JETMOD` | `OPEN_HISTORICAL` | OPEN for historical/certificate route |
| H5-ZBAND | _(not a separate gate node)_ | — | do not conflate with newer normalizer scopes; historical predicate unchanged |
| H5-REMOTE-THRESHOLD | see D4/D5 region nodes | region inventory | Stays OPEN. A fixed-remote height/spatial candidate is a different statement and does not discharge this predicate. The shrinking/global complement stays OPEN |
| `lemma_closed` | `hist.lemma_closed` | `FALSE` | no successor is an operator fold of D3 |
| `certified_C_H` | _(register companion)_ | false | newer qualitative constants are not this certificate |

## D4/D5 region complement (from the Math- gate)

`COVERED_BY_CANDIDATE` below is the Math- draft gate's own label. It is not
acceptance, not a discharge of `OBL-H5-JETMOD` or `D3-LEMMA-RN-UNIF`, and not a
SIDE24 source-of-truth carrier. SIDE24 stays ABSENT. Engineering classification
is not discharge.

| Region | Gate node | Classification |
|---|---|---|
| Fixed-remote covered set | `math.rn-region.fixed-remote` | `COVERED_BY_CANDIDATE` (draft-gate label only; not discharge) |
| Mesoscopic scaled annulus | `math.rn-region.mesoscopic-scaled-annulus` | `OPEN_ACTIVE` |
| Pin-collision | `math.rn-region.pin-collision` | `OPEN_ACTIVE` |
| Intermediate r≪\|x\|≪ρ | `math.rn-region.intermediate-r-to-rho` | `OPEN_ACTIVE` |
| Witness collision | `math.rn-region.witness-collision` | `OPEN_ACTIVE` |

## Author-side candidates (classification only)

These rows name drafts. They are not accepted theorems. No flag is flipped.
`OBL-H5-JETMOD` and `D3-LEMMA-RN-UNIF` stay OPEN. Engineering records are not
discharge. SIDE24 stays ABSENT; this page invents no source-of-truth carrier.

1. UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1 / main [#63](https://github.com/d6g8k5htny-coder/main/issues/63) / gate `math.uniform-matrix-cap-lifetime` (author-side candidate, not accepted here): claims a compact-mark global elder defect O(r³), unrestricted leading finite-bar density, full normalizer and marked Kac–Rice interfaces.
2. LIFETIME-BOUNDED-REMAINDER-20260924-v1 / main [#67](https://github.com/d6g8k5htny-coder/main/issues/67) / gate `math.lifetime-remainder` (author-side candidate, not accepted here): claims an unrestricted cell^(-1/3)+O(1) bound and bounded nonselected density, importing named #63 interfaces.
3. RN-FIXED-REMOTE-WINDOW-20260924-v1 / main [#76](https://github.com/d6g8k5htny-coder/main/issues/76) / gate `math.rn-fixed-remote-window` (author-side candidate, not accepted here): claims a three-determinant O(r⁵) numerator and O(r³) expected count for the between-pin height window at fixed positive spatial distance.
4. RN-MESOSCOPIC-ANNULUS-REDUCTION-20260925-v1 / [Math- PR #7](https://github.com/d6g8k5htny-coder/Math-/pull/7) (open draft) / gate `math.rn-mesoscopic-reduction`: candidate reduction for x = r y on fixed scaled annuli; exact divided-difference determinant ledger still open.

None is a replacement carrier for the historical ABSENT files. None flips
`lemma_closed` away from false. Reading this page does not discharge
`OBL-H5-JETMOD` or `D3-LEMMA-RN-UNIF`. The Math- hard-gate draft is intended to
refuse CONTROLLING promotion while required dependencies stay non-terminal, but
until its own-node eligibility repair lands it must not be cited as a complete
[#90](https://github.com/d6g8k5htny-coder/main/issues/90) enforcer.

## Downstream consequence

If the goal is to reproduce/promote the historical D1/D3 certificate exactly, the
ABSENT carriers and ordered walls remain blockers. If the goal is a new author-side
statement by the 2026-09-24 analytic route, those carrier identities are not
dependencies unless explicitly imported; the named analytic-review interfaces plus
the shrinking-annulus/collision complement stay open. This page does not promote
either route. A future operator may formally retire the
historical route as superseded only through the applicable register/fold mechanism.
This crosswalk does not perform that governance action.

## Related reading

- [RN status (historical route walls)](math_status/STATUS_RN_UNIF.md)
- [Math status packet](math_status/README.md) — OPEN/HOLD; closed directory
- [Open problems §A5](OPEN_PROBLEMS.md) (pinned; do not rewrite for navigation)
- [Research index — RN counting](RESEARCH_INDEX.md)
- Fixed-remote candidate (external file named PROOF.md; not accepted here): [Math- remote_window PROOF](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/remote_window_20260924/PROOF.md)
- Mesoscopic draft: [Math- PR #7](https://github.com/d6g8k5htny-coder/Math-/pull/7)
- In-development D0–D7 hard gate (eligibility repair still open): [Math- PR #8](https://github.com/d6g8k5htny-coder/Math-/pull/8)
- Math- #8 eligibility repair handoff (main cannot push Math-): [MATH_PR8_ELIGIBILITY_HANDOFF_20260925.md](MATH_PR8_ELIGIBILITY_HANDOFF_20260925.md)
- Mesoscopic reduction challenge (not R17): [RN_MESOSCOPIC_REDUCTION_CHALLENGE_20260925.md](RN_MESOSCOPIC_REDUCTION_CHALLENGE_20260925.md)
- Owner surfaces: [#86](https://github.com/d6g8k5htny-coder/main/issues/86), [#90](https://github.com/d6g8k5htny-coder/main/issues/90)
