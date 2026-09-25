# Downstream closure crosswalk — historical RN route versus 2026-09-24 successors

**Date:** 2026-09-25  
**Disposition:** dependency classification only. This document changes no scientific
Boolean, frozen object, register row, or historical theorem statement.

**Placement:** `docs/DOWNSTREAM_RN_CROSSWALK_20260925.md`. It does **not** live under
`docs/math_status/`. That packet directory is closed (`EXPECTED_NAMES`); an extra file
there fails `tools/math_status_check.py` without discharging or promoting anything.
This note is navigation and classification only.

## Purpose

The hardening branch still correctly records the historical D3-LEMMA-RN-UNIF route as
OPEN: ENV-RESCOV → ALLCELL-FDZ-Q4 → LOGQ-TAIL → SYM-Fw-jet → CH-LIFT, with Piece-2,
OBL-H5-JETMOD and FREEZE downstream. Historical objects `rnu_env.py`,
`CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip`, and `allcell_fdz_enclosures.json` are recorded
ABSENT. This crosswalk prevents newer author-side proofs from being mistaken for those
absent carriers or, conversely, old route-specific walls from being silently treated as
dependencies of every newer theorem.

## Classification rule

A newer theorem can bypass a historical-route requirement only when its own statement
and proof do not import that predicate. This does not set the old predicate TRUE.
Distinguish historical-route requirements, mathematical interfaces rederived by a
successor, and carrier/replay identities required only for the historical certificate.

## Crosswalk

| Historical object / predicate | 2026-09-24+ successor relation | Active classification |
|---|---|---|
| `rnu_env.py` | no successor claims byte identity or reconstruction | **ABSENT; historical-route blocker only** |
| `CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip` | no successor claims to be this archive | **ABSENT; historical-route blocker only** |
| `allcell_fdz_enclosures.json` | newer proofs do not claim this payload | **ABSENT; historical-route blocker only** |
| ENV-RESCOV | newer matrix/lifetime proof derives finite-jet covariance positivity from positive Fourier spectrum | **OPEN for historical D3; rederived qualitatively for newer route** |
| ALLCELL-FDZ-Q4 | fixed-remote count is analytic on fixed separated domains and does not consume the old cell carrier | **OPEN for historical D3; not a dependency of fixed-remote theorem** |
| LOGQ-TAIL | unrestricted lifetime proof supplies a separate Gaussian target-growth majorant; fixed-remote count uses compact marks | **OPEN historically; replaced only inside named successors** |
| SYM-Fw-jet | newer sources use centered Hermite/divided-difference transforms and finite-jet rank; no historical certificate identity claimed | **OPEN historically; analogous interface rederived where proved** |
| CH-LIFT | fixed-remote theorem avoids shrinking chart singularity by fixed ρ; mesoscopic Math- PR #7 reopens the scaled chart problem | **OPEN and still mathematically relevant** |
| Piece-2 annulus | fixed-remote covers only distance ≥ ρ; Math- PR #7 addresses x = r y on fixed scaled annuli; intermediate/pin-collision regions remain | **OPEN; active mathematical blocker** |
| OBL-H5-JETMOD | no 2026-09-24 proof claims a certified 24-jet band enclosure | **OPEN for historical/certificate route** |
| H5-ZBAND | newer sources have different normalizer results/scopes | **do not conflate; historical predicate unchanged** |
| H5-REMOTE-THRESHOLD | fixed-remote count supplies cubic expected count only on its declared height/spatial region | **partially bypassed there; shrinking/global complement OPEN** |
| `lemma_closed` | no successor is an operator fold of D3 | **FALSE** |
| `certified_C_H` | newer qualitative constants are not this certificate | **FALSE** |

## Current active proof chain

1. UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1 / main [#63](https://github.com/d6g8k5htny-coder/main/issues/63): compact-mark global elder defect O(r³), unrestricted leading finite-bar density, full normalizer and marked Kac–Rice interfaces.
2. LIFETIME-BOUNDED-REMAINDER-20260924-v1 / main [#67](https://github.com/d6g8k5htny-coder/main/issues/67): unrestricted cell^(-1/3)+O(1) candidate and bounded nonselected density, importing named #63 interfaces.
3. RN-FIXED-REMOTE-WINDOW-20260924-v1 / main [#76](https://github.com/d6g8k5htny-coder/main/issues/76): actual three-determinant O(r⁵) numerator and O(r³) expected count for the between-pin height window at fixed positive spatial distance.
4. RN-MESOSCOPIC-ANNULUS-REDUCTION-20260925-v1 / [Math- PR #7](https://github.com/d6g8k5htny-coder/Math-/pull/7) (open draft): reduction for x = r y on fixed scaled annuli; exact divided-difference determinant ledger still open.

None is a replacement carrier for the historical ABSENT files. None flips
`lemma_closed` away from false. Reading this page does not discharge
`OBL-H5-JETMOD` or `D3-LEMMA-RN-UNIF`.

## Downstream consequence

If the goal is to reproduce/promote the historical D1/D3 certificate exactly, the
ABSENT carriers and ordered walls remain blockers. If the goal is a new theorem by the
2026-09-24 analytic route, those carrier identities are not dependencies unless
explicitly imported; the live blockers are the named analytic-review interfaces plus
shrinking-annulus/collision complement. A future operator may formally retire the
historical route as superseded only through the applicable register/fold mechanism.
This crosswalk does not perform that governance action.

## Related reading

- [RN status (historical route walls)](math_status/STATUS_RN_UNIF.md)
- [Math status packet](math_status/README.md) — OPEN/HOLD; closed directory
- [Open problems §A5](OPEN_PROBLEMS.md)
- [Research index — RN counting](RESEARCH_INDEX.md)
- Fixed-remote proof: [Math- remote_window PROOF](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/remote_window_20260924/PROOF.md)
- Mesoscopic draft: [Math- PR #7](https://github.com/d6g8k5htny-coder/Math-/pull/7)
