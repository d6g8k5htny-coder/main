# STATUS — JETMOD and RN-UNIF (OPEN/HOLD)

**As of:** 2026-09-22
**Base:** `chatgpt/drive-github-hardening-20260919` at `988b0db6fb8cc4debff5d1b5db9d90f72cfdceaf` (merge of PR #9)
**Audience:** eng lead (agent 5). Skim-trap for agents 1 and 2 before any merge.
**Discipline:** fail-closed. OPEN/HOLD only. No lemma flips. No prize closes.

This file is a status snapshot. It is not a claim log, not a register write, and not an operator decision. It does not edit `claims/graph.json` or `registers/json/work_events.json`. Authority is NONE. `independence_credit` is 0.

A green run of `tools/math_console.py`, of `tools/rn_bernstein_sharp_check.py`, or of CI is a run. Green ≠ discharge.

## Controlling flags

| Flag | Value |
|---|---|
| disposition | **OPEN/HOLD** |
| `OBL-H5-JETMOD` | **OPEN** (display only) / **HOLD** |
| `D3-LEMMA-RN-UNIF` Piece 1 | **OPEN** / **HOLD** |
| `D3-LEMMA-RN-UNIF` Piece 2 | **OPEN** / **HOLD** |
| `lemma_closed` | **false** |
| `prizes_solved` | **0** |
| `original_prizes_solved` | **0** |
| `independence_credit` | **0** |
| lemma flips | **0** |
| tip Bernstein discharges an obligation | **false** |
| green checker discharges an obligation | **false** |
| authority | NONE |
| float path | NON-CERTIFYING |

`lemma_closed` stays false. `prizes_solved` stays 0. `original_prizes_solved` stays 0. Nothing in this packet moves those flags.

## `OBL-H5-JETMOD` — OPEN / HOLD

Contract, from `docs/OPEN_PROBLEMS.md` A1 (no new mathematical claim):

The obligation is interval bounds for the full 24-jet set — not just the displayed `c₂` — over the r-bands `[r_{k+1}, r_k]`, with lattice-tail constants re-certified uniformly in the band. For each jet `J` and band `B`, `J(B)/r^{p_J}` lies in a certified interval. The named proof step evaluates the lattice sums with `r` as an interval over the band, yielding G12-band enclosures and `Î(r)/r³ ≤ F(G12-band)` for the whole band.

**Width-vs-modulus falsifier.** A band enclosure whose width exceeds the claimed modulus. That comparison is the falsifier named in A1 and in `claims/graph.json` (`OBL-H5-JETMOD`: "Falsifier: a band enclosure wider than the claimed modulus."). A width that exceeds the claimed modulus falsifies a modulus-ready enclosure of that band. Recording the falsifier shape does not close the obligation, does not discharge it, and does not flip `lemma_closed`.

RUNG2 (`r = 0.025`) and RUNG3 (`r = 0.035355`) certify those rungs only. RUNG2 and RUNG3 do not discharge `OBL-H5-JETMOD`.

`research/bands/` holds interval-`r` machinery exercised on reference kernels. `docs/OPEN_PROBLEMS.md` A1 records what that code still lacks before it can bear on the obligation: a decay envelope for the program's `kplane` at every order the 24-jet set reaches, the 24-jet definitions with their powers `p_J`, the band endpoints `r_k`, and the six-pin `r`-to-displacement geometry. That paragraph is repository state (code, not status). `OBL-H5-JETMOD` remains OPEN (display only) and on HOLD.

## `D3-LEMMA-RN-UNIF` — both Pieces OPEN / HOLD

Contract, from `docs/OPEN_PROBLEMS.md` A5:

* **Piece 1** OPEN. **Piece 2** OPEN.
* Both Pieces remain OPEN.
* Receipts that this repository already carries still read `lemma_closed: false`. This snapshot does not flip that flag.

A5 records that an annulus driver now exists at `research/cover/`, that `total()` refuses to return while any cell is pending, and that the later RN cell work (SIDE24 point law, density/window candidate, local square) does not turn the generic cover into an RN cover of `0.1 ≤ |y| ≤ 5`. The local square is a nonzero-area calculation on one square. It is not the annulus. The frozen engine under `engine/rn_engine/frozen/` is `mpmath` throughout. `claims/graph.json` keeps `D3-LEMMA-RN-UNIF` at `status_frozen_v2_2: NOT_CLOSED` and `status_register_note: OPEN`, with evidence of kind carrier binding only (`certifying: false`). This packet does not amend that node.

Piece 1 stays OPEN. Piece 2 stays OPEN. `lemma_closed` stays false. `discharges_lemma` stays false.

## Tip Bernstein ≠ discharge

The tip tree contains the Bernstein and sharp-variance successor described in `docs/RN_BERNSTEIN_SHARP.md` and replayed by `tools/rn_bernstein_sharp_check.py`. That successor improves an existing local RN wedge at fixed normalized SIDE24 radius `r = 1/20`. Its own note says it does not certify the full annulus, all radii, or pin orientations, and it does not close q0 or an original prize problem. No scientific status changes. Reviews there are source-exposed, same-provider technical checks with zero organizational independence credit.

That Bernstein successor is not a discharge of `OBL-H5-JETMOD`. It is not a close of Piece 1 or Piece 2 of `D3-LEMMA-RN-UNIF`. A green Bernstein checker is not obligation discharge. `prizes_solved` stays 0. `original_prizes_solved` stays 0. `independence_credit` stays 0.

## What this snapshot does not establish

* It does not discharge `OBL-H5-JETMOD`.
* It does not close Piece 1 or Piece 2 of `D3-LEMMA-RN-UNIF`, and it does not FREEZE either.
* It does not flip `lemma_closed`. The flag stays false.
* It does not solve a prize. `prizes_solved = 0`. `original_prizes_solved = 0`.
* It does not award independence. `independence_credit = 0`.
* It does not treat a display, a width-vs-modulus comparison, a local Bernstein enclosure, or a green process exit as a certified enclosure of either obligation.
* It does not compose the 2D track with the 3D lifetime track, and it does not move the prize track into the q0 dependency graph.

## Console

```bash
python3 tools/math_console.py
```

The console prints an OPEN/HOLD snapshot and exits 0. The exit prints `lemma_closed false` and `original_prizes_solved 0`. Exit 0 means those constants were still false and 0. Exit 0 certifies nothing.

## Skim-trap (agents 1 and 2, before merge)

Read this file and `tools/math_console.py` for a status word that leaves OPEN/HOLD.

* `lemma_closed` is false in both files. There is no assignment that sets it true.
* `prizes_solved` and `original_prizes_solved` are 0.
* `independence_credit` is 0.
* The width-vs-modulus sentence is the A1 falsifier. It is not a discharged enclosure.
* "Bernstein" on this tip is the local wedge successor. Tip Bernstein ≠ discharge.
* Green ≠ discharge. Do not merge on a green console.
