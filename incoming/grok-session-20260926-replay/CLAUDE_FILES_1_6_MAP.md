# Claude Files 1–6 (2026-07-05) — scoped map, not promotion

**Scientific effect:** NONE.
**Does not accept GitHub D1 Theorem A.** Does not close Condition (ND).
**Does not adopt manuscript [PROVEN-HERE] self-labels into STATUS.md.**

Sources: attached PDFs File1–File5, O1 audit, GCJA. These are **not** on Math- default as proof bytes.

## What File 1 actually claims

On `T_L^2` with **L fixed**, hypotheses (H1+)+(H2') only (no isotropy):

**Theorem A (selection, q0=1).** `lim_{r	o0} q(r,b)=1` uniformly on compact heights.
Remainder written as

    1-q(r,b) ≤ C(L,B)[ r^3 + r^β ] + C exp(-c/r^2).

Label of record: **PROVEN-MODULO** {Lemma P, Lemma I, Sublemma R0}.
Candidate `β=4` is **CONJECTURE** as an exact exponent; any `β>0` gives the limit.
Matching lower bound `1-q ≥ c r^3` is **OPEN**.

**Theorem B (near-diagonal law).**

    ν(ℓ) = C_* ℓ^{-1/3} (1+o(1)),
    C_* = (1/3) 6^{2/3} ∫_B ρ_0(b) E_b[κ^{-2/3}] db.

Selection factor in C_* is exactly 1 *if* Theorem A holds. Label: **PROVEN-MODULO** Theorem A + Prop B2 + fold lock (0.1).

Fold lock `ℓ=(κ/6)r^3` is elementary and labeled PROVEN-HERE. That identity is the same cubic already used on GitHub D2 / fold ledger.

## Load-bearing OPEN set in File 1 §6

```
{ Lemma I }   = two-scale 15×15 jet nondegeneracy at (r, r^2)
```

File 3 isolates this as Condition (ND). File 4 discharges Lemma P and does **not** touch ND. File 5 does **not** upgrade ND.

Sublemma R0 (a.s. Morse–Smale / no saddle connections) is a **literature gap** (File 2: no published Gaussian-field Morse–Smale theorem). Parent Math- §8 is Morse + distinct values only — a weaker object.

## Kill registry that must travel

| ID | Status |
|---|---|
| K1 r^{6/5} rate | RETRACTED |
| K2 O(K^{-3}) annulus floor / two-parameter (K,r) limit | RETRACTED-SUPERSEDED by full pair Palm |
| K3 b-threshold / percolation at fixed L | RETRACTED |
| O1 / R3 super-exp `E[N_inner]=O(exp(-c r^{-4}))` | **RETRACTED** (see `O1_R3_VS_R3B.md`) |

## Do not flatten onto GitHub STATUS

| July-5 object | GitHub object | Same? |
|---|---|---|
| File1 Thm A `q	o1` on `T_L^2` | D1 parent `1-p≤ C r^3` all-d + cap `G_r` | **No.** Same slogan, different architecture and dimension. D1 still AMEND. |
| File1 Thm B `ℓ^{-1/3}` | D2 Theorem R scoped ACCEPT | Related descendant. D2 does **not** consume unreviewed File1 Thm A. |
| File3 ND / Lemma I | D5 pin / microdisk + A2 Σ_r | Same *regime* (degenerating third point). Not closed. |
| File2 Morse–Smale | SARD-G A1 + parent §8 | File2 says the global MS statement is unpublished. |
| C006 Arm1c | registered test of O1 flat-saddle tail | v2 ABORTED BY GATE. Not a theorem. |

## Fixed-L warning

File 1 §7.1 excludes `L	o∞` before `r	o0`. SIDE24 is periodized side 24, i.e. fixed L. That is compatible. Infinite-volume percolation is a different theorem and remains relocated/OPEN.
