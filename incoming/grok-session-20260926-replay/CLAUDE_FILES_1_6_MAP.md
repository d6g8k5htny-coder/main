# Claude Files 1–6 map (2026-07-05) — not Math- default

Scientific effect: NONE. Does not accept GitHub Theorem A, File-1 Theorem A, File-5 Theorem B, Morse–Smale, or Condition (ND).

Sources: user-attached PDFs dated 2026-07-05. Org-wide GitHub search for `Condition (ND)`, `PROVEN-MODULO`, `q0 = 1`, `Pinning Lemma`, `near-diagonal`, `C005R`, `GCJA`: **zero hits**. This stack is **ABSENT from public GitHub**. Drive/chat attachments only.

Two File-1 copies and two File-5 copies are the same text. Do not count them twice.

## What the stack claims, with its own labels

File 1 Theorem A (selection): on \(T_L^2\), \(L\) fixed,

    lim_{r\u21920} q(r,b) = 1 uniformly on compact B,
    1 − q(r,b) ≤ C(L,B)[r³ + r^β] + C exp(−c/r²).

Label: **PROVEN-MODULO** Lemma P, Lemma I, Sublemma R0.
Matching lower bound \(1−q ≥ c r³\) is **OPEN**.
Candidate \(β=4\) is **CONJECTURE** as an exact exponent; any \(β>0\) gives (2.1).

File 1 Theorem B (near-diagonal law):

    ν(ℓ) = C_* ℓ^{−1/3}(1+o(1)),
    C_* = (1/3) ∫_B h(b) J(b) db.

Label: **PROVEN-MODULO** Theorem A’s lemma set + Proposition B2.
Selection factor in \(C_*\) is exactly 1 *if* Theorem A holds.

File 2: a.s. Morse is literature; a.s. Morse–Smale (no saddle connections) is **OPEN** for every ensemble checked. Sublemma R0 is a folklore gap, not a citation.

File 3: two-scale three-point Kac–Rice framework claimed new. Lemma I counting bound **PROVEN-MODULO Condition (ND)**. (ND) = nondegeneracy of the leading 15×15 blown-up jet covariance. File 1 §5: this is the single architected failure point.

File 4: pair-Palm + cubic pinning. Self-label PROVEN-HERE for P2/P3. K2 annulus floor claimed discharged by P3+P4. Does **not** touch (ND) or Lemma I.

File 5: derivation architecture of Theorem B. Does not upgrade (ND).

O1 audit: **RETRACTION R3**. Super-exponential \(E[N_{\mathrm{inner}}]=O(\exp(−c r^{−4}))\) is wrong. Palm \(|\det|\)-bias fattens the small-transverse-curvature tail; restored law \(E[N_{\mathrm{inner}}] ≍ r^4\). File-1 Theorem A is unaffected *if* \(β=4>0\). Remaining kill objects named there: O2, B2.

GCJA: graded conditional-jet asymptotics for collapsing designs. Separate lemma package; does not close (ND).

C006 Arm1c-v2 (already ingested): ABORTED BY GATE at C-2. Does not confirm the discriminator. Queued in File 1 §8.3 as the window-count cycle.

## Kill registry in File 1 §9 (not GitHub STATUS)

- K1 RETRACTED: \(r^{6/5}\) rate for \(1−q\).
- K2 RETRACTED-SUPERSEDED: \(O(K^{−3})\) annulus floor.
- K3 RETRACTED: \(b\)-threshold refinement at fixed \(L\).

Arm1c’s standing names K4 / R3 / R3b are a **later instrument registry**, not these three lines.

## Map onto public Math- (no status transfer)

| July-5 object | Public Math- object | Transfer? |
|---|---|---|
| File-1 Thm A, \(q\to1\) on \(T_L^2\) | D1 parent Thm A, \(1−p\le C r^3\) all \(d\), periodized BF | **No.** Different dimension, different remainder architecture, different review. GitHub D1 is AMEND on A1–A7. |
| File-1 Thm B \(\nu\sim C_*\ell^{−1/3}\) | D2 Thm R remainder + D3 SIDE24 coefficient | **No.** D2 ACCEPT is scoped existential \(O(1)\) remainder and does not consume D1. D3 is coefficient arithmetic. |
| File-2 R0 Morse–Smale | Parent §8 Morse + distinct values | **No.** Cap pairing on Math- already flags Morse/§8 as a remaining import. File 2 says the global Morse–Smale statement is literature-open. |
| File-3 (ND) / Lemma I | D5 pin / two-scale addendum | Related *theme* (two-scale inner zone). Not the same matrix. (ND) is OPEN here; D5 pin is AMEND there. |
| File-4 pinning / \(U_r\)-style divided differences | Parent §3 contact frame \(T_r\), det \(12 r^{−(d+3)}\) | Cousin constructions. Do not identify. |
| O1 \(E[N_{\mathrm{inner}}]\asymp r^4\) | Parent §7 \(E[W 1_{\mathrm{depth}}]=O(r^5)\) then \(/Z=O(r^3)\) | Different measures (count vs typed weight). Do not add exponents. |
| Fold lock \(\ell=(\kappa/6)r^3\) | D2 fold ledger / lifetime remainder | Same elementary normal form. Already used on Math- at reviewed D2 scope. |

## What this round does not do

- Does not adopt any `[PROVEN-HERE]` self-label into STATUS.
- Does not close Condition (ND).
- Does not prove Morse–Smale.
- Does not merge main #163 as acceptance.
- Does not treat Arm1c P-1c-* numbers as a discriminator.
