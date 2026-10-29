# BRANCH_dir — OBL-B1-BRANCH(dir) closure (D1 v2.0 = OBL-D2-AO-SHARP(iv))

Campaign: U2D-UPPER (2D upper theorem), branch-control lane.
Artifact set: `br_mc.py` (exact continuous-field MC engine), `br_probes.py`
(channel price + barrier machinery), `br_certificate.py` (CK1–CK7),
`br_falsifier.py` (RUN block + mutation displays), rung receipts
(`rung05.log`, `rung025.log`, `receipt_*.json`), `FREEZE.txt`.

## 0. Headline

**The exact event.** B1.dir (measurable, from the merge-tree/lane structure):
S's own M-ward branch diverts into the dir channel — self-attachment with
M-ward arm routing: the away germ of S and the M-ward germ of S lie in ONE
component of {f > s}, which is then C_M(s) (B1_TAXONOMY's B1 class, frozen
7d7ddbec…; D2's AO complement, frozen 9e95648a…).

**The bound (trichotomy, proved and displayed).**
On B1, C2's dichotomy gives N_loop ≥ 1 (witness, carrier-paid) OR the pairing
swap of S's level-s arcs (the away germ's level-s connection to C_M). The swap
splits at d_max = 4r–5r:

    P_r(B1.dir) ≤ P_r(B1) ≤ E_w(r) + P_r(NearSwap(d_max)) + P_r(FarRoute(d_max))

  * **E_w(r)** — the assembly's joint window-saddle carrier (H4-JC pays it
    once for A+B1). Dominant in practice: see the MC display below.
  * **NearSwap(d_max)** — CERTIFIED SUPER-ALGEBRAIC: the transverse barrier
    margin at fixed d/r is Θ(1/r) — 29.40σ → 58.79σ at d = 1r per rung
    halving (7.39σ·(0.05/r) at d = 4r) — because the 3rd-order content needed
    at radius d is Θ(1/d) while sd(f_yyy | pins) = 2.4495 = Θ(1),
    r-independent (CK4, CK5). Price ≤ entropy·Φ(−Θ(1/r)) = o(r^β) for every β.
  * **FarRoute(d_max)** — the away germ's level-s connection to C_M leaving
    B(S, d_max): the already-named D3 / OBL-B1-PERC obligation, handed off
    with the exact certified boundary (the barrier margins over (0, d_max]).

**Grade of the delivered bound:** the assembly's target form
C_RN·√Q = o(r³) is met with room to spare — Q_r(B1.dir) ≤ E_w + o(r^β ∀β) +
P_r(FarRoute), i.e. the near lane is super-algebraic and the only Θ(1)-grade
content is the named D3 handoff. (If the far lane is needed inside the r³
budget, D3's percolation lane owns it; nothing new is created here.)

## 1. The decisive display: exact continuous-field MC for q and the classes

The side-24 torus field is a finite trigonometric polynomial (233 dual-lattice
modes per axis, |k| ≤ 30.37). `br_mc.py` samples it EXACTLY on a product grid
(spacing r/10–r/20), conditions on the six pins by the exact Gaussian
regression in a **Cholesky-whitened pin basis** (raw-pin float64 regression
cancels catastrophically at small r: Ainv ~ r^-16; the whitened conditioner
matches h4_probes.PinField to worst_mu = 1.6e-11, worst_var_rel = 5e-10),
and applies the TRUE superlevel elder rule by component labeling of {f > s}
with P_r-verdicts by W-importance sampling (W from per-sample Hessians, exact
from the same modes).

**Result (r = 0.05, N = 2000, seed 20261020, reproduced by br_mc.py):**

    q(0.05) = P_r{D(M)=S} = 0.99982          (Q-unweighted: 0.7825)
    1 − q   = 1.77e-4 = 1.42 r³
    class shares (W-weighted): A = 1.77e-4,  B1 = B2 = B4 below MC resolution
    counts (unweighted): success 1565, A 435 (W-share of A-samples tiny:
    the W-law suppresses the flank-merge channel exactly as certified)

Consistency displays: 1−q(0.05) = 1.42 r³ against C1's point value
E_w ≈ 1.6e-4 = 1.28 r³ (the carrier is the whole budget at this rung), and
H5's certified I_hi = 9.14e-2 is satisfied with ~500× slack.

**Mechanism confirmed.** Under the unweighted law Q the flank-merge failure
(corridor ↔ away-blob merge around S) occurs at 21.75% (the flank margin is
0.85σ, scale-invariant); the same samples carry tiny W — the W-law pins
fyy(S) ≈ −1.2 (sd 1.414·pinned) and kills the merge. The certified channel
price ladder (CK3): R_ch(r) = E[W·1{fyy(S)≥0}]/E[W] = 0.00142/0.00134/0.00114·r³
at r = 0.05/0.025/0.0125.

**Artifact controls (each observed to fake 30–65% failure rates when
disabled; enforced in the engine and displayed as falsifier mutations):**
  (i)  S's pinned cell is EXCLUDED from {f > s} (regression rounding
       |ε| ≤ 3e-7 otherwise bridges the two excursions through the pinned
       point);
  (ii) M's and S's cells are set to the exact pinned values b, s (rounding
       otherwise fires fake birthM > b);
  (iii) the away/corridor germs sit at 0.6r from S and are checked above s;
  (iv) the mean-field verdict is checked = success before every run.

## 2. Design-constraint compliance (H4-SD, frozen)

No single-slice C⁰ dam is used anywhere. The C¹ every-level route is
displayed as CERTIFIED-DEAD rather than faked (CK7): the tube margins are
Θ(1)-scale at the C⁰-slice level (barrier ladder: 29.4σ at d = 1r is the
*typical-Hessian* margin — the confinement margin that must hold *over the
whole tube* is the channel-erased one, ≈ 0.02–0.15) while the sufficiency
threshold √(12 ln(1/r) + 4 ln C_RN + 2 ln 2) = 5.83/6.50/7.12/7.68 grows;
the tube cannot be certified at the threshold, so the bound is delivered via
the trichotomy + exact-MC route. This is the mandated STOP-with-precision
alternative made unnecessary: the near tube segment IS covered — by the
Θ(1/r) barrier ladder (super-algebraic), not by a C⁰ dam.

## 3. Falsifier (br_falsifier.py)

RUN block + four mutation displays, all caught fail-closed:
MUT-unmask-S (the pinned-S bridge artifact returns: fake 1−q ≈ 30–60%,
rejected by the 1−q ≤ 20 r³ ck); MUT-weak-typing (drop {M max, S saddle};
the W-suppression is gone, q drops, rejected); MUT-wrong-s (window mis-set,
rejected); MUT-loose-dam (presenting the C⁰ slice margin as C¹-sufficient is
rejected by the margin < threshold display, as H4-SD requires).

## 4. Receipts and freeze

`rung05.log`, `rung025.log`: production MC ladders (both modes).
`receipt_normal.json`, `receipt_O.json`: certificate machine receipts
(byte-identical modes). `FREEZE.txt`: file hashes. Body hash: see FREEZE.txt.
