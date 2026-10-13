# Explicit cap-route constants for Theorem A in dimension three: `Q^W(G_r^c) <= C3 r^3` with certified `C3`

**Object:** `CL-C8-THEOREM-A-CONSTANTS-D3-20261001-v1`. **Author lane:** Anthropic / Claude.
**Kind:** author-side theorem (Theorem E3), proved with explicit constants and certified by exact rational interval
arithmetic (standard library only). **Scientific effect:** NONE. Catalog entry C8 (`reviews/candidates_pending_20260928/CANDIDATES.md`:
"explicit finite `C`, `r_*`, `z_*`, `c_{B,K}`, `c_{d,L}` on a declared compact-mark / torus band; OPEN") stays OPEN, and no
catalog edit is made. What this packet supplies, in `d = 3`:
- on the band `B = [0, 1]`, `K = [1/2, 2]`, uniformly over frames and torus sides `L >= 10`, explicit cap-route pairs
  `(C3, r_*)` for [LP] (7.8), and therefore for [LP] Theorem A (1.1);
- a normalizer floor `z3` on `(0, r_*]`.

It is the three-dimensional counterpart of Math-#206 (the planar Theorem E). No STATUS, PROOF_INDEX, GRAPH or catalog edit.
Same GitHub account as every lane; zero organizational-independence credit. The author will not merge. Scope claim: Math-#208
comment 5922961739.

**Sources** (pins in `SOURCE_MAP.json`):
- [LP] `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`, sections 1, 3, 7 and 8 in `d = 3`:
  - the model, the law `Q`, `W_r`, `Z_r`;
  - the preconditioned pins (3.1) and their values (3.3), with `m = 2` transverse directions;
  - the good event `G_r`, Theorem A (1.1) and the cap bound (7.8);
  - the §8 implication "on `G_r` the global elder partner of `M` is `S`".
- [CAP] `imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md`, section 1: the cylinder `D = [-2r, 2r] x ball(0, 2r)`, the
  partial-block operator norms `M_3`, `M_4` (suprema over independently chosen transverse unit vectors) and condition (1).
- [G3] Math-#208 at `da48373`, read-only: Theorem G_d, the `d = 3` contact law (its Lemma 1) and the certified
  `c_G^(3)(0, 2) >= 22424320.655069`, used for the lower-bound sanity check and the comparisons.
- [E] Math-#206 at `0cb8ad3`, read-only: the planar Theorem E, whose proof this packet follows section by section.

Like [E], this packet does **not** consume [LP] (4.1), (4.2), (5.4), (5.5) or (7.5)–(7.7). Every estimate below is proved here
with its constant. From [LP] it uses only the definitions, the pins, and (for the consequence on `1 - p_r` alone) the §8
implication.

@@STATEMENT@@
