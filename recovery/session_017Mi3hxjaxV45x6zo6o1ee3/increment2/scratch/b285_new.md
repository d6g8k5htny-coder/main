**Object:** `CAP-FIRST-EXIT-LEAN-20261002` v2.5, on `main` `d47b90e` (v2.4 merged as `8815d05`, #284).
- **Author:** Anthropic / Claude, session `017Mi3hx…`. This lane authored the packet; no other lane has a claim on it.
- **Scientific effect NONE.** No register, catalog, STATUS, PROOF_INDEX, GRAPH, workflow or `formal/` change.
- Same GitHub account as every lane, so organizational-independence credit is 0.
- **The author will not merge.**

## What changes

CAP §1 ([`MARKED_CYLINDER_CAP_PROOF.md`](https://github.com/d6g8k5htny-coder/Math-/blob/d47b90e/imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md), lines 15–26) states the deterministic theorem with the partial-block norms

    M_j = max_{a+c=j} sup_D ‖∂_x^a D_y^c f‖_op,      λ = λ_min(−D_y² f(M)),
    (1)  λ > (4/(3κ)) r M_3²,   r M_4 ≤ 3κ/10.

The landed `ridge_capHyp_K` (v2.4) takes the third derivative as one aggregated **block inequality** `hT`, not as CAP's partial blocks.

v2.5 adds 6 declarations and changes nothing else:

| Declaration | Content |
|---|---|
| `iteratedFDeriv_three_swap₁₂`, `iteratedFDeriv_three_swap₂₃` | **`D³f` is symmetric for `f ∈ C³`.** Mathlib proves the symmetry of iterated derivatives only for analytic functions. For `C²` it has the second derivative (`ContDiffAt.isSymmSndFDerivAt`). The first swap applies that to `Df` through `D³f[a, b, c] = D²(Df)[a, b] c`. The second uses `D³f[a, b, c] = (D(D²f)[a])[b, c]`: `D²f` is symmetric near `p`, so `q ↦ D²f(q)[b, c] − D²f(q)[c, b]` vanishes near `p`, and its derivative at `p` is zero. |
| `block_of_partial` | For a trilinear form `B` on `ℝ × E` that is invariant under both swaps, CAP's three transverse partial blocks (`c = 1, 2, 3`) bounded by `m` give `hT` with constant `m`. The proof expands `B[(s₁, y₁), (s₂, y₂), (s₃, y₃)]` multilinearly. Each of the seven mixed terms is a partial block in some slot order. The swaps bring it to the canonical order, and the seven bounds add up to `m(Π(|sᵢ| + ‖yᵢ‖) − Π|sᵢ|)`. No sign of `m` is assumed. |
| `ridge_capHyp_blocks` | **The cap under CAP (1) as written:** `ridge_capHyp_K` with `hT` replaced by CAP's partial-block bounds on `D`, and `M3`, `M4` in (1). Its `c = 0` block is assumed on the pin segment only, and the fourth-derivative blocks only for `c = 0, 1`, as in v2.4. CAP's `M_j` bound all of these, so CAP's (1) implies the hypotheses. |
| `mixedToyF`, `mixedToy_blocks` | `h1ToyF + xy²`. The pins and the gap `1/6` are unchanged. Lean computes `D³f[a, b, c] = 2a₁b₁c₁ + 2(a₁b₂c₂ + a₂b₁c₂ + a₂b₂c₁)` and `D⁴f = 0`. So the partial blocks are `2, 0, 2, 0`, `M3 = 2`, `M4 = 0` and `λ = 41`. (1) reads `32 < 41` and `0 ≤ 1/20`. The mixed block `∂_x D_y² f = 2` is nonzero, so the derived `hT` is not trivial, unlike in the earlier toys. |

**What it closes.** Within I1, the gap between CAP's own norms and the Lean hypotheses. The cap theorem now takes exactly CAP §1's (1), with CAP's `M_j` and `λ`.

**Not closed.** I2 (persistence bookkeeping), I3 (that the local field on `G_r` satisfies the §2 setting), I4 (every probabilistic statement), and the identification of P's frame and the periodic lift with the Lean objects. `E` is any finite-dimensional normed space; CAP's transverse norm is Euclidean, which is a special case.

**Unchanged.** The 132 v2.4 declarations and all their lines. The toolchain, the pins and the workflow `cap-first-exit-lean.yml` (blob `d2bf2186`) are also unchanged. The diff against `main` is 5 packet files:
- `CapFirstExit.lean`: +338 lines and no deletions. One v2.5 paragraph is added to the header comment; everything else is a new section `PartialBlock` and the toy after `ridge_capHyp_norm_K`;
- `ALIGNMENT.md`: three §2 rows, one §3 sentence, the revised I1 row and the additivity note;
- `README.md`: one paragraph, four table rows and one sentence in "What is not proved here"; the count changes from 132 to 138;
- `MANIFEST.json`: six targets with their axiom sets, and new file hashes;
- `test_gate.py`: the count changes from 132 to 138.

## Validation (local)

- `gate.py` in source mode: PASS, 138 targets, at `9680d24` (after merging `main` `d47b90e`). `MANIFEST.json` SHA-256: `9fc8510a5ba7f4f2673bb7afa7a293f7588116224f53f09e2288d356e72b8337`. Packet tree `6e533356`.
- Unit tests: OK in `-B -S` and in `-B -O -S`.
- `gate.py --execute` at `4f2d048` (identical packet bytes): PASS. The receipt records:
  - a fresh build;
  - `leanchecker` PASS;
  - all 138 declarations using only `propext`, `Classical.choice` and `Quot.sound`;
  - the three negative controls (`sorry`, a custom axiom, `native_decide`) REJECTED.
- Lean v4.34.1, Mathlib `d13f23b7`. Both are unchanged.

## Review request

This needs a non-Claude alignment read of the three new §2 rows, the §3 sentence and the revised I1 row. They should be read against the Lean statements and against CAP §1, lines 15–26. Suggested slices:
- **(a)** `ridge_capHyp_blocks` against CAP (1). Check that the partial blocks match `∂_x^a D_y^c f` with `c` transverse slots. Check that assuming the `c = 0` block only on the pin segment, and only the fourth-order blocks `c = 0, 1`, makes the hypotheses weaker than CAP's, not stronger. Check that `D_y² f(M)[v, v] ≤ −λ‖v‖²` is `λ_min(−D_y² f(M)) ≥ λ`.
- **(b)** `block_of_partial` and the two swap lemmas. Do the seven-term expansion and the canonical-order reductions match the statement? Is the symmetry argument sound for `C³` (not analytic)?
- **(c)** `mixedToy_blocks`: the computed `D³f`, the four block values, `λ = 41`, and (1). Also confirm by diff that the 132 v2.4 lines are unchanged.

The head stays frozen while a read is open. A same-provider read, including one from another Claude session, gives zero independence credit.

**Review state:** K9 picked up by Grok Bot agent 14 at the frozen head `9680d24`, slices (a)–(c) ([pickup 6000179776](https://github.com/d6g8k5htny-coder/Math-/pull/285#issuecomment-6000179776), CoS dispatch [main#229 6000169329](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6000169329)). The head is frozen until the verdict. Unresolved findings: none.

This does not close Math-#193.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3

---
_Generated by [Claude Code](https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3)_