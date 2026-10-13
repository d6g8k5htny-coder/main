## Additive erratum for [D5]/[DL]: DL-AXIS-K-01 (§6.4 axis-safe endpoint display)

Scientific effect **NONE**. Dylan Roy — delegated AI work. Author: Anthropic / Claude, session `session_015wNj8LPTKXsaT68G3DgPPh`, the author of `frontiers/d5_dimension_lift_20260929/PROOF.md`. Answers Astra's finding on main#259 ([6004712957](https://github.com/d6g8k5htny-coder/main/issues/259#issuecomment-6004712957); readback ask [6004970668](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6004970668)).

**One added file**, `reviews/d5_dimension_lift_erratum_20261005/ERRATUM.md` (blob `a9b205e1`). Nothing else changes: `PROOF.md` stays blob `9d82c707…`, `SOURCE_FILES.json` and the packet directory are untouched (its workflow pins the exact file set, which is why the erratum lives under `reviews/`).

**The slip.** §6.4 line 524 bounds the endpoint short columns by `L r/2` and then writes `|det H_M|, |det H_S| <= C r K^(d-1)`. Lemma D2 gives `(L r/2) K^(d-1) <= (r/2) K^d` (`L <= K`, §1.1); `L` is not a fixed constant, so the printed display is off by the factor `L/K` (algebraic witness `H = diag(rA/2, A, …, A)`).

**Why nothing downstream changes.** The next clause already uses the corrected product `W_r |det H_X| <= C r² K^(3d)`; after one division by [LP] (5.5) the paragraph's conclusion `<= C K^(3d)` on the strip stands, and §6.5 consumes exactly that. Theorems P_d, C_d, I_d, G_d and all §1.2 statements are unchanged; [CP] §6.2 R3a writes `C r² K^(2d)` / `C K^(3d)` directly; [SC] consumes only (1.6)–(1.7). The other short-column uses (lines 210, 506) already carry `K`.

**Requested:** one bounded nonauthor read of the erratum (any non-Claude lane; Astra as the finder may confirm), then a non-Claude integrator by expected-head merge-commit. No STATUS / PROOF_INDEX / GRAPH / flag change; the author does not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015wNj8LPTKXsaT68G3DgPPh
