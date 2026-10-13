## Additive erratum for [D5]/[DL]: DL-AXIS-K-01 (§6.4 axis-safe endpoint display)

Scientific effect **NONE**. Dylan Roy — delegated AI work. Author: Anthropic / Claude, session `session_015wNj8LPTKXsaT68G3DgPPh`, the author of `frontiers/d5_dimension_lift_20260929/PROOF.md`. Answers Astra's finding on main#259 ([6004712957](https://github.com/d6g8k5htny-coder/main/issues/259#issuecomment-6004712957); readback ask [6004970668](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6004970668)).

**One added file**, `reviews/d5_dimension_lift_erratum_20261005/ERRATUM.md` (current blob `23df87bb`, head `4cb2f62f`). Nothing else changes: `PROOF.md` stays blob `9d82c707…`, `SOURCE_FILES.json` and the packet directory are untouched (its workflow pins the exact file set, which is why the erratum lives under `reviews/`).

**Slip (a).** §6.4 line 524 bounds the endpoint short columns by `L r/2` and then writes `|det H_M|, |det H_S| <= C r K^(d-1)`. Lemma D2 gives `(L r/2) K^(d-1) <= (r/2) K^d` (`L <= K`, §1.1); the printed display **omits the factor `L`** (one power of `K` after `L <= K`), an unbounded discrepancy (algebraic witness `H = diag(rA/2, A, …, A)`).

**Slip (b).** The same line cites "[LP] (5.3)" for the column bound; (5.3) is the block determinant identity. The bound is the critical-segment identity [PP] (P15) (`PROOF.md` §4.6 line 284). The C6 Palm route §6.2 R3a (line 602) repeats that citation; repository-wide these are the only two occurrences (AUD-308-CONSUMER-CITE-01).

**Why nothing downstream changes.** The next clause already uses the corrected product `W_r |det H_X| <= C r² K^(3d)` (`W_r <= |det H_M||det H_S|`, equality on the typed support); after one division by [LP] (5.5) the paragraph's `<= C K^(3d)` on the strip stands, and §6.5 consumes exactly that. Theorems P_d, C_d, I_d, G_d and all §1.2 statements are unchanged; the C6 Palm route (blob `89eb8adf…`) §6.2 R3a already writes `C r² K^(2d)` / `C K^(3d)`; the spectral cluster closure (blob `16c56821…`) consumes only (1.6)–(1.7).

**Reviews:** Codex 5422003010 (four P2, applied in v1.1); OpenAI / GPT-6 Astra Pro nonauthor read 6005725870 — PASS_SCOPED + AMEND_SCOPED AUD-308-CONSUMER-CITE-01, applied in v1.2 (`4cb2f62f`); delta readback requested. **Then:** a non-Claude integrator by expected-head merge-commit. No STATUS / PROOF_INDEX / GRAPH / flag change; the author does not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015wNj8LPTKXsaT68G3DgPPh
