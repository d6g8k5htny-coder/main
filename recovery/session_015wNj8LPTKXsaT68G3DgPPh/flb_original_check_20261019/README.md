# Evidence for C8's ORIGINAL-correspondence check of the finite-L binding

This folder holds the check behind `FLB-ORIGINAL-CORRESPONDENCE-CHECK`. It was offered by Claude Code session `session_014KRmPPxsG978L4hgsx2kCy` in [main#419/6278343295](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6278343295) and picked up by Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8), in [main#419/6278469127](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6278469127). Dylan Roy — delegated AI review. The result is posted on main#419. Credit 0. Scientific effect NONE.

## Question

Do ORIGINAL's exact bytes match the definitions that the finite-L binding states for them? ORIGINAL is `independent-derivation.md` in Math-#193 comment 6242350213. The binding is `frontiers/original_finite_l_gaussian_binding_20261017/PROOF.md` in Math-.

014K's review covers the binding against GAD, plus numerics. It left this correspondence UNVERIFIED because that session could not read Math- issue comments.

## Files

| File | What it is |
|---|---|
| `extract_original.py` → `extract_log.txt` | Extracts ORIGINAL from the comment's REST `body` (the lines strictly between "```markdown" and the next "```", joined with LF, plus a terminal LF) and prints its bytes, lines, SHA-256 and git blob. ORIGINAL's bytes are not copied here; the public comment reproduces them. |
| `COMPARISON.md` | The item-by-item comparison: the law, the equations and the d ≥ 3 points. It was written before C8 read the packet's `REVIEW.md` (beyond line 25), the focused tensor review or 01Cz's input. |
| `tensor_check.py` → `tensor_check_log.txt` | Exact sympy checks, in d = 3 and 4, of the shared identities ORIGINAL (5)–(9) and (11), the continuum limit (30), and the constants behind (1), (22), (23) and (31). It also gives ORIGINAL's c_{2,∞} from (32) against 014K's C(2,24). Run with sympy 1.14.0. |

## Result

Every item in scope matches. The binding states ORIGINAL's law and equations exactly, in d = 2 and in d ≥ 3, and there is no finding. The posted result gives the effect on 014K's PASS_SCOPED and the notes.
