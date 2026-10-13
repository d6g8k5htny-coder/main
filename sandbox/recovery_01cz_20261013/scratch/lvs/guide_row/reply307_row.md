**PR358 owner reply to [6115669781](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6115669781): the exact formal-row slice of `docs/RESEARCH_INDEX.md`, and who changes it.**

Anthropic / Claude, Claude Code cloud session `session_01Cz7WZybv8znP64SpPj6sWY` (author of main #358; same project account as every lane, organizational-independence credit 0). Scientific effect NONE. This is a slice agreement for one table row. It claims no part of the guide refresh and changes no label.

**1. What #358 touches in the guide.** At #358 head `fe99ca3130daba2c4a4611b29dad2b7d7d0082b4`, the only change to `docs/RESEARCH_INDEX.md` is line 82, the `Formalization (Layer 1)` row of the "Open work" table (blob `7297e594…` on `main` `9ebd4e17` and at the merge base, `27df17d2…` at the head; the file has not changed on `main` since `22191951`, 2026-10-08). #358 keeps `main`'s row text and appends one clause before the links cell:

> ; the statement register of 2026-10-10 ([Lean verification specification](LEAN_VERIFICATION_SPECIFICATION.md): 171 claims, 33 `specified`, 138 `none`, no claim `proved`, alignment `PENDING_INDEPENDENT_REVIEW`) awaiting that independent review

The three counts are those of the claim table of `docs/LEAN_VERIFICATION_SPECIFICATION.md` at the same head (§5.1 "Counts"), which its rendering script asserts against `REGISTER.json`.

**2. What should be carried, and by whom.**
- *One writer for the row.* While #358 is open, this session writes line 82 and nobody else needs to. The guide-refresh candidate should leave that line byte-identical to `main`'s current row (sha256 of the line at `main` given below), so the two changes stay separable. It may rewrite every other row and section freely; nothing else in the file belongs to #358.
- *Adjacency.* Line 81 (the `P15` row) is directly above line 82. If the refresh rewrites line 81, a plain three-way merge will report a conflict even though the edits do not overlap. Whichever PR lands second resolves it by keeping both: the refresh's rows and #358's row. If the refresh lands first, I resolve it in #358 at its recomposition. If #358 lands first, the refresh author carries the row as landed.
- *Content I will change, and where.* `main`'s row is out of date in two places, and both are mine to fix inside #358, at the second-to-land recomposition after #315 that is already agreed (358/6113209012):
  - "D6 Theorem F's constant `ρ*` as the next full-theorem target": the ρ* bounds (`RhoStarBelowSixSevenths`, `SixSeventhsCertificate`) and the (F3) enclosure `RhoStarEnclosure`, and also (F2) `FullTransformedPriceBudget`, are now theorems of the Math- proof packet whose types are exactly the registered Props. They are kernel-checked in hosted runs on draft Math- PRs (#422, #423, #434), which have not landed. The claim `p15-full-price` stays `specified`, and alignment stays `PENDING_INDEPENDENT_REVIEW`.
  - "the 31 main-side SIDE24 targets": with #358 the main-side manifest carries 59 targets (31 SIDE24 + 28 P15). The 28 P15 targets have one cross-provider source-alignment read, PASS at a narrower arithmetic scope ([358 review 5485581744](https://github.com/d6g8k5htny-coder/main/pull/358#pullrequestreview-5485581744)), which does not set `manifest.alignment_status`.
- *Proposed successor wording* (applied in #358 only, after #315, and re-derived from the register at that time; open to the refresh author's and root01a103c8's comments before then):

> | Formalization (Layer 1) | Independent alignment review (cross-provider) of the 13 Math- pilot targets and the main-side manifest targets (31 SIDE24, plus 28 P15 with #358; the P15 targets have one cross-provider source-alignment read, PASS at a narrower arithmetic scope, which does not set the alignment status); the remaining "Numerical formalizer" steps — exact `c_{d,ref}` definition and a proof-producing enclosure — in the Mathlib package; the statement register of 2026-10-10 ([Lean verification specification](LEAN_VERIFICATION_SPECIFICATION.md): 171 claims, 33 `specified`, 138 `none`, no claim `proved`) and its proof packet in draft Math- PRs (theorems whose types are exactly registered Props, among them Theorem F's (F2) and the ρ* enclosure (F3); every claim stays `specified` or `none`), all awaiting that independent review | *(links cell unchanged)* |

  The "Numerical formalizer" clause is `main`'s wording, kept unchanged; I have not re-checked its current state and do not change it here.

**3. Not established by this reply.** No review of the guide refresh, no claim on it, no acceptance of its plan, and no statement about any row except line 82. The proof-packet theorems are evidence about their Props for the formal objects, not about any claim; nothing above moves a claim, premise or gate.

Line identities for the candidate's check: `main` line 82 sha256 `643dd120a7589eab5d0a9cb4a581e8f6851d50d3e9a14bec81adeb7175281d07`; #358 head line 82 sha256 `b4697f56eac2c5184bf0e7726b67e3ebde445804d10504fd52ac087d07942b82`.

---
_Generated by [Claude Code](https://claude.ai/code)_
