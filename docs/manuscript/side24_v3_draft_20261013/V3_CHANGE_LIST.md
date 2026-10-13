# SIDE24 manuscript — V3 draft change list (13 October 2026)

Dylan Roy — delegated AI work. Actual performer: Anthropic Claude, Claude Code session `session_01NMeKEismAyeqgdB4sy2NJU` (Claude01N), the author of the V3 edit pack. Author-side draft; scientific effect NONE. Nothing here changes a theorem; every statement is the manuscript's own (V1/V2) or an edit-pack item at its stated scope.

## Inputs (exact identities)

| input | identity |
|---|---|
| V1 source `SIDE24_MANUSCRIPT_CORE.md` (= the V1 manifest's `03_SIDE24_MANUSCRIPT_SOURCE.md`) | 21,550 B, SHA-256 `fb576de64f17f64a23423b185e2abc902a436d3a6a1bbb2aee04d756ad371183`; main branch `chatgpt/drive-github-hardening-20260919` @ `a01c72f1`, blob `99aea93b` |
| V1 bibliography `08_REFERENCES.bib` | 2,915 B, SHA-256 `3feb3afe84e1aaf1e49520cf9283f3b169f6401276b5f2a3ab6d3cc9827e3942`, blob `06970b5d` |
| V2 amendment sheet (M1–M7) and integration sheet | `SIDE24_v2_2026-08-01/build/03.html`, `MANUSCRIPT_V2_INTEGRATION.md` |
| Gap-fill supplement | `SIDE24_gap_fill/SIDE24_GAP_FILL_SUPPLEMENT.md`, 48,972 B |
| V3 edit pack | `V3_EDIT_PACK_20260930.md` (archive V2_2), items E1–E78, last entry 13 Oct 15:52Z |
| Literature delta | main#275 comment 6146291432 (C273, two independent searches, 13 Oct) |

## Author decisions as drafted (defaults)

1. **Theorem 2.1 strength.** Theorem 2.1 = E10's two-provider form `c ℓ^{-1/3}(1+O(ℓ^{1/3}))`, proved in §3; the merged three-term law (E18/E19/E43) = Theorem 2.3; the later refinements (E60–E77) = Remark 2.4. Switches: put Theorem 2.3 into Theorem 2.1, or move it to the Discussion.
2. **Human review.** A disclosure (§7): "No independent human specialist has reviewed this work. The author has reviewed it throughout." (owner statement main#229 5973603003).
3. **Planar material.** Discussion only (§5.5).

## Placement of the V2 amendments

| V2 item | V3 location |
|---|---|
| M1 density convention | Convention 2.2 (with #191's density-version qualifier, E43/E44) |
| M2 Palm law, elder mark | §3.1, Definition 3.1 and Lemma 3.2 |
| M3 trichotomy, Cor. A.2.2 | §3.4, Lemma 3.4 and Corollary 3.5 (3.13) |
| M4 near pushforward, limit interchange | Proposition 3.3 (§3.3), Theorem 3.9 (§3.5) |
| M5 off-diagonal pairs | Proposition 3.10 as A.3.2′ (sup bound); the July `O(ℓ)` is withdrawn (E4) |
| M6 capture/escape | Theorem 3.6 (§3.4), with the normalized potential (3.18a) |
| M7 first-variation provenance | §3.8, paragraph after (3.31) |
| Appendix-level items (6, 8–10, 12a) | stay in the technical appendix; the B.1 sign corrections (E11) are Remark 3.8; the closed form of D₂ is cited in §3.7 |

The V2 integration sheet keys M2–M7 to section numbers ("§3.1 Palm framework", "§3.3 elder selection", "§3.6–3.7 capture/escape") that do not match the V1 PDF, whose §3 runs 3.1 ordered pairs … 3.8 correction. V3 places each amendment by content.

## Placement of the edit-pack items

| item | V3 location / treatment |
|---|---|
| E1 | abstract, (2.10), §3.8 (both bounds, the weaker stated) |
| E2 | Corollary 3.5 (3.13a) and the sentence after it |
| E3 | Remark 2.5 (with the optional `B_{3,24} ≈ 10.2` sentence as exploration) |
| E4 (with the 3 Oct revision and C89) | Proposition 3.10 and the paragraph after it |
| E5, E47, E48 | §6, together with C273's additions (Feldbrugge et al., Azaïs–Delmas, Ladgham–Lachièze-Rey, Bobrowski–Borman, Chenavier–Hirsch, Hirsch–Lundbye–Otto, Pérez) and its Chazal–Divol correction |
| E6, E7 | §7; the cover letter itself is not yet updated |
| E8 | draft-status box and §4 |
| E9 | Remark 3.7 |
| E10 | Theorem 2.1 (2.1) and Proposition 3.12 (§3.9) |
| E11 | Remark 3.8 |
| E12, E13, E15, E18, E19, E43, E44 | Theorem 2.3 and Remark 2.5 |
| E14 | §5.1 (last sentence in E15's form) |
| E16, E20, E45, E46, E49 | §5.4 |
| E17, E66 | §5.2, "Size of the corrections" |
| E21 | superseded by notes TL/TS (Remark 2.4(a)); not printed separately |
| E22 (with the #243/#244 revisions) | §5.2, soft pairs |
| E23, E26–E39, E58, E59 | §5.5 |
| E24, E49–E58 | §5.3 |
| E25 | Remark 3.11 |
| E40, E42 (replacement bars) | not used in this draft (optional Discussion remark) |
| E41 | no main-text insert (as the pack says) |
| E60–E65 | Remark 2.4 (a), (b) |
| E66–E75, E77 | Remark 2.4 (c) (DD cited from `main`; v1.5 once Math-#453 lands, no number differs) |
| E76 | Remark 2.5, (2.17) with δ = 1/50 |
| E78 | nothing |

## Corrections found while drafting

1. **E1's "relative precision 10⁻¹⁰⁰".** With the stated bound `|ε₂₄| < 10⁻²¹⁶`, the ratio to the correction `1.4862…·10⁻¹¹⁸` is `6.73·10⁻⁹⁹`, so V3 says "better than 10⁻⁹⁸". The figure 10⁻¹⁰⁰ holds only with the stronger bound 10⁻²¹⁸.
2. **"Planar" in V1.** V1 used "planar Bargmann–Fock value/jet" for the non-periodized field on ℝ³. V3 writes "the value on ℝ³" (§3.7) and "the jet of e^{−|z|²/2}" (§3.8), so that "planar" refers only to d = 2.
3. **Chazal–Divol.** V1 described their result as applying to "broad classes of random filtrations". Per C273 it covers random point-cloud filtrations and Brownian sublevels; V3 says so.
4. **Novelty wording.** V3 adds "Substantial constituent mechanisms — the fold, the close-pair correlation, the selection kernel — are prior art" (C273-L3), keeping "not located".
5. **`B_{3,24}` ratio.** E3 says "about 250 times `c_{3,24}`"; `10.2/c = 244.2`, so V3 says "about 240".
6. **Klein–Agam.** The V1 bibliography gives the first author as "Agam Klein"; arXiv:1111.5286 lists Avraham Klein. V3 corrects the entry.

## Pre-delivery check (13 Oct, about 17:50Z)

A fresh read-only checker that had not seen the drafting compared every display, statement and number with V1, the V2 sheet, the supplement, the edit pack, note C3-R and C273, and checked the Appendix A paths and hashes against Math- `main` `b68d88b3`. It is the same provider as the drafter, so it adds no independence.

- **Verdict:** PASS WITH FIXES. No blocker; 6 major and 25 minor findings. Every V1 display matches apart from the intended changes, all derived numbers check, and every record path, blob and merge hash exists.
- **All findings applied:**
  - the disclosure keeps only "No independent human specialist has reviewed this work" (the draft no longer asserts anything about the author's own reading);
  - the Dimension-one literature attribution is corrected (Rice and CLH: heights of maxima; Lindgren 1972 and Lindgren–Rychlik: maximum-to-minimum amplitude);
  - the status labels in §§5.3–5.5 and Remark 2.5 follow E49, E56, E58 and E59 (read, author-side, the conditional chains named), and §7 covers Remark 2.5, Corollary 3.5's lower bound, Remark 3.7 and Remark 3.11;
  - the abstract keeps V1's "by only O(r³)";
  - Theorem 2.3(iii) is reworded as enclosures with their scopes;
  - the far-elder paragraph carries E4's status wording and dates the withdrawn O(ℓ) to the 1 August amendment;
  - Proposition 3.10 is "the bounded part of Supplement A.3.2";
  - M7's provenance sentence is restored in full;
  - Theorem 3.6's hypotheses and (iii) follow the supplement;
  - Corollary 3.5 now precedes (3.14), and the near/far cutoff is ϱ (ρ stays the collar variable);
  - "cancel" is replaced by "rejected mass";
  - Convention 2.2 covers all three density versions;
  - the scope sentence is per record;
  - the simulation wording is corrected (connectivity by ascending lines, χ² window [10⁻⁴, 3·10⁻²], "not evidence for an ℓ^{1/2} term");
  - the §5.3 coefficient α⁽³⁾ is identified and C⁽³⁾_fail defined;
  - the 1-D coefficients are said to be those of the lifetime density;
  - compact windows are stated;
  - the Chazal–Divol, Klein–Agam and Chenavier–Hirsch descriptions follow C273;
  - the computer-assistance paragraph is corrected;
  - the [R18], [R20] and [R21] locators are added;
  - the bibliography is corrected (Rice, Cartwright–Longuet-Higgins doi, Lindgren 2019 pp. 100–128, Lindgren–Rychlik doi, Klein's first name).

## "Do not print" compliance (checked item by item)

No numerical ε₂₄ (E1). No `O(ℓ)` for the far density, and the far rejected density is not called vanishing (E4). No numerical selection-loss constant (E2, E9). No "first" or "absent from the literature", and Pérez is not said to call the smooth law open (E5). No "1% at ℓ ≈ 2.3·10⁻⁵" (E15). The simulation is an illustration, and its residuals are not called an ℓ^{1/2} term (E17, E43). `B_{3,24}` and `I^{cand}` are not certified digits (E19). No sharpness of 3/4, and no d ≥ 2 edge statement (E20). Proposition 2′ of #242 is not stated about the field (E22). No planar result is called d = 3, and the planar chain is not called independently reviewed (E23). The d = 3 rate carries its conditionality (E49–E58). No C104 quotient line (E25). "Certified" carries #219's and #223/#232's scope (E43). [K]'s normalized cap bound is not stated (E50). PD_U is not cited (E59). δ is not called sharp, and the elder ℓ^{2/3} statement stays o(ℓ^{2/3}) (E76).

## Build

`pdflatex main && bibtex main && pdflatex main && pdflatex main` with TeX Live 2023 (amsart, natbib-free amsplain). The sources are `main.tex`, `s0_status.tex` … `s7_disclosure.tex`, `appA_records.tex` and `refs.bib`. The output is 18 pages, with one 2 pt overfull line. Section "Status of this draft" is for the author and is to be removed before submission.
