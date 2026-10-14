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

## Other manuscript lines (not used as the base)

- **The September edition.** This repository's branch `chatgpt/drive-github-hardening-20260919` also mirrors `PKG-03 — PREPEER_MANUSCRIPT_v2_SEP13_WITH_ERRATA`, a broader "pre-peer-review manuscript, second edition" of 13 September 2026, with errata (SHA-256 `3df5b5fc…`). It covers the SIDE24 3D track together with the 2D q0 program.
- **Why V3 does not use it.** The V3 edit pack is keyed to the 30 July manuscript and its V2 sheet, so V3 continues that line.
- **Its errata firewall still holds.** Errata §1 says that the ratified upper chain is the 3D SIDE24 theorem and must not be composed with the 2D lower wall into a two-sided 2D law. V3 respects this: planar results appear only as separate $d = 2$ remarks (§5.5), and no cross-dimension inference is made.
- **Its other errata do not apply here.** Errata §2 concerns that edition's campaign records, which V3 does not cite.

## Author decisions as drafted (defaults)

1. **Theorem 2.1 strength.** Theorem 2.1 = E10's two-provider form `c ℓ^{-1/3}(1+O(ℓ^{1/3}))`, proved in §3; the merged three-term law (E18/E19/E43) = Theorem 2.3; the later refinements (E60–E77) = Remark 2.4. Switches: put Theorem 2.3 into Theorem 2.1, or move it to the Discussion.
2. **Human review.** A disclosure (§7): "No independent human specialist has reviewed this work." (owner statement main#229 5973603003). The draft says nothing about the author's own reading; the pre-delivery check removed a sentence that did (the first bullet under "All findings applied", below).
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

## Revision r1 (13 October, about 18:20Z)

r0 is the PDF delivered first: 533,836 B, SHA-256 `8270e498e142f3cb94f1d1d45b9302b2518912593af2de4bf5478894d59d916e`. Git keeps it at commit `6deb0899` of main#389, and the author's archive keeps it too. r1 is this PDF, 19 pages.

**Commit 1 (`46474c0f`): the Slice A AMEND of Cursor (Grok), main#389 6152054602, applied exactly.**
1. Corollary 3.5: (3.13a) holds for `0 < r ≤ r_*`, with `c, r_* > 0` depending on the compact mark set. This is Theorem E1_d's scope.
2. The far-elder vanishing sentence cites [R9] instead of [R7].
3. The sentence after Proposition 3.12 is now the reader's text: Theorem R does not prove that the remainder tends to zero; the far elder density tends to zero; the rejected density tends to a positive constant.
4. §4: "the relative bound `|c_{3,24}/c_{3,∞} − 1| < 10⁻¹⁰⁶`" replaces "the enclosure of `c_{3,∞}` to 10⁻¹⁰⁶". This follows `coefficients/side24_v1/PROOF.md` (4).

**Commit 2: the four findings of the documentary consistency read of r0 (main#307 6151934937), assessed and all accepted, plus consequential fixes.**
1. §5.1 now writes the pair measure as `r⁻² dr dh`. With `dh = r³ dk` it becomes `r dr dk`, which is (3.10) with `k = κ/6`, and the pushforward is `r dr = ⅓k^{−2/3}ℓ^{−1/3}dℓ` (cf. (3.12)). The r0 text left out the height-gap Jacobian.
2. Remark 3.7: "rejected exactly when …" is restricted to the limiting pinned planar model. The transfer to the actual field is stated to hold only up to exceptional sets and under the conditions of §§5.3 and 5.5.
3. §5.2: "only through this cusp scale" is restricted to the order `ℓ^{1/3}`. Fold-scale rejection is stated to enter at `ℓ^{2/3}` through `R_{2/3}`.
4. Abstract: "for compact positive marks … in Palm probability" is added to the `O(r³)` selection statement.
5. Optional wording, accepted. §7 now reads "All nonauthor checks of the proof chain to date were AI checks …"
6. Consequential to AMEND item 4, so that the same record is described the same way everywhere:
   - the status box uses the relative bound;
   - Theorem 2.3(iii) states the twenty-digit enclosure `0.04177593184059834334 < c_{3,24} < 0.04177593184059834335` of [R5];
   - §7 says that `c_{3,24}` is enclosed to twenty digits;
   - [R5] describes both statements of the record.

**Pages of r1 whose text changed:** 1, 2, 5, 8, 10, 12, 13, 14, 16 and 17. Every other page differs only by reflow.

**Correction to the edit pack.** E8 item 3 called `coefficients/side24_v1` "the 10⁻¹⁰⁶ enclosure of `c_{3,∞}`". The record gives a twenty-digit enclosure of `c_{3,24}` and the bound `|c_{d,24}/c_{d,ref} − 1| < 10⁻¹⁰⁶`. This is recorded as E80.

## Revision r2 (13 October, about 19:40Z)

r1 is commit `56b680cd`: 19 pages, 536,244 B, SHA-256 `aa8667258dbf92d066d23e4ee22b76709f3a4246f4fa2da3a9033f9988c30077`. Cursor's byte-diff readback of `6deb0899..56b680cd` passed (main#389 6152358620). r2 is this PDF: 19 pages, 536,835 B, SHA-256 `b00c323ca827672c2072ffd257d957f33210fa04f4446fc6f620da66f215163b`.

**Commit 1 (`f94ab1d6`): the Slice B AMEND of Cursor (Grok), main#389 6153161213, applied exactly.** In §5.4, "A simulation over 8.4·10⁷ correlation lengths" replaces 8·10⁷. The Theorem 1D record gives a total length of 8.4·10⁷ (Math- `frontiers/d1_third_order_law_20261001/PROOF.md`, line 442).

**Commit 2 (`ece84b95`): the Slice C AMEND of Cursor (Grok), main#389 6153388424, applied exactly.**
1. [R3] reads "main issue 67". It read "main PR 67", but the xAI R1–R6 acceptances are comments on issue 67.
2. §0, the status row of Theorem 2.3:
   - the links in [R11], and `c₁` and the closed form of `c₂`, each have an OpenAI/Codex review and a Grok Bot read;
   - the torus transfer of `c₂` has an OpenAI `ACCEPT_SCOPED` of slices A–C and an Anthropic acceptance of slices A and C, and no Grok Bot read.
3. §7 makes the same two points. It also says that Theorems U, K, P and Z in Remark 2.5 rest on records with one or two providers' reviews, and that `B_{3,24} ≈ 10.2` is exploration only.

**Commit 3: author corrections to this change list.** No draft text changes.
- The r1 heading said "about 18:40Z", but r1 was pushed at 18:20Z.
- The description of decision 2 still quoted the sentence that the pre-delivery check removed ("The author has reviewed it throughout"). It now describes the draft's actual wording.

**Where r2's text changed:** §0's status table (page 2), §5.4, §7 and [R3]. The longer status row moves the page breaks. The text changes are on page 2, in §5.4 (page 15), in §7 (page 16) and in [R3] (page 17); the other pages differ only by reflow.

## Revision r3 (14 October, about 18:30Z)

r2 is commit `ff73d996`: 19 pages, 536,835 B, SHA-256 `b00c323ca827672c2072ffd257d957f33210fa04f4446fc6f620da66f215163b`. Cursor's byte-diff readback of `e4aca581..ff73d996` passed (main#389 6154948011). r3 is this PDF: 19 pages, 542,201 B, SHA-256 `7549678738ee73b913943c0f652bb9e2ec9c70cd723d6b29ec83a002078fcf75`.

r3 applies the author's dispositions in main#389 6155958479: items 1, 2, 3 and 5. It runs under claim `R3-389-01N-20261014` (main#307 6175707479), on the owner's 18:42Z goal "Continue to advance the project. Resolve any issues. Help any and all agents in whatever ways possible." The request to the coordinator is main#307 6175251175. Item 4 needs no main-text change. The three owner decisions keep their drafted defaults. Item 1 changes the proof route of (3.13), so r3 needs a substantive nonauthor read of the changed passages.

**Item 1, §3.4 (selection).**
- **The proof of Corollary 3.5** cites Theorem A of [R2] for (3.13), with that theorem's own scope: every fixed d ≥ 2 and L > 0, uniform over compact birth and gap windows and over all frames, with existential constants (Math- `reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md`, §2). The statement of the corollary does not change.
- **The witness argument** is now described as a second, three-dimensional route that the proof does not use. Its two gaps are stated: the singular-near envelope near c = 0, and five local repairs in the proof of Theorem 3.6.
- **Lemma 3.4** names same-maximum self-attachment. A second half-branch ending at a different maximum in the same component of {f > h} gives a merge saddle with value in (h, b), so that case lies in B_r. The three regions are measured from the midpoint x.
- **(3.17)** is stated uniformly on compact positive mark sets, without the Gaussian factor.
- **§0 and §7** record that (3.13) is Theorem A of [R2].

**Item 2, §3.8.** Two sentences say that the bound 10⁻²¹⁶ in (2.10) is proved in [R4], and that (3.26)–(3.33) give a self-contained argument for the weaker bound 10⁻¹⁸⁰. No display changes.

**Item 3, Theorem 2.3.**
- (ii) gives the definition of c₂ from [R11] (Theorem T, (0.1)): the Hadamard finite part at k = 0 of (1/3)∫A₂k^{−4/3}.
- (iii) adds c₂/c_{3,24} ∈ [3.859496174547456, 3.859496174548331]. I rechecked it in exact rational arithmetic: the c₂ enclosure, widened by the 10⁻⁶⁰ transfer error, divided by the twenty-digit enclosure of c_{3,24}.

**Item 5, §5.1.** The cusp variable ℓ/r⁴ is now ϑ, with ϑ = κ/(6r). The records write it κ. κ keeps its §3 meaning.

**Author correction to this change list.** The Build section said 18 pages. r1, r2 and r3 are 19 pages.

**Where r3's text changed:**
- the §0 status table (page 2);
- Theorem 2.3 (pages 5–6);
- §3.4 (pages 8–10);
- §3.8 (page 12);
- §5.1 (page 14);
- §7 (page 17).

The other pages differ only by reflow. The build has one overfull line of 2 pt, as before.

## "Do not print" compliance (checked item by item)

No numerical ε₂₄ (E1). No `O(ℓ)` for the far density, and the far rejected density is not called vanishing (E4). No numerical selection-loss constant (E2, E9). No "first" or "absent from the literature", and Pérez is not said to call the smooth law open (E5). No "1% at ℓ ≈ 2.3·10⁻⁵" (E15). The simulation is an illustration, and its residuals are not called an ℓ^{1/2} term (E17, E43). `B_{3,24}` and `I^{cand}` are not certified digits (E19). No sharpness of 3/4, and no d ≥ 2 edge statement (E20). Proposition 2′ of #242 is not stated about the field (E22). No planar result is called d = 3, and the planar chain is not called independently reviewed (E23). The d = 3 rate carries its conditionality (E49–E58). No C104 quotient line (E25). "Certified" carries #219's and #223/#232's scope (E43). [K]'s normalized cap bound is not stated (E50). PD_U is not cited (E59). δ is not called sharp, and the elder ℓ^{2/3} statement stays o(ℓ^{2/3}) (E76).

## Build

`pdflatex main && bibtex main && pdflatex main && pdflatex main` with TeX Live 2023 (amsart, natbib-free amsplain). The sources are `main.tex`, `s0_status.tex` … `s7_disclosure.tex`, `appA_records.tex` and `refs.bib`. The output is 19 pages, with one 2 pt overfull line. Section "Status of this draft" is for the author and is to be removed before submission.
