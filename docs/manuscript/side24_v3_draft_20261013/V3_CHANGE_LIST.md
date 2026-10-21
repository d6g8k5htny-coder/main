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

**Item 5, §5.2.** The cusp variable ℓ/r⁴ is now ϑ, with ϑ = κ/(6r). The records write it κ. κ keeps its §3 meaning.

**Author correction to this change list.** The Build section said 18 pages. r1, r2 and r3 are 19 pages.

**Where r3's text changed:**
- the §0 status table (page 2);
- Theorem 2.3 (pages 5–6);
- §3.4 (pages 8–10);
- §3.8 (page 12);
- §5.2 (page 14);
- §7 (page 17).

The other pages differ only by reflow. The build has one overfull line of 2 pt, as before.

## Revision r4 (14 October, about 21:10Z)

r3 is commit `7aff36d0`: 19 pages, 542,201 B, SHA-256 `7549678738ee73b913943c0f652bb9e2ec9c70cd723d6b29ec83a002078fcf75`. Cursor's readback of `13c1868d..7aff36d0` passed (main#389 6176065794). r4 is this PDF: 21 pages, 555,513 B, SHA-256 `3fb9a94e1e919681481ba0c71781c1fc77ffa45fb4f7183797152d21f987d5e8`.

r4 runs under claim `R4-389-01N-20261014` (main#307 6178533667), on the owner's goal of 14 October. It has three parts.

**Part 1: the Supplement replaces the 1 August numbering.** [R1] is now the V3 supplement, checkpoint draft 4: branch `claude/side24-v3-supplement-draft-20261013` at `c6aa790d`. Its nonauthor read of §S5, Lemma S4.29 and Proposition S8.13 is main#389 6178514413 (Cursor, PASS, no AMEND). Every citation of the 1 August numbering now points to a Supplement statement:
- Definition 3.1 → S5.1–S5.2; Lemma 3.2 → S5.3 and S5.4; Z_r > 0 → S2.9(i); nondegeneracy → S2.3(i).
- Proposition 3.3 → S6.1; Lemma 3.4 → S5.7 (with S5.6); Theorem 3.6 → S5.31 (r_mat: (S5.94); Q_⊥: (S5.95)).
- Theorem 3.9 → S6.3, with its hypotheses and 0 < c < ∞ from S6.4; Proposition 3.10 → S6.2; the far rate O(ℓ^{2/3}) → S8.13.
- The contact limits → S2.8 and (S3.27); D_t → S3.16; (2.10) → S4.4; J^(1)_near → S4.22; (3.29) → S4.29; the first variation → S4.24, S4.25 and S4.26.
- Remark 3.8 → S5.15; §7's elimination → S5.15; the status table → "Supplement Part I".
- Two citations keep the 1 August numbering on purpose, as history: the far bound O(ℓ) of the 1 August amendment ("its Proposition A.3.2"), and the display (B.1) of 1 August in Remark 3.8.

**Part 2: §3.4 is aligned with the Supplement.** The witness route is now "written out in full in Supplement §S5 (Theorem S5.36)". The two gaps that r3 stated are closed there:
- Lemma S5.22 proves the envelope (3.16) for every angle. Lemmas S5.19 and S5.23 give the collar and singular-near counts.
- §S5.5 contains the five local repairs to the proof of Theorem 3.6.

The proof of Corollary 3.5 still cites Theorem A of [R2]; the witness route remains a second argument. Lemma 3.4's explanatory sentence now covers a half-branch that ends at any critical point, not only at a maximum (Lemma S5.7, Case 0).

**Part 3: findings of a same-provider referee read of r3.** A read-only subagent of the author's session read r3 as a referee would and listed 20 findings, F1–F20. It found no arithmetic, power-counting, sign or exponent error in §3. It earns no independence credit. Each finding was checked before it was applied:
- **F1 (major): relation to [R2].** It is now owner decision 4 (§0). The default is a new §6 paragraph, "Relation to the parent record". It states which results are cases of [R2] (Theorem C, (15.2), Theorem A, (14.1)) and which are specific to the side-24 field. It also says that the Supplement imports some statements from [R2], in its Sections S7, S8 and S10.
- **F2: Definition 3.1** now defines the elder mark through the maximin level, e = 1{d_f(M) = f(S)}, as Supplement S5.2. The unused 𝔪_g and 𝔟_g are removed. Lemma 3.2 states the equivalence with the elder-pair condition.
- **F3: definitions and missing steps.**
  - K^all_r = p_Π(z_r) Z_r is defined in Lemma 3.2, and G^all_r = r⁴K^all_r in Proposition 3.3.
  - The limit G^all_0 is displayed as (3.19a), with the contact limits r⁻¹det H_M → −κ det Q_t and r⁻¹det H_S → κ det Q_t. This explains κ^{4/3} = κ^{−2/3}·κ² in (3.21).
  - (3.19) holds for all 0 < r < ϱ, and (3.12a) integrates r over (0, ϱ) (also F10).
  - 0 < c_{3,24} < ∞ is cited to S6.4.
- **F4: (3.28).** J^(1)_near includes the images' share 6e^{−288} of the normalizer (Supplement S4.22). Without that share, the first variation would be off by 2e^{−288}. The variations δa, δm₄, δχ are now stated per unit e^{−L²/2}. Rechecked: DF(J₀)[J₀] = −1/3, since F(aJ₀) = a^{−1/3}.
- **F5: §3.8 "self-contained" is withdrawn.** (2.10) cites Supplement Theorem S4.4. (3.26)–(3.33) are "a shorter route": (3.29) is Supplement Lemma S4.29, while (3.27) is quoted from the technical appendix and not reproved. R_quad is defined. The undefined F(C_u) representation is reworded.
- **F6: §3.4.** Done in Part 2.
- **F7: s2:69.** "Proved" now applies to Theorem 2.3 only. Remarks 2.4–2.5 are reported "at the review status and scope stated with each".
- **F8: §5.4.** The amplitude law is identified as the one-dimensional candidate law. Its second- and third-order terms nearly cancel: relative coefficients −1.36 and +2.09, net below 0.5% on [10⁻⁴, 10⁻²]. Those of the lifetime density (−2.07, +2.09) cancel only in part: net −0.5%, −1.6% and −4.4% at 10⁻⁴, 10⁻³ and 10⁻². Recomputed in floating point.
- **F9: §5.2.** The aggregate agreement over [10⁻⁴, 10⁻²] is separated from the bin-by-bin χ² over [10⁻⁴, 3·10⁻²]. The simulated sides (16 and 64, not 24) are named, with the size of the periodization effect.
- **F11: s1.** "Uniformly in x ≠ y" becomes "uniformly on that set" (dist ≥ δ).
- **F12: Remark 2.5.** It points to Corollary 3.5's compact-mark restriction. Candidate pairs are "ordered typed pairs at every separation".
- **F13: Remark 3.8.** It uses Lemma S5.15's variables (ρ is the sine, not s), and "the κ-free term of N_y" replaces "det H_P". The variables of (3.14)–(3.16) are defined.
- **F14: symbols.**
  - The two-point kernel of Theorem 2.3 is 𝒜_r, distinct from the adjacency event A_r, with 𝒜_r(b, κ/6, t) = G^all_r(t, b, κ) (Supplement S6), and k = κ/6 is stated.
  - Other renamings: §5.1's gap h becomes ℓ; the test function φ becomes ψ; §3.7's difference coordinate D becomes Δ.
  - Defined: Q_⊥, r_mat, R_quad, λ, β (§5.3) and the inverse-moment threshold (Remark 2.5).
  - F keeps its two local meanings, (3.19a)–(3.20) and (3.25), as in Supplement S3 and S4.
- **F15: (3.24).** The GOE normalization is stated.
- **F16: cross-references.** §3.7 is cited for (2.11), and Convention 2.2 points to Proposition 1.2 and §§3.1–3.3.
- **F17: §4.** The statement names the two places that go beyond d = 3, L = 24: L ≥ 24 in Theorem 2.3(iii), and P_d(L) in §3.8.
- **F18: proof of Corollary 3.5.** "joined, within {f > f(S)}, to a component with an older birth maximum".
- **F19: Theorem 2.3(iii).** The enclosure of the integral (2.13) and the closed form of c₂ are unconditional. The reading of the integral ratio uses Theorem CU (Supplement Theorem S9.2 and Remark S9.17); the enclosure itself is Corollary S9.16. The transfer of the c₂ expression is Corollary S11.10. Its identification with the constant of (2.12), and the ratio c₂/c_{3,24}, use the two interfaces (Hypothesis S11.11; Corollary S11.12).
- **F20: Remark 2.5** now names Theorems U, K (with Corollary T), Z and P, as §0 and §7 do.

**Literature (§6).** A new paragraph, "Universality of persistence ratios", cites three works:
- Bobrowski–Skraba, Sci. Rep. 13 (2023) 12274 (arXiv:2207.03926), the conjecture;
- Bobrowski–Skraba, arXiv:2406.05553, the weak form;
- Heo, arXiv:2610.08257 (6 October 2026), the failure of the strong form.

The paragraph says that these concern point clouds, positive degree and a multiplicative statistic. Source: RLOOP's handoff (main#275 6177725462) and TASKWATCH's source-fidelity check (main#275 6177892575). The titles and authors were checked on arXiv, and the Sci. Rep. data on an institutional listing of the article.

**Where r4's text changed:** §0, Theorem 2.3 and Remark 2.5 (§2), §§3.1–3.8 (almost every subsection), §4, §§5.1–5.4, §6, §7 and Appendix A ([R1]). The build has one overfull line of 2 pt, as before.

## "Do not print" compliance (checked item by item)

No numerical ε₂₄ (E1). No `O(ℓ)` for the far density, and the far rejected density is not called vanishing (E4). No numerical selection-loss constant (E2, E9). No "first" or "absent from the literature", and Pérez is not said to call the smooth law open (E5). No "1% at ℓ ≈ 2.3·10⁻⁵" (E15). The simulation is an illustration, and its residuals are not called an ℓ^{1/2} term (E17, E43). `B_{3,24}` and `I^{cand}` are not certified digits (E19). No sharpness of 3/4, and no d ≥ 2 edge statement (E20). Proposition 2′ of #242 is not stated about the field (E22). No planar result is called d = 3, and the planar chain is not called independently reviewed (E23). The d = 3 rate carries its conditionality (E49–E58). No C104 quotient line (E25). "Certified" carries #219's and #223/#232's scope (E43). [K]'s normalized cap bound is not stated (E50). PD_U is not cited (E59). δ is not called sharp, and the elder ℓ^{2/3} statement stays o(ℓ^{2/3}) (E76).

## Build

`pdflatex main && bibtex main && pdflatex main && pdflatex main` with TeX Live 2023 (amsart, natbib-free amsplain). The sources are `main.tex`, `s0_status.tex` … `s7_disclosure.tex`, `appA_records.tex` and `refs.bib`. The output is 21 pages (r4), with one 2 pt overfull line. Section "Status of this draft" is for the author and is to be removed before submission.

## Revision r5 (14 October, about 23:10Z)

r4 is commit `d4740a50` (21 pages, 555,513 B, SHA-256 `3fb9a94e…`), with its read AMEND applied in `7bab515f` (21 pages, 555,544 B, SHA-256 `a22db882acf787ade47a17b3c3b0c4efb2e4fc351091dc5aa89dfbb9e473de18`). Cursor's readback of `d4740a50..7bab515f` passed (main#389 6179449309). r5 is this PDF: 22 pages, 557,655 B, SHA-256 `95960dc58f0ed258b41935a3700ac9f67adb71430a386b3d7fc32d7adc919e6b`.

r5 follows the Supplement's checkpoint drafts 5–7, which add Part III. It has five parts.

**Part 1: [R1] and [R2].**
- [R1] is now checkpoint draft 7 at `9d8fc85b`. Its Part III (§§S12–S13) transcribes the parent record in its reconciled reading. §§S1–S7 and §§S12–S13 have had nonauthor slice reads, and §§S8–S11 are being read; the PR body of main#389 lists the verdicts.
- [R2] now names the whole reconciled identity set, as the reconciliation record asks of every consumer: the parent `UNIFORM_MATRIX_CAP_AND_LIFETIME.md` (`dfed3b8d`) with its cap proof `MARKED_CYLINDER_CAP_PROOF.md` (`0633aca3`), read as reconciled in `reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md` (`75da2597`): with the congruence erratum `ERRATUM_CONGRUENCE.md` (`213594d6`), with §9 replaced by `reviews/d1_section9_borel_repair_20260925/REPAIR.md` (`fe9b9ce4`), and with the wording W1 and the embedding radius of that record's §5. All four blobs were checked at Math- `b68d88b3`, the commit the appendix names.

**Part 2: decision 4 and §6.**
- §6, "Relation to the parent record": the Supplement no longer imports statements from the record. Part III states them with their proofs, in the reconciled reading; where the record only sketches a step, the step is given as sketched there or written out in full. For the leading term, Supplement Theorem S5.36 and Proposition S6.2 prove (3.13) and Proposition 3.10 without the record.
- §0, decision 4: the default now includes Part III. The alternatives are restated:
  - cite Supplement Theorem S5.36 and Proposition S6.2 in the proofs of Corollary 3.5 and Proposition 3.10, so that the leading term rests on the Supplement alone; or
  - publish the parent record as a companion preprint and cite it in place of Part III.
- §0, status table: the leading-term row adds "transcribed in Supplement Part III".

**Part 3: Theorem 2.3(ii).** "c₂ is the fold-scale finite part of the two-point contact kernel" now reads "… of the r²-coefficient of the two-point kernel". The contact kernel is 𝒜₀, and c₂ comes from 𝒜₂ (S11 concordance, issue 1).

**Part 4: the c₁ erratum.** The records entry for the certified constants now names the erratum of the c₁ packet, `ERRATUM_20261013.md` (blob `a9c3f95a`; Math- PR 454, `a6116d8a`, landed after `b68d88b3`). It says that the enclosure is of the ratio of the integral (2.13) to c_{3,24}, and that its reading as a coefficient ratio is conditional. Theorem 2.3(iii) already says so.

**Part 5: §6, Heo.** "Heo, in a preprint, computes …". The bibliography already gives the arXiv number. This matches the attribution of main#407's novelty addendum.

**Build.** `build.sh` compiles with no LaTeX error and the same 2 pt overfull line as before.

## Revision r6 (14 October, about 23:35Z)

r5 is commit `b86271a5`: 22 pages, 557,655 B, SHA-256 `95960dc5…`. Its read AMEND (main#389 6180733414) added the third amendment of the reconciled parent to [R2]: the wording W1 and the embedding radius. That AMEND was applied exactly in `74bc4988` (557,677 B, SHA-256 `0d8113f7…`), and its readback passed (6180780388). r6 is this PDF: 22 pages, 557,728 B, SHA-256 `602762a235b5fabe388c1cb7bc5025b55b1807c193ae02a06b2eb76a63a466f2`.

r6 records that the Supplement's slice reading is complete. Every section, §§S1–S13, has had one bounded nonauthor read (Cursor, Grok; same account). Each requested change was applied exactly and read back. The Supplement's own "Checkpoint draft 8" lists the reads.
- **[R1]** is now the Supplement's checkpoint draft 8 at `a39aeadd`. That draft is documentary on top of `3eb1d594`, which carries the last two AMENDs: the kernel integral in the proof of Lemma S13.3 and an exponent in Remark S11.6. Neither changes a statement that the main text cites.
- **§0, status table.** The leading-term row adds "every section of the Supplement READ (+GB), not yet read as a whole".
- **§0, "Before submission".** "Complete the Supplement and its nonauthor reading" becomes "have the Supplement read as a whole".
- **§0.** "Revised through r6".

**Build.** `build.sh` compiles with no LaTeX error and the same 2 pt overfull line as before.

## Revision r7 (15 October, about 00:40Z)

r6 is commit `398cf6fb`: 22 pages, 557,728 B, SHA-256 `602762a2…`. Its joint read with supplement draft 8 passed (main#389 6180842354). r7 is this PDF: 22 pages, 558,366 B, SHA-256 `ae553f12d512305429743d8c66a2ddad54a271fe825c597616fcb27b3c645a0f`.

**Why.** A provider-distinct read of Supplement §S5.4, by an OpenAI agent, returned BLOCK (main#389 6181596945). The author accepted it (6181660569):
- **F1.** The density bound (S5.56) uses the weight `ρ^{−m}`. That weight is not integrable at fixed positive scale, because the conditional density is positive at the pair axis there. The axis suppression `exp(−c_*/ρ²)` holds only in the contact limit. Lemma S5.23 uses (S5.56), and the collar count, Lemma S5.19, has the analogous transfer.
- **F2.** `f(x)` in `V_0` is not pinned. The fix is local.
- **F3.** A wording point.

So the second route to (3.13) in Supplement §S5 has an open gap until those finite-scale estimates are repaired. The main text's proof of Corollary 3.5 takes (3.13) from Theorem A of [R2] and does not use §S5, so no statement of the main text changes. r7 qualifies the places that described §S5 as complete.
- **§3.4, first paragraph.** "Written out in full" becomes "written out". It adds that a provider-distinct read found an open gap: Supplement Lemmas S5.19, S5.20 and S5.23 pass from contact-limit to finite-scale densities without a proof that is uniform near the pair axis.
- **§3.4, the envelope paragraph.** The collar and singular-near counts "are to give" `O(r³)`, "subject to the open gap".
- **§3.4, the proof of Corollary 3.5.** The witness route "outlines" a second argument, "complete except for the open gap".
- **§6, "Relation to the parent record".** The §S5 route is called a route with an open gap. Proposition S6.2 proves Proposition 3.10 without the record, and Theorem S5.36 "would do the same for (3.13) once that gap is closed".
- **§0, decision 4.** Alternative (b) applies "once the open gap in Supplement §S5 is closed".
- **§0, status row of (3.13).** "The second route in Supplement §S5: one nonauthor read (+GB), and a provider-distinct read (OpenAI) that found an open gap in its finite-scale witness densities."
- **§0.** "Revised through r7 on 15 October 2026".
- **§7.** The AI-review sentence now calls the Supplement §S5 argument a route with that open gap. It keeps the one nonauthor read of the route, together with Lemma S4.29 and Proposition S8.13 (added by the r7 read AMEND, main#389 6181730647).

**Build.** `build.sh` compiles with no LaTeX error and the same 2 pt overfull line as before.

## Revision r8 (15 October, about 08:30Z)

r7 is commit `5543bd20`: 22 pages, 558,417 B, SHA-256 `af944708…`. Its read AMEND and readback passed (main#389 6181730647, 6181766356). r8 is this PDF: 22 pages, 559,682 B, SHA-256 `633cff2a5351def94045994a893eef89bd19848128268c0eed90919a6d85dd50`.

**Why.** Two reasons.
1. Supplement draft 9 (`4db2c57a`, branch `claude/side24-v3-supplement-draft9-20261015`) repairs the gap that the OpenAI read of draft 8 found in Supplement §S5.4 (findings F1–F3 of main#389 6181596945). It adds §S5.8 (Lemmas S5.39–S5.41), the finite-scale densities near the pair axis by a radial blow-up, and proves Lemmas S5.19 and S5.23 again from it. No number of draft 8 changes. At this revision, the repair has one nonauthor read so far, by a Grok Bot support agent (PASS, main#389 6182910843), while an OpenAI re-read awaits publication of its verdict and an xAI read is open.
2. A same-provider audit of Supplement §§S2–S4, S6, S7 and S12–S13 against the main text found no mathematical error, and five citation and wording points in the main text. It earns no independence credit.

**Changes for the repair (no statement changes).**
- **Appendix A, [R1].** The Supplement is cited at draft 9: commit `4db2c57a` of the branch `claude/side24-v3-supplement-draft9-20261015`. Every Supplement number the main text cites is the same in draft 9.
- **§3.4, first paragraph.** The read of the eighth draft "found a gap"; "Supplement draft 9 repairs it in §S5.8 (Lemmas S5.39–S5.41), by a radial blow-up of the parameters at the axis with explicit limit functionals, and proves the two counts again from it (§7 records the reads of this repair)."
- **§3.4, the envelope paragraph.** The collar and singular-near counts "give" `O(r³)`, "with the finite-scale densities of Supplement §S5.8".
- **§3.4, the proof of Corollary 3.5.** The witness route outlines a second argument "with the repair noted at the start of this subsection".
- **§6, "Relation to the parent record".** The §S5 route is the one "whose finite-scale witness densities Supplement §S5.8 supplies". Theorem S5.36 "does the same for (3.13)".
- **§0, decision 4.** The alternatives no longer wait for the gap to close, and they add Supplement Theorem S12.3 (Theorem A of [R2], with its proof in Part III) with Proposition S6.2.
- **§0, status row of (3.13).** The OpenAI read "found a gap in its finite-scale witness densities, which Supplement draft 9 repairs (§S5.8)", followed by the read status above.
- **§7.** The AI-review parenthesis records the gap, the repair and the read status.
- **§0.** "Revised through r8 on 15 October 2026".

**Changes from the audit (wording and citation only).**
- **Convention 2.2, and the end of §1.** The canonical versions are now described exactly: the marked two-point formula in the pair coordinates of §3.1 for separations below a fixed `ϱ ∈ (0,12)`, and in the coordinates `(M,S)` beyond (Supplement (S6.7) and (S6.10); the same formulas without the elder mark for the candidate density). The substitution `κ = 6ℓ/r³` shows that the sum does not depend on `ϱ`. The pair coordinates exist only for separations below 12, so r7's "the representatives obtained from the pair coordinates" did not cover the far pairs. The sentence after the proof of Proposition 1.2 now points to Convention 2.2 instead of repeating r7's description.
- **§3.2, after (3.9).** "Uniformly bounded conditional Gaussian moments" becomes: moments "bounded by a polynomial in `|b|+κ`, uniformly in `t` and in `r` below any fixed `ϱ<12`, and hence uniformly bounded on compact mark sets (the proof of Supplement Proposition S6.4)". The conditional means are linear in the pin target, so the moments are not bounded uniformly in the marks.
- **§3.5, proof of Proposition 3.10, and §6.** (14.1) of [R2] is stated there for pairs at distance at least a small radius `r₀`. The proof now adds that its proof, a nonsingular two-site covariance on a compact set of pairs, applies at every separation `ϱ` (Supplement Lemma S12.29 and Proposition S6.2). §6 says the same in one clause. The proof still cites [R2], as decision 4's default requires.
- **§3.8, the first-variation paragraph.** The jet coordinates `a, m₄, χ, Ω` and the variations are cited to Supplement Lemma S4.20 and Corollary S4.25. Remark S4.26, which r7 cited there, is the scale check, and is now cited at "the scale check".
- **§3.9, proof of Proposition 3.12.** Theorem R is "proved in Supplement Theorem S7.9; Supplement Remark S7.22 identifies its density and its constant with those of Theorem 2.1."
- **README.** Revision r8; the earlier-revisions list adds r7; decision 4 as in §0; the supplement line cites draft 9 and its read status; "Requested: a read of r8".

**Not changed.** No theorem, display, constant or proof step. The other audit points concern the Supplement only, and wait for a later supplement draft.

**Build.** `build.sh` compiles with no LaTeX error and the same 2 pt overfull line as before.

## Revision r9 (19 October, about 22:10Z)

r8 is commit `2f176488`: 22 pages, 559,624 B, SHA-256 `9e051654…`. Its read AMEND and readback passed (main#389 6187682210, 6187755703). r9 is this PDF: 23 pages, 562,211 B, SHA-256 `2bea32be6684e19815875d4979e72ddb661bf18bd1a83a5e9df34db3e7eb8170`.

**Why.** Three reasons.
1. Supplement draft 10 (`5aa2fe29`, branch `claude/side24-v3-supplement-draft10-20261019`) turned Hypothesis S11.11, the identification of c₂, into Corollary S11.11, proved from the fold expansion of §S10. Its Cursor read checked the proof against everything it cites (AMEND on one word; readback PASS, main#389 6276344629). So Theorem 2.3(iii) no longer rests on two interfaces.
2. Supplement draft 11 (`c1d92e94`, branch `claude/side24-v3-supplement-draft11-20261019`) proves the cap-region bound of Proposition S8.3 (Theorem K of [R7]), on which the c₁ term rests, as Lemma S8.16. Its Cursor read found the proof complete (AMEND on four documentary lines; readback PASS, 6278180423). r9 cites draft 11 as [R1]; every Supplement number the main text cites is the same in drafts 10 and 11.
3. The same-provider audit of Supplement §§S10–S11 (zero credit) found two wording points in the main text, applied here.

**Changes (no proof step changes. The conditionality on the two identifications is removed, and the displayed enclosure of c₂ is widened by 10⁻¹⁶ at each end, from [0.1612340491269447, 0.1612340491269810] to [0.1612340491269446, 0.1612340491269811], as Corollary S11.10 states. The ratio interval and (2.14) do not change).**
- **Theorem 2.3(iii).** "Second" now names the fixed-cone coefficient expression c₂[K_ref] of the Gaussian kernel on ℝ³ (Supplement Definition S11.2; [R12]) and its closed form with the certified enclosure (Theorem S11.5); c₂[K₂₄] differs from it by less than 10⁻⁶⁰ (Corollary S11.10); and c₂ = c₂[K₂₄] is Corollary S11.11, "which derives from the fold expansion of Supplement §S10 the two identifications that [R12] states as interfaces". So c₂ ∈ [0.1612340491269446, 0.1612340491269811], and "Third" gives the ratio interval without "under the same interfaces", citing Corollary S11.12. The ratio, its digits and (2.14) are unchanged.
- **§0, status row of Theorem 2.3.** Adds that the identification c₂ = c₂[K₂₄] is Supplement Corollary S11.11 (draft 11), with a Grok Bot read of its proof.
- **§0, status row of (3.13), and §7.** The reads of the draft-9 repair: OpenAI (PASS_SCOPED, bound to the TeX source) and a Grok Bot support agent (PASS); the other slices of §S5 have xAI reads (PASS_SCOPED).
- **§5.1.** "of order L⁶e^{−L²/2}" becomes "of order at most L⁸e^{−L²/2}, as in (2.10) and Supplement Corollary S11.10" (audit S10S11-4; for c₂ the transfer bound is of order L⁸e^{−L²/2}).
- **Appendix A, [R1].** The Supplement is cited at draft 11, commit `c1d92e94` of the branch `claude/side24-v3-supplement-draft11-20261019`.
- **Appendix A, [R7].** Adds that the Supplement proves the bound (K2) of Theorem K as its Lemma S8.16.
- **Appendix A, [R12].** The reading of the c₁ enclosure as a coefficient ratio "uses Theorem CU" rather than "is conditional" (Theorem CU is Supplement Theorem S9.2, proved there).
- **§0.** "Revised through r9 on 19 October 2026".
- **README.** Revision r9; the earlier-revisions list adds r8; the reads line records M3 (PASS_SCOPED), the owner-authorized end of the M1 and M2 reservations (main#389 6278216270), and the supplement's state; "Requested: a read of r9".

**Build.** `build.sh` compiles with no LaTeX error and the same 2 pt overfull line as before. The bibliography now starts a 23rd page.

## Revision r10 (20 October, about 12:00Z)

r9 is commit `3022daae`: 23 pages, 562,211 B, SHA-256 `2bea32be…`. Its read AMEND and readback passed (main#389 6278373147, 6278394928). r10 is this PDF: 23 pages, 562,001 B, SHA-256 `edf1bc7704a22ddebc5a52a7324561637c1a8ff41f43c78a56700ef59060bbc7`.

**Why.** Two reasons, both documentary.
1. Supplement draft 12 (`59d20df5`, branch `claude/side24-v3-supplement-draft12-20261019`) proves Proposition S8.4(i)–(ii), which drafts 1–11 imported from Theorem Z of [R7], in a new §S8.6 (Cursor read: AMEND on one majorant constant and two parent hashes, applied; readback PASS, main#389 6278603420). With Lemma S8.16 of draft 11, Part II of the Supplement imports no analytic statement from a repository record. r10 cites draft 12 as [R1]; every Supplement number the main text cites that already appears in draft 10 is the same statement in drafts 11 and 12. Draft 11 adds Lemma S8.16 at the end of §S8, and draft 12 adds only the unnumbered subsection §S8.6.
2. The three provider-distinct reads offered on 19 October were delivered on 20 October by OpenAI Codex invocations, each PASS_SCOPED and released: M1 (§§1–2 of r9 at `3022daae`, main#389 6289211527), M2 (§3 of r9, 6289166739) and D10-C1111 (Corollary S11.11 and its proof at draft 12, whose §S11 is byte-identical to drafts 10 and 11, 6289117999). With M3 of r8 (xAI, 6191908445) and the Cursor read of the r8→r9 delta, every passage of the main text has a provider-distinct read.

**Changes (no theorem, display, constant or proof step changes).**
- **Appendix A, [R1].** The Supplement is cited at draft 12, commit `59d20df5` of the branch `claude/side24-v3-supplement-draft12-20261019`.
- **Appendix A, [R7].** "The Supplement proves the bound (K2) of Theorem K as its Lemma S8.16 (draft 11) and Theorem Z as its Proposition S8.4(i)–(ii), in its §S8.6 (draft 12); with these, Part II of the Supplement imports no analytic statement from a repository record, and its only computer-assisted inputs are the certified enclosures of c₁ and c₂ (Supplement §§S9, S11)."
- **§0, status row of Theorem 2.3.** The identification c₂ = c₂[K₂₄] (Corollary S11.11) now records the OpenAI read of its proof at draft 12 (PASS_SCOPED) beside the Grok Bot read at draft 10.
- **§0, status row of Remark 2.4.** Theorem Z is "R·2P, and proved in the Supplement (§S8.6, draft 12)"; Theorem K's bound (K2) is "proved in the Supplement (Lemma S8.16, draft 11)".
- **§0.** "Revised through r10 on 20 October 2026".
- **README.** Revision r10; the earlier-revisions list adds r9; the reads line records the three provider reads, the supplement at draft 12, and the reads of drafts 11 and 12; "Requested: a Cursor read of the r9→r10 delta".

**Build.** `build.sh` compiles with no LaTeX error and the same 2 pt overfull line as before; 23 pages.

## Revision r11 (20 October, about 22:00Z)

r10 is commit `d9f01444`: 23 pages, 562,001 B, SHA-256 `edf1bc77…`. Its read AMEND and readback passed (main#389 6290059705, 6290117154). r11 is this PDF: 23 pages, 562,006 B, SHA-256 `cbfe9db647bb20eeec477e5abd8715690673ec3c8b5a13ee6eeac80d71473027`.

**Why.** Supplement draft 13 (`d28e6ab4`, branch `claude/side24-v3-supplement-draft13-20261020`) is documentary over draft 12. It adds a proof of Proposition S12.24, item-level dependency sentences in S5 and S12, and the proof-dependency check. Its Cursor read had two AMENDs, both applied, and the readback passed (main#389 6290560837). All 948 labels of draft 12 keep their numbers, so every Supplement number the main text cites is the same statement in drafts 12 and 13. The provider slice reads SUPP-A to SUPP-E bind draft 12 (`59d20df5`), and draft 13 changes none of the statements or proofs they read, beyond the added proof and sentences.

**Changes (no theorem, display, constant or proof step changes).**
- **Appendix A, [R1].** The Supplement is cited at draft 13, commit `d28e6ab4` of `claude/side24-v3-supplement-draft13-20261020`.
- **§0.** "Revised through r11 on 20 October 2026".
- **README.** Revision r11; the earlier-revisions list adds r10; [R1] is draft 13, and draft 12 is named as the bytes the slice reads bind; "Requested: a Cursor read of the r10→r11 delta".

**Build.** `build.sh` compiles with no LaTeX error and the same 2 pt overfull line as before; 23 pages.

## Revision r12 (21 October, about 19:30Z)

r11 is commit `b033a118`: 23 pages, 562,006 B, SHA-256 `cbfe9db6…`. Its Cursor read passed (main#389 6302472048). r12 is this PDF: 23 pages, 562,474 B, SHA-256 `72f6b0566c8225ca30594cc451ec9fad39c38261e38a61676faddfa8e5672cf5`.

**Why.** Dylan Roy decided the four author decisions (main#389 6324313740): he kept the defaults of decisions 1–3 and chose the Supplement-only form of decision 4, which §0 had drafted since r7. So the leading term now rests on the Supplement alone.

**Changes (no theorem, display or constant changes; the proofs are the same, but the statements they cite now come from the Supplement).**
- **§3.4, the paragraph after Corollary 3.5, and the proof of Corollary 3.5.** (3.13) is Supplement Theorem S12.3, which is Theorem A of [R2] with its proof in Supplement Part III. Before r12 these passages cited Theorem A of [R2].
- **§3, proof of Proposition 3.10.** "This is Supplement Proposition S6.2." The proof applies the marked two-point Kac–Rice formula (Supplement Proposition S12.22) on the separated pair set, where the covariance is nonsingular. It cites no result of [R2] that the Supplement does not prove itself. For `0<ℓ≤1` and pairs at distance at least the small radius `r₀` of [R2], the same bound is its (14.1) (Supplement Lemma S12.29). Before r12 it was "the bound (14.1) of [R2]".
- **§6, "Relation to the parent record", last sentence.** §3 cites the Supplement statements, not the record: S12.3 for (3.13), and S6.2, whose proof uses only S12.22 from Part III, for Proposition 3.10. S5.36 gives a second, three-dimensional proof of (3.13). The rest of the paragraph is unchanged.
- **§0.** "Revised through r12 on 21 October 2026". The heading is now "Author decisions (decided)", with a sentence citing the decision record. Item 1 drops its "Alternatives" sentence. Item 4 states the decided form. "Settle decision 4" leaves "Before submission".
- **README.** Revision r12. The earlier-revisions list adds r11. "Author decisions (decided)" replaces "Decisions still with the author". The r11 read is recorded, and a read of the r11→r12 delta is requested. The integration line reflects the decisions.

**Build.** `build.sh` compiles with no LaTeX error and the same 2 pt overfull line as before; 23 pages. Built by `make_r12.py` (exact-match edits), then rebuilt.
