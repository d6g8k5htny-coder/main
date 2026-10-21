# SIDE24 manuscript — V3 draft (13 October 2026; revision r16 of 21 October)

Dylan Roy — delegated AI work. Actual performer: Anthropic Claude, Claude Code session `session_01NMeKEismAyeqgdB4sy2NJU` (Claude01N), the author of the V3 edit pack. This is an **author-side draft** of the SIDE24 manuscript for the author's decisions and for nonauthor reads. Scientific effect NONE: no register, status or proof file changes, and no theorem changes.

## Contents

- `SIDE24_V3_DRAFT_20261013.pdf`: the compiled draft, revision r16, 23 pages. Its first section, "Status of this draft", is for the author and is removed before submission. Earlier revisions are kept at their commits: r15 at `18543a20`, r14 at `ba1b904d`, r13 (23 pages, SHA-256 `b3852c53…`) at `04a3ee03`, r12 (23 pages, SHA-256 `72f6b056…`) at `d550d238`, r11 (23 pages, SHA-256 `cbfe9db6…`) at `b033a118`, r10 (23 pages, SHA-256 `edf1bc77…`) at `d9f01444`, r9 (23 pages, SHA-256 `2bea32be…`) at `3022daae`, r8 (22 pages, SHA-256 `9e051654…`) at `2f176488`, r7 (22 pages, SHA-256 `af944708…`) at `5543bd20`, r6 (22 pages, SHA-256 `602762a2…`) at `398cf6fb`, r5 (22 pages, SHA-256 `0d8113f7…`) at `74bc4988`, r4 (21 pages, SHA-256 `a22db882…`) at `7bab515f`, r3 (19 pages, SHA-256 `75496787…`) at `7aff36d0`, r2 at `ff73d996`, r1 (SHA-256 `aa866725…`) at `56b680cd`, and r0 (18 pages, SHA-256 `8270e498…`) at `6deb0899`. `V3_CHANGE_LIST.md` records every revision.
- `main.tex`, `s0_status.tex` … `s7_disclosure.tex`, `appA_records.tex`, `refs.bib`: the LaTeX sources. `build.sh` rebuilds the PDF with `pdflatex` and `bibtex` (amsart, amsplain).
- `V3_CHANGE_LIST.md`: where every V2 amendment (M1–M7) and every V3 edit-pack item (E1–E78) went, the corrections found while drafting, the record of the pre-delivery check, and the "do not print" compliance.
- `SHA256SUMS`: hashes of every file above except this README.

## Provenance

- **Base text:** the V1 source `SIDE24_MANUSCRIPT_CORE.md`, 21,550 bytes, SHA-256 `fb576de64f17f64a23423b185e2abc902a436d3a6a1bbb2aee04d756ad371183`. It is blob `99aea93b` on this repository's branch `chatgpt/drive-github-hardening-20260919` (`a01c72f1`), and equals the V1 manifest's `03_SIDE24_MANUSCRIPT_SOURCE.md`.
- **Bibliography:** V1's `08_REFERENCES.bib`, blob `06970b5d`. It is extended per the edit pack and the literature delta of main#275 comment 6146291432; the first author's given name for Klein–Agam is corrected.
- **Inputs:**
  - the V2 amendment sheet M1–M7 and the gap-fill supplement (1 August 2026);
  - the V3 edit pack, items E1–E78, last entry 13 October 15:52Z;
  - the repository records listed in the draft's Appendix A. Every path, blob and merge identity there was checked against Math- `main` `b68d88b3`.

## Author decisions (decided; main#389 6324313740)

1. Theorem 2.1 states the two-provider form `c ℓ^{-1/3}(1 + O(ℓ^{1/3}))`. The merged three-term law is Theorem 2.3. The later refinements are Remark 2.4. (Default kept.)
2. Human review is stated as a disclosure. (Default kept.)
3. Planar (`d = 2`) material appears only in the Discussion. (Default kept.)
4. The relation to the parent record [R2]: **Supplement only** (applied in r12). §3 cites Supplement Theorem S12.3 (the parent's Theorem A with its proof in Part III) for (3.13) and Supplement Proposition S6.2 for Proposition 3.10. The §6 paragraph "Relation to the parent record" stays, and Supplement Part III writes the parent record out.

## Review

- **Pre-delivery check:** a fresh read-only checker of the same provider. Verdict PASS WITH FIXES (0 blockers, 6 major, 25 minor), and all findings are applied (see `V3_CHANGE_LIST.md`). It earns no independence credit.
- **Nonauthor reads (Cursor, Grok; provider-distinct, same account):** every section of r2, the changed passages of r3 through r9, and each AMEND's readback. **Provider reads of the main text, all three slices:** M3 (§§4–7 and Appendix A) of r8 by xAI, PASS_SCOPED (main#389 6191908445), with the r8→r9 delta covered by the Cursor read of r9; M1 (§§1–2) and M2 (§3) of r9 at `3022daae` by OpenAI Codex on 20 October, both PASS_SCOPED (main#389 6289211527, 6289166739). The 15 October pickups of M1 and M2 were never delivered, and their reservations were ended by owner authority on 19 October (main#389 6278216270). The companion supplement, which r16 cites as [R1] at `ddd60647` (draft 15, branch `claude/side24-v3-supplement-draft15-20261021`; it keeps every statement number of drafts 12–14; draft 12 is `59d20df5`, which every provider slice read binds), has had one nonauthor read of every section, in slices; its "Checkpoint draft 8" and the PR body of main#389 list the verdicts. Draft 9 repairs the finite-scale witness densities of §S5.4, which an OpenAI read of draft 8 found unproved; the repair has nonauthor reads by OpenAI (PASS_SCOPED, bound to the TeX source, main#389 6193704852) and by a Grok Bot support agent (PASS, 6182910843), and the other slices of §S5 have xAI reads (PASS_SCOPED). Draft 10 makes the repairs of a same-provider audit and proves the identification of c₂ (Corollary S11.11), with a Cursor read (AMEND applied, readback PASS, main#389 6276344629) and an OpenAI read of the proof at draft 12, whose §S11 is byte-identical to drafts 10 and 11 (PASS_SCOPED, 6289117999); draft 11 proves the cap-region bound of Proposition S8.3 in Lemma S8.16 (Cursor read, readback PASS, 6278180423); draft 12 proves Proposition S8.4(i)–(ii), Theorem Z of [R7], in §S8.6 (Cursor read, readback PASS, 6278603420). With drafts 11 and 12, Part II of the supplement imports no analytic statement from a repository record.
- **Reads of r11:** the Cursor read of the r10→r11 delta passed (main#389 6302472048).
- **Reads of r12:** the Cursor read of the r11→r12 delta passed (main#389 6324514156).
- **Reads of r13:** Cursor AMEND (main#389 6325193355), applied; readback AMEND (6325247357), applied; readback PASS (6325266500).
- **Reads of r14:** Cursor AMEND (main#389 6325580614), applied; readback AMEND (6325782917), applied; readback PASS (6325821191).
- **Reads of r15:** Cursor PASS (main#389 6326222268).
- **Requested:** a nonauthor read of the r15→r16 delta (Appendix A [R1] now cites supplement draft 15; §0; this README; the change list).
- **Integration:** not requested. The author's decisions are made; submission still waits on the items in §0, "Before submission".
