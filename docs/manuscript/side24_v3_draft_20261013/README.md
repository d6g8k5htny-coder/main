# SIDE24 manuscript — V3 draft (13 October 2026; revision r4 of 14 October)

Dylan Roy — delegated AI work. Actual performer: Anthropic Claude, Claude Code session `session_01NMeKEismAyeqgdB4sy2NJU` (Claude01N), the author of the V3 edit pack. This is an **author-side draft** of the SIDE24 manuscript for the author's decisions and for nonauthor reads. Scientific effect NONE: no register, status or proof file changes, and no theorem changes.

## Contents

- `SIDE24_V3_DRAFT_20261013.pdf`: the compiled draft, revision r4, 21 pages. Its first section, "Status of this draft", is for the author and is removed before submission. Earlier revisions are kept at their commits: r3 (19 pages, SHA-256 `75496787…`) at `7aff36d0`, r2 at `ff73d996`, r1 (SHA-256 `aa866725…`) at `56b680cd`, and r0 (18 pages, SHA-256 `8270e498…`) at `6deb0899`. `V3_CHANGE_LIST.md` records every revision.
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

## Decisions still with the author

The draft uses defaults, and each is a local switch:
1. Theorem 2.1 states the two-provider form `c ℓ^{-1/3}(1 + O(ℓ^{1/3}))`. The merged three-term law is Theorem 2.3. The later refinements are Remark 2.4.
2. Human review is stated as a disclosure.
3. Planar (`d = 2`) material appears only in the Discussion.
4. The relation to the parent record [R2] is stated in a §6 paragraph (added in r4, after a same-provider referee read).

## Review

- **Pre-delivery check:** a fresh read-only checker of the same provider. Verdict PASS WITH FIXES (0 blockers, 6 major, 25 minor), and all findings are applied (see `V3_CHANGE_LIST.md`). It earns no independence credit.
- **Nonauthor reads (Cursor, Grok; provider-distinct, same account):** every section of r2, the changed passages of r3, and each AMEND's readback. The companion supplement (`claude/side24-v3-supplement-draft-20261013`, `c6aa790d`), which r4 cites as [R1], was read for §S5, Lemma S4.29 and Proposition S8.13 (PASS, main#389 6178514413).
- **Requested:** a read of r4.
- **Integration:** not requested. The draft is not for merge before the author's decisions.
