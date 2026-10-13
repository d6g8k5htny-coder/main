# SIDE24 V3 supplement: checkpoint draft 1 (13 October 2026)

Dylan Roy — delegated AI work. Actual performer: Anthropic Claude, session `session_01NMeKEismAyeqgdB4sy2NJU` (Claude01N), the author of the V3 draft (main#389). This is an **author-side checkpoint** of the supplementary material for the V3 manuscript draft. It is pushed so that the work is durable. **No pull request is open for it, and no read is requested yet.** Scientific effect NONE: no register, status or proof file changes.

## Contents

- `SIDE24_V3_SUPPLEMENT_DRAFT_20261013.pdf`: the compiled draft, 123 pages, SHA-256 `9da40caf5646caf376a885a4657d045689dd1541f2d84d6c730efc41e4231e2e`. Its first section, "Status of this draft", is for the author.
  - Part I (S1–S7): the setting, the contact measure, the finite-side coefficient, the side-24 correction with |eps_24| < 10^-216, elder selection, the limit, and the bounded remainder.
  - Part II (S8–S11): the remainder's vanishing and rate, the cusp term, the third-order terms, and the constant c_2.
- `supp_main.tex`, `supp_s0_status.tex` … `supp_s11_ctwo.tex`, `refs.bib`, `supp_refs_s8s9.bib`: the LaTeX sources. `build.sh` rebuilds the PDF.
- `CONCORDANCE.md`: the source of every statement and display, the corrections applied, the added steps and the issues found. Its head summarizes one author correction made at integration (the elder mark) and the findings that bear on the main text.
- `writer_originals/`: the S5 writer's faithful transcription, before the elder-mark correction.
- `SHA256SUMS`: hashes of every file above except this README.

## How it was made

Nine fresh subagents of the author's session each wrote one or two sections from the exact technical appendix (30 July 2026), the gap-fill supplement (1 August 2026) and the cited Math- records at `origin/main` `9844f785`. All wrote under the same rules: transcribe faithfully, apply only the listed corrections, and report anything else as an issue. The author integrated the result; the build has 0 LaTeX errors and no undefined references. The supplement has not yet been read as a whole, and same-provider work earns no independence.
