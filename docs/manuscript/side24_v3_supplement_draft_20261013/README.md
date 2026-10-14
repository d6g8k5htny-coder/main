# SIDE24 V3 supplement: checkpoint draft 4 (14 October 2026)

Dylan Roy — delegated AI work. Actual performer: Anthropic Claude, session `session_01NMeKEismAyeqgdB4sy2NJU` (Claude01N), the author of the V3 draft (main#389). This is an **author-side checkpoint** of the supplementary material for the V3 manuscript draft. It is pushed so that the work is durable. **No pull request is open for it, and no read is requested yet.** Scientific effect NONE: no register, status or proof file changes.

## Contents

- `SIDE24_V3_SUPPLEMENT_DRAFT_20261013.pdf`: the compiled draft, 129 pages, 1,579,632 bytes, SHA-256 `e303d3d385c66e857aadb863e802b29b071b55a2ba9299cb3981718cd9971c28`. Its first section, "Status of this draft", is for the author. Draft 1 (`4f9e6180`) had 123 pages, draft 2 (`238ca036`) 126, and draft 3 (`e0d08f0c`) 127.
  - Part I (S1–S7): the setting, the contact measure, the finite-side coefficient, the side-24 correction with |eps_24| < 10^-216, elder selection, the limit, and the bounded remainder.
  - Part II (S8–S11): the remainder's vanishing and rate, the cusp term, the third-order terms, and the constant c_2.
- `supp_main.tex`, `supp_s0_status.tex` … `supp_s11_ctwo.tex`, `refs.bib`, `supp_refs_s8s9.bib`: the LaTeX sources. `build.sh` rebuilds the PDF.
- `CONCORDANCE.md`: the source of every statement and display, the corrections applied, the added steps and the issues found. Its head summarizes:
  - the author correction made at integration (the elder mark);
  - the findings that bear on the main text;
  - in "Checkpoint draft 2", the disposition of every issue of S1–S6, and the repairs that make S5 a complete second proof of the selection bound (3.13);
  - in "Checkpoint draft 3", the far elder rate (Proposition S8.13) and two pointer and citation fixes in S10–S11;
  - in "Checkpoint draft 4", a correction to the singular-near chart of S5 (a random `η²` term in the residual of `V_0`), the proof of the kernel transfer in S5 by exact factorizations, and the jet bounds (3.29) of the main text in Lemma S4.29.
- `checks/s5_envelope_check.py` and its output `checks/s5_envelope_check.out.txt`: a SymPy check of the cancellations behind the singular-near envelope of Lemma S5.22.
- `checks/s5_factor_check.py` and its output `checks/s5_factor_check.out.txt`: an exact SymPy check of the singular-near residual limits, the planar covariances of Lemmas S5.16–S5.21, the factorizations behind Lemmas S5.18 and S5.21 for random anisotropic even kernels, and the numbers of Lemma S4.29.
- `writer_originals/`: the S5 writer's faithful transcription, before the elder-mark correction.
- `SHA256SUMS`: hashes of every file above except this README.

## How it was made

Nine fresh subagents of the author's session each wrote one or two sections from the exact technical appendix (30 July 2026), the gap-fill supplement (1 August 2026) and the cited Math- records at `origin/main` `9844f785`. All wrote under the same rules: transcribe faithfully, apply only the listed corrections, and report anything else as an issue. The author integrated the result; the build has 0 LaTeX errors and no undefined references. The supplement has not yet been read as a whole, and same-provider work earns no independence.

Draft 2 is the author's repair pass over the concordance issues of S1–S6; the concordance records each disposition. In S5:
- the trichotomy uses same-maximum self-attachment, with a new merge-level lemma;
- the regions are measured from the midpoint;
- the fixed-distance count holds on compact marks;
- the singular-near envelope is proved by Taylor's formula for every angle, which closes the `c = 0` case;
- the capture-escape lemmas carry the five local repairs.

The main text (main#389) cites Theorem A of the companion paper for (3.13); S5 is a second proof.

Draft 3 adds the proof of the far elder rate `O(ℓ^{2/3})` (Proposition S8.13, transcribed from Math- record [R9] in Part II's notation). Part II previously cited that record for the rate. It also fixes two pointers and adds a DLMF citation.

Draft 4 makes four repairs, none of which changes a conclusion.
- It corrects the singular-near residual of `V_0`, which has a random `η²` term that the sources omitted. The planar covariances now depend on `η`, and the axis suppression used is `exp(−c/ρ²)`.
- It proves the positivity of `A_col` and `A_sing` for the side-24 field by exact factorizations and the derivative nondegeneracy of S1. This replaces a derivative bound `10^40` that S4 does not contain.
- It adds the main text's jet bounds (3.29) to Lemma S4.29.
- It displays the level terms of Lemma S5.28 and identifies the near kernels of S6 and Part II.
