# SIDE24 V3 supplement: checkpoint draft 3 (14 October 2026)

Dylan Roy — delegated AI work. Actual performer: Anthropic Claude, session `session_01NMeKEismAyeqgdB4sy2NJU` (Claude01N), the author of the V3 draft (main#389). This is an **author-side checkpoint** of the supplementary material for the V3 manuscript draft. It is pushed so that the work is durable. **No pull request is open for it, and no read is requested yet.** Scientific effect NONE: no register, status or proof file changes.

## Contents

- `SIDE24_V3_SUPPLEMENT_DRAFT_20261013.pdf`: the compiled draft, 127 pages, 1,560,193 bytes, SHA-256 `c8fc570a6f727669f2033d61fb13d34b59c0bee1d31821c702163adeb092bdea`. Its first section, "Status of this draft", is for the author. Draft 1 (`4f9e6180`) had 123 pages, and draft 2 (`238ca036`) 126.
  - Part I (S1–S7): the setting, the contact measure, the finite-side coefficient, the side-24 correction with |eps_24| < 10^-216, elder selection, the limit, and the bounded remainder.
  - Part II (S8–S11): the remainder's vanishing and rate, the cusp term, the third-order terms, and the constant c_2.
- `supp_main.tex`, `supp_s0_status.tex` … `supp_s11_ctwo.tex`, `refs.bib`, `supp_refs_s8s9.bib`: the LaTeX sources. `build.sh` rebuilds the PDF.
- `CONCORDANCE.md`: the source of every statement and display, the corrections applied, the added steps and the issues found. Its head summarizes:
  - the author correction made at integration (the elder mark);
  - the findings that bear on the main text;
  - in "Checkpoint draft 2", the disposition of every issue of S1–S6, and the repairs that make S5 a complete second proof of the selection bound (3.13);
  - in "Checkpoint draft 3", the far elder rate (Proposition S8.13) and two pointer and citation fixes in S10–S11.
- `checks/s5_envelope_check.py` and its output `checks/s5_envelope_check.out.txt`: a SymPy check of the cancellations behind the singular-near envelope of Lemma S5.22.
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
