# `00.1_GOVERNANCE_CANON/02_PDF_EXPORTS_AND_ORIGIN_DOCUMENTS`

Drive folder id `1CvRve9avU92nkgj0gRBHsRY5naowE1xE`, **3 inventory items**: the
folder itself (indexed tree-only in the lane-root manifest) and two PDFs, both
held byte-exact.

| file | identity |
|---|---|
| `GATE_FRAMEWORK_MASTER — CURRENT-v1.2-RECONCILED-PDF — 92754B — SHAe2a8b7d3 — ID1nBgMT82.pdf` (92,754 B) | **byte-exact**: SHA-256 `e2a8b7d3…`, id `1nBgMT82BOUBAKO_xw-lSPn0-_HRzWNuD` |
| `GATE_FRAMEWORK_MASTER — v1.0-ORIGIN-PDF — 77525B — SHAd933bcf4 — ID1tX2omUL.pdf` (77,525 B) | **byte-exact**: SHA-256 `d933bcf4…`, id `1tX2omUL6saHa5omzRi8LialKiRZWHov1` |

## No banner is quoted from these two files, deliberately

A PDF's text lives inside compressed content streams. Any sentence this README
pulled out of one would be the product of a text extractor, not a run of bytes
this repository holds, and could not be checked against the stored file by
`tools/mirror_quotes_check.py`. So nothing inside either PDF is put in quotation
marks anywhere in this lane's READMEs.

What the file names and the Drive path say for themselves is the rest: one is
the v1.0 origin PDF, the other a PDF of the reconciled v1.2. The reconciled
Markdown v1.2 in the sibling `00_CURRENT_RECONCILED_SPECIFICATION` lists, under a
heading reading "Supersedes without overwriting:", a "Master v1.0 PDF;" among the
three lineages it supersedes without overwriting. Whether the v1.0 origin PDF
stored here is that object is not established by this port: the v1.2 file names
it by description, not by Drive id, and no register row in this repository binds
the two.

## What this does not establish

Both byte-exact rows establish that this directory holds the same bytes the
2026-09-17 inventory declares for those ids, and nothing further. A PDF rendering
is not a frozen body, and a byte-exact copy of a rendering is still a rendering.
Mirroring is not review, replay, endorsement or promotion, and nothing here
grades, closes or discharges any claim, premise or obligation. Neither file is
read, parsed, imported or tested by anything in this repository.
