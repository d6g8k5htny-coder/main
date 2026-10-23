# Historical mathematical carriers in the public Dropbox import — provenance crosswalk

**Engineering archive index only; NOT A RESULT.** Prepared 2026-10-23. Scientific effect: **NONE**. No new graph nodes, v2 bindings, mathematical proof review, accepted theorem, source promotion, or external-custody certification.

## Scope and immutable inputs

This is a narrow appendix to the **already-public and privacy-screened** [Dropbox import of 2026-09-27](README.md), not a new import from private Dropbox. The canonical source inventory is [MANIFEST.csv](MANIFEST.csv), Git blob `aea93c0498e370c8333faad88eed0de832c77e46`, at `d6g8k5htny-coder/main@a487cfd41bfe095842d9d9f5cec26fc23877b028`. Its `Math-imports-candidate` label appears in **11** rows. All row files are tracked public Git blobs in this frozen tree. **Seven** are byte-exact published text originals (as described by the historical import); **four** are *extracted-text surrogates* for PDF/Word originals, not the original document bytes.

For this appendix, the eleven public GitHub file bodies were fetched at the pinned commit and their UTF-8 byte streams independently SHA-256 checked against the prior manifest: **11/11 MATCH**, **149,524 published bytes**, zero mismatches. The checksum implementation was checked against the standard SHA-256 `""` and `"abc"` vectors. Every file's native Git blob is given below. This check authenticates the **already-published Git content** against the **existing manifest**, not a fresh download or SHA-256 of the private Dropbox original. In particular, a manifest `dropbox_content_hash` is not an ordinary raw-file SHA-256 and an extracted PDF text hash is not the PDF hash.

The broader original import recorded 5,105 Dropbox objects, 107 publications (93 exact originals and 14 extracted texts), and 26 project-related files not imported. The full import has **110** Git blobs including `README.md`, `MANIFEST.csv` and `NOT_IMPORTED.csv`. Those figures are historical as-of September 27, not a claim about current private cloud completeness. Never publish newly discovered private-source paths, names or access links on the basis of this appendix.

## Eleven previously published mathematical carriers

Archive paths below are relative to this directory and were **already public** in the original manifest. The `archive_sha256` column is SHA-256 of the published Git body; for extracted text it is not the original source file.

| Row | Existing public `archive_path` | Publication form | Pinned Git blob (SHA-1) | Published `archive_sha256` | Provenance role / boundary |
|---:|---|---|---|---|---|
| 1 | `files/Github/02_PRE_REVIEW_DOSSIER.pdf.txt` | Extracted text | `6ab9bd8e8cc73391302b421a11a3d4a0a093f743` | `f70684ba563e9e008e9590f943a03fb07b4c9ff3e59cc8fff15e45d88eafdf8a` | Reviewer-facing SIDE24 dossier; PDF extraction only; no original PDF bytes or code implied |
| 2 | `files/Github/03_SIDE24_MANUSCRIPT.pdf.txt` | Extracted text | `4fe100fee09c2a3e744b4caf67922f2667aebe6d` | `d0dbc29e77529fe217f76c30efecc531647420f60759570775af4b9afaaef773` | SIDE24 manuscript draft; PDF extraction only; do not equate to a reviewed Math proof |
| 3 | `files/Github/070326claufe.pdf.txt` | Extracted text | `eab23bf56e2e0a97d25687aeee315cec86fb1385` | `0d7515b6dec5b7ee02297510f7df79e323ff755be01abcb3be815518e721fee6` | Topological-fingerprint overview; PDF extraction only; no tested implementation here |
| 4 | `files/Github/LPW_Return_05_Archive_Audit_2026-09-12 (1)/02_CONSTANT_FINDINGS_AND_REPAIR.md` | Original text | `0f2fbad209857d807a3ac31acab3a3ff917ef9ae` | `feb8aa30795cfe44d5c9e49b1793e6cd5f62729b922af0b0aff5fe076868146a` | LPW Return-05 audit, AMEND REQUIRED for strongest displayed constant |
| 5 | `files/Github/LPW_Return_05_Archive_Audit_2026-09-12 (1)/05_MATCHING_2D_UPPER_WORK_ORDER.md` | Original text | `7437eab6b92d8d851564a20365891c2c305be5a5` | `521870da3c56e1c0b01d1ddd95ed159b89925bc31957b6efecd8aa307e76c479` | 2D matching upper-bound work order; an open target, not an established theorem |
| 6 | `files/Github/LPW_Return_05_Archive_Audit_2026-09-12 (1)/receipts/W8_RECOVERED_PREFLIGHT.out` | Original text | `9f94ba06ed6f13b05b8bb2de81551761e2b42ec5` | `85b1c5c424c931e1e01c99c2adbc38dafe2b50a3dcb7cf8c1e159f6272c5e953` | W8 lambda-grid preflight output; execution source is not among the archived public members |
| 7 | `files/Github/LPW_Return_05_Archive_Audit_2026-09-12 (1)/source_excerpts/LPW_CONSTANT_REPORT.numbered.txt` | Original text | `dcc840cb5d7fd56e4440bdfb8ffd368f10af2e4d` | `b82f1031c64cd1872c7bdfa2f621c8fecc7b2c88d8626a4938119c996d3cddc0` | Line-numbered historical LPW constant report; received report, not a current verdict |
| 8 | `files/Github/LPW_Return_05_Archive_Audit_2026-09-12 (1)/source_excerpts/lpw_constant.py.numbered.txt` | Original text | `a74c0e86dcb4a157c69cdb9b6cc3955833d64a4c` | `2f66f49286e6d90df34f933314b84ad8dc0bf5c6c47b552678ca89bae38143be` | Line-numbered Python certifier excerpt; not identical to an executable raw .py file |
| 9 | `files/Github/LPW_Return_05_Archive_Audit_2026-09-12 (1)/tools/check_findings.py` | Original text | `54123700fb2b79e32fbd574536bda0eb8e4cd03f` | `55d6a213a42e729d5ff6687d1552a1b512e00638d73f9d4e5be6b867e86aa67e` | Return-05 regression checker; requires absent raw/constant package and sympy |
| 10 | `files/Github/Three Bridges from Persistence Thermodynamics to Geometry, Entropy, and Field Theory13M+.pdf.txt` | Extracted text | `aac40e32c6b5de8ddb8966ecf94ff4748d3752b9` | `702eb70d1786216310b86ff92fe52f9cfdf67876cd8cb7d51d660c93c4c8bc52` | Cross-disciplinary literature synthesis; extracted PDF text, not proof implementation |
| 11 | `files/Github/synthetic results.json` | Original text | `7124945b135c70b05e67049033a846edfe4644e6` | `151fd881d01ad01c273e5442ebb48a67e6f8faba188f62d289ca6dbd9b06307d` | Synthetic numerical results; no authenticated generator identified in this import |

## Explicit, limited historical source → checker → output relationships

The **LPW Return-05** group (rows 4–9) contains multiple, *different* evidence types, connected by their own already-public text:

- The archived **`02_CONSTANT_FINDINGS_AND_REPAIR.md`** identifies its received `raw/constant/LPW_CONSTANT/` source and directly discusses the `lpw_constant.py` computation and `LPW_CONSTANT_REPORT.md`. Its declared disposition is **AMEND REQUIRED** for the *strongest displayed quantitative constant*. It documents a denominator/rounding mismatch and an invalid half-normal fourth-moment identity. Neither this appendix nor a successful checksum removes those defects. It expressly does **not** refute the qualitative pairing probability or endorse a replacement numerical theorem.
- **`source_excerpts/LPW_CONSTANT_REPORT.numbered.txt`** is the line-numbered historical report; **`source_excerpts/lpw_constant.py.numbered.txt`** is the line-numbered Python source excerpt. The numbered text is **not** byte-equivalent to an executable `lpw_constant.py`; its original unnumbered raw program is **not present in this limited public archive subtree**. Do not invent a source blob or claim the original checker executed in the current checkout.
- **`tools/check_findings.py`** is the archived *regression checker*, not a theorem verifier. Its source reads `ROOT/raw/constant/LPW_CONSTANT/lpw_constant.py` and pins that historical source's SHA-256; its imports also require `sympy`. That raw directory and program are absent from this public import, so **no self-contained reproduction of the checker is established from these public files alone**. It can be associated as an authored implementation artifact, not recorded as a current passing calculation.
- **`receipts/W8_RECOVERED_PREFLIGHT.out`** is a historical *output*. Its header attributes itself to `verify_lambda_grid_v3.py` with an earlier v2 fail-closed quotient guard; the source program and complete original W8 inputs are **not present in these 110 imported public Git blobs**. Neither the received output nor the separate Return-05 replay receipt is a new execution performed for this appendix.
- **`05_MATCHING_2D_UPPER_WORK_ORDER.md`** defines a separate target, the matching two-dimensional upper bound; it is not evidence that the all-small-r upper theorem was proved. **`synthetic results.json`** is a separate synthetic data carrier with no authenticated generating program identified in this import.

The remaining documents are a pre-review dossier, a manuscript draft, a topology overview and a literature synthesis. The four extracted-document texts may be valuable historical *reading sources*, but do not carry the original PDF layout, formulas or figures by byte identity. None is silently joined to an independently reviewed present-day Math- theorem or implementation on the basis of a shared title.

## Retroactive graph/accounting implications and next permitted joins

The existing [RETRO-MATH-BIND-01 inventory and allocation](https://github.com/d6g8k5htny-coder/main/pull/527) counts 313 units on its frozen Math- and selected main source cuts; it does **not** by itself certify this older `audits/dropbox_import` collection as direct current Math- source. This appendix accounts for the *11 previously published, privacy-approved candidate carriers* without creating a competing ownership register or adding node IDs.

To add a future **graph-to-artifact** edge, a separate source owner must first identify an *existing genuine graph node*, a reviewed target `owner/repo:path@40hexcommit#40hexblob`, the exact repository-qualified checked ref, and valid public bytes. Then distinguish `historical`, `external_unverified`, `absent` and current `file`/verified `tree` exactly as [the reviewed GB-5 v2 proposal](https://github.com/d6g8k5htny-coder/main/pull/522) requires; only actually supported current pins may be BOUND. Do **not** create an Artifact Index row or mathematical status from these archive excerpts. The ongoing GB-8a–x Math frontier assignment, future private-cloud DRV lanes and the original import's publication/hold decisions retain their own custody.

## Reproduce the published archive identity checks

From a checkout containing this pinned Git history, the following **standard-library-only read** checks the exact manifest-selected already-public math rows; it does not access Dropbox, execute mathematical programs or create files.

```bash
git switch --detach a487cfd41bfe095842d9d9f5cec26fc23877b028
python -B -S - <<'PY'
import csv, hashlib
from pathlib import Path
root = Path("audits/dropbox_import/2026-09-27")
with (root / "MANIFEST.csv").open(newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f)
            if r["suggested_home"] == "Math-imports-candidate"]
assert len(rows) == 11, len(rows)
assert sum(r["disposition"] == "PUBLISHED_EXACT_ORIGINAL" for r in rows) == 7
assert sum(r["disposition"] == "PUBLISHED_EXTRACTED_TEXT" for r in rows) == 4
count = 0
for r in rows:
    raw = (root / r["archive_path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == r["archive_sha256"], r["archive_path"]
    count += len(raw)
assert count == 149524, count
print("PASS: 11 published math candidate archive carriers, 149524 bytes")
PY
```

**Limit:** this checksum recipe was independently evaluated against the GitHub-provided content bytes in the authoring session; the Bash checkout command and this exact Python snippet were not executed here. GitHub checkout reproductions must preserve the original bytes. Consult [NOT_IMPORTED.csv](NOT_IMPORTED.csv) and the privacy/approval rules before any new copy or restoration. Scientific and reviewer statuses remain unchanged.

[Complete owner mandate](https://github.com/d6g8k5htny-coder/main/blob/b665307520948308a53d190d7341bf301df3b3a2/governance/OWNER_MANDATE_20261015.md) · [Standing delegation](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OP-AUTONOMY-20260923-v2.1.md).
