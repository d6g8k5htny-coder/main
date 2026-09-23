# Local Recovery Sweep — 2026-08-01

## Outcome

The allegedly missing SIDE24 carriers were not lost. The current local corpus
contains byte-exact copies of the pre-review R2 ZIP, the V5 supplement through
§G.10, and the historical v8 verifier. The most important naming distinction
is that two unrelated objects contain `r2` in their names:

1. `SIDE24_PRE_REVIEW_2026-07-31.zip` is the relevant SIDE24 R2/pre-review
   package. It is 2,615,005 bytes with SHA-256
   `14eaf7d403cd8c103576807719e8ee031cad1fc5bba58f1c37574a790849e44b`.
2. `research_formal_core_r2.zip` is an unrelated Lean source-recovery
   successor. It is 10,644 bytes with SHA-256
   `a6440511fc259706457b18227734013966251105d538fdeabe633bb73f168631`.

The later, disputed §G.9.1 was never part of the recovered V5 supplement or
its v8 verifier. The V5 source stops at a §G.9 status section that keeps
RP-C/RP-S open, followed by §G.10. Consequently, the exact later §G.9.1 prose
and its own verifier cannot be claimed as byte-for-byte local recovery. Its
mathematical core has now been reconstructed separately, and the exact AO48
review/receipt carriers were recovered by the concurrent Drive provenance
sweep.

No source package was edited. All writes below are copies, extractions,
transcripts, inventories, or this report.

## 1. Exact SIDE24 carriers recovered locally

### 1.1 R2/pre-review ZIP

- Original local path:
  `/workspace/scratch/eef1f5ced3ed/SIDE24_PRE_REVIEW_2026-07-31.zip`
- Recovery copy:
  `/workspace/scratch/4c1cbf90475f/recovery/local_sweep/SIDE24_PRE_REVIEW_2026-07-31.zip`
- Bytes: `2,615,005`
- SHA-256:
  `14eaf7d403cd8c103576807719e8ee031cad1fc5bba58f1c37574a790849e44b`
- `unzip -t`: no errors.
- Payload: 69 files (73 ZIP entries including directories).
- Internal `10_SHA256SUMS.txt`: 68/68 listed payloads match, exit 0.

Verification records:

- `SIDE24_PRE_REVIEW_2026-07-31.unzip_test.txt`
- `SIDE24_PRE_REVIEW_2026-07-31.sha256_check.txt`

This hash is exactly the `14eaf7d4…` R2 identity cited by AO48-REC-034 and
AO48-AUD-033.

### 1.2 V5 supplement and historical verifier v8

These were extracted from the attached operator carrier
`OKComputer_Project_Gap_Closure(1).zip` (archive SHA-256 prefix
`119ffbfae2d11cc2`) and copied under `recovered_side24_v5/`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `SIDE24_GAP_FILL_SUPPLEMENT.md` | 105,980 | `05967f14a8b8d6a9e2637210807872ba4f30810987cb7f3017e10b186d07f6da` |
| `verifier/v8/criteria.md` | 1,495 | `9862c1f3cf73613ae8ce4975da90ff808341d096649afb5066c335674c69e6ee` |
| `verifier/v8/verify.py` | 5,365 | `c95a86504e8b62ade4bd7d20956415db2cfe95cc5897815dae3411c12af709c5` |
| `verifier/v8/verify.out.txt` | 1,676 | `dc1efe4502708d5c5bdcde48e2ec8528f127b8250474373b747a3cc9197a4e22` |
| `verifier/v8/verify.O.out.txt` | 1,676 | `dc1efe4502708d5c5bdcde48e2ec8528f127b8250474373b747a3cc9197a4e22` |

Every hash matches `SIDE24_v5_2026-08-02/09_PACKAGE_MANIFEST_V5.csv`.

I reran `verifier/v8/verify.py` using the existing side24 audit virtual
environment. Result: exit 0, 30/30 checks, `ALL_ASSERTIONS_PASS`; the fresh
transcript is byte-identical to both committed transcripts. The fresh copy is
`recovered_side24_v5/verifier/v8/verify.fresh.out.txt`.

Scope caveat: v8 checks that §G.9 retains RP-C/RP-S as OPEN and the theorem as
HOLD. It does not contain or test the later §G.9.1 planar determinant claim.

## 2. C030/C031 lower-bound set recovered from the q0 carrier

The pre-existing 5.91 MB file
`/workspace/scratch/eef1f5ced3ed/q0_machine_active.json` (SHA-256
`c3a93bd250c49e256467939bf714c18488fa506e87ae10636608fae99fb9cd38`)
contains all six JSON artifacts semantically and preserves the original
hash/size manifest for the full ten-file C030/C031 set.

Using the q0 verifier's own `_hydrate` serialization rule recovered four JSON
files byte-for-byte:

| File | Bytes | SHA-256 |
|---|---:|---|
| `c030_uncond.json` | 307 | `2f0f5802c9577c0e6dc7db910c66b378b6fabd40ee9b543487cc6d682b642e39` |
| `c030_lemmas.json` | 2,418 | `5afabb604b35b28ce3f89a99859aa08887d7c4579d563ecd86deb4635afbcf19` |
| `c030_assembly.json` | 131 | `49e197e978d950ac356ec96866760a7d0d8337341415e501b6ac7905dcd69990` |
| `c031_verify.json` | 1,603 | `83313c96d1610130451a7e936245a06e80d8cc84df00a20ae9e46e7aeef2a271` |

The two observed-update JSONs were also recovered semantically. Their key
order/whitespace did not reproduce the original bytes from the q0
serialization alone. The concurrent Drive recovery later supplied the exact
files, and normalized JSON comparison confirms the local reconstructions are
semantically identical:

| Exact file | Bytes | Exact SHA-256 |
|---|---:|---|
| `C030_Observed_Update.json` | 1,269 | `03490fe16e4e797f83792b2da758939874fd98a0b07741d62eb0e029d64af2f9` |
| `C031_Observed_Update.json` | 1,331 | `5df4205f2f496f28668c7f1ea22e2a1af9cde00a72cb39a6956b20b32c995063` |

The q0 carrier did not embed the three Markdown bodies, but did preserve their
exact identities:

| File | Bytes | SHA-256 |
|---|---:|---|
| `C030_CountingLemmas_Package.md` | 3,442 | `aed6683edaed6d483704a63cd321fcb300e329a26bf6c143042ead58a60bfc7b` |
| `C030_Freeze.md` | 2,673 | `ae20f4e3a239b17a6b747eba881d18cc7cf8e51e4d6a86eaefc40d9504fdef4b` |
| `C031_Freeze.md` | 1,739 | `e165821b1479f619af8ebdb1438460edfebe88f44c189553ca86f73f8fe1afbe` |
| `C031_LBRATE_Integration.md` | 13,690 | `e7998ef0d17d951bc89978f9fe32e510019059dd0650c8f0e0ae33273f40f32e` |

The concurrent Drive sweep recovered all ten exact carriers at
`/workspace/scratch/4c1cbf90475f/work/drive_recovery/C030_C031/`. The four
locally byte-reconstructed JSONs compare equal byte-for-byte to those Drive
copies; the two semantic-only reconstructions compare equal after canonical
JSON normalization.

Recovered numerical content includes:

- `Emax_b = 1.4135040354430306`
- `Esad_b = 0.9852262912564838`
- `rho_mx_12 = 0.04368529001235909`
- `C_wp_r3 = 0.2125711409212122`
- `C_mb_halfell = 0.7557785836498359`
- C031 total C1 perturbation `0.003629792782641913` versus gate `0.1307`
- far expected upcrossings `2.434`

Local reconstructions are under `recovered_c030_c031/`.

## 3. Other exact local evidence

### 3.1 EC-019 Path-B package-integrity replay

The alternate scratch contains an intact EC-019 T2 Path-B replay:

| File | Bytes | SHA-256 |
|---|---:|---|
| `GP-AUD-EC019-PATHB-20260730-v1.0.md` | 10,314 | `9b97779cc9affcd95c996f2ea1c45f60597a1465b7ba06b7eb8ab5833dc8ce52` |
| `cl219_cleanroom.py` | 18,604 | `74a228daf0971f93473d4739d3dfac4e0a018c75126e4c174508601d6cb7d85b` |
| `cl219_results.drive.json` | 30,800 | `90cad61f0f9412ad9218f301fa3eb78a99230f03e5e81947f6e933a962b611dc` |
| `cl219_results.replay.json` | 30,800 | `90cad61f0f9412ad9218f301fa3eb78a99230f03e5e81947f6e933a962b611dc` |
| `cl219_results.json.gz` | 6,262 | `6c58814a4e838641f16690b29af73356fe4b36f1eda24fd806cc64917a7dd38d` |

The Drive and replay JSONs are byte-identical; the gzip decompresses to the
same 30,800-byte JSON and hash. The receipt records 19/19 positive margins,
5/5 controls, and 9/9 mutations.

This is not the T1 determinant-identity core described in AO48-REC-034. Its
own receipt labels it a package-integrity replay and assigns zero additional
mathematical-independence credit. It is useful corroborating provenance, not
a substitute for AO48-AUD-030.

### 3.2 Unrelated formal-core R2

`research_formal_core_r2.zip` was recovered from the alternate scratch and
copied under this recovery directory. Its nine payload hashes all match
`SOURCE_MANIFEST.json`. A fresh `static_audit.py` run exits 0, reports 13
expected narrow Lean theorems, no `sorry`/`admit`/`axiom`, and explicitly
states that no Lean compilation was attempted because no Lean runtime is
present. This object is a GP-FOR-189 source-recovery successor and must not be
confused with the SIDE24 pre-review R2 ZIP.

## 4. AO48 records and the exact G.9.1 boundary

At the start of the local sweep, the attached SIDE24 packages and the
pre-existing alternate scratch contained no files or text hits for
`AO48-REC-034`, `AO48-AUD-030`, or the exact later `G.9.1` source. The
concurrent read-only Drive provenance sweep subsequently recovered:

| File | Bytes | SHA-256 |
|---|---:|---|
| `AO48-AUD-030.md` | 11,310 | `3c6493b6979da59e322bfb0e83d9356f70b287e81a48e4b545ae90835ecc9e91` |
| `AO48-AUD-033.md` | 10,154 | `895adbf10e3ebabe457a8aa293aed8be19562dd657d151622ce753de2aee83da` |
| `AO48-REC-034.md` | 7,453 | `b8ee5feea01db10c34a9d4ac7e6ccac82c2232d7aa69536a3bd6e377282f6116` |
| `AO48-REC-035.md` | 5,137 | `f8359b1cf389c899f0b8393281d9f2b513770bfe0c60952e6455bfdd11ddef6a` |

Those files now live under
`/workspace/scratch/4c1cbf90475f/work/drive_recovery/` and establish the
receipt-level provenance that the initial local corpus lacked.

AO48-REC-034 restates the recovered cores: EC-019's polynomial determinant,
the six exact Sylvester minors, the `2.098×10^28` reserve audit with more than
70 orders of margin, the exact `Q ≡ 1` identity, and three acceptance
fixtures. AO48-AUD-033 is also explicit that the relayed all-face G.9.1
closed form was wrong away from the transverse face and that RP-C/RP-S
remain open.

Therefore the honest recovery statuses are:

- **V5 §G.9 source:** byte-exactly recovered.
- **Historical v8 source/transcripts:** byte-exactly recovered and rerun.
- **SIDE24 R2 ZIP:** byte-exactly recovered and internally verified.
- **AO48 audit/receipt records:** byte-exactly recovered from Drive by the
  concurrent provenance sweep.
- **Exact later G.9.1 prose and its own verifier:** not found byte-for-byte;
  only its relayed claim, independent audit, receipts, and a corrected
  mathematical reconstruction survive.

## 5. Exhaustive-search coverage

The read-only search covered:

- all files under the current scratch workspace;
- the 2,564-file alternate scratch `/workspace/scratch/eef1f5ced3ed`;
- every attached/nested ZIP then present (19 ZIP instances, 10 unique hashes
  at initial inventory), with safe-name validation and extraction of every
  unique archive;
- 362 PDF/DOCX instances, deduplicated to 117 unique binary documents and
  text-extracted successfully with zero extraction failures;
- every gzip carrier found under `/workspace/scratch`;
- the 49 source files embedded in `q0_verify_active.py`;
- filename and content searches for AO48/AUD/REC identifiers, G.9.1,
  `Q ≡ 1`, Sylvester/base-mass language, EC-019, C030/C031, fixtures, and the
  quoted reserve values.

Machine-readable inventories and search logs are retained in this directory:

- `ZIP_SHA256_INVENTORY.txt`
- `ALL_ZIP_ENTRIES.tsv`
- `UNIQUE_ARCHIVE_EXTRACTIONS.tsv`
- `OTHER_WORKSPACE_FILE_INVENTORY.tsv`
- `PDF_DOCX_INVENTORY.tsv`
- `BINARY_HASH_PATHS.tsv`
- `BINARY_TEXT_EXTRACTION.tsv`
- `EEF_TARGETED_TEXT_HITS.txt`
- `GZIP_TARGET_HITS.txt`

The search result supports recovery and disambiguation, not theorem
promotion. In particular, neither the historical v8 packaging verifier nor
the reconstructed planar G.9.1 identity closes the side-24 RP-C/RP-S facewise
compactification work order.
