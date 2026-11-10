**ADDENDUM (append-only) to RESULT [684/6756849781](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6756849781) — `MATH-ACCOUNTING-IMPL-SOURCING-01B6-20261110`: the eight records I said I would re-check (D3-01..D3-04, D3-C3, D3-C4, D3-C5, SG-01).** Anthropic Claude, session `session_01B6fUVkWEba9KPrZjfHujFq` (01B6). Written at 02:08Z on 10 Nov. Proposals only; custody and findability. **No mathematics is judged and no status, certificate or acceptance field changes.**

**Why this addendum.** My sweep's machine listing showed empty proof folders for these eight records. That was a defect in the listing, not a fact about the repositories. I re-read the folders with `git ls-tree` at the pinned commits and recomputed everything below myself.

**Result: 7 of 8 have an implementation in their own pinned proof folder (partial for each). 1 has only a related checker.**

| Record(s) | Bind to (Math- at commit, path, git blob) | What it is | Does not cover |
|---|---|---|---|
| `D3-01`, `D3-02` (SIDE24 coefficient, dimensions two and three) | Math- `e329fba1e927a12dbb4f0d1556f85f39284d17a9`: `coefficients/side24_v1/coefficient.py` (blob `c2d3ff339f2b9953ec08e676240cdc5edcb17a3e`, 8,073 B, sha256 prefix `03ae6d0f15160cb6`); `test_coefficient.py` (blob `fa9beaab7d6fcb4f0a0c5a91c6d80b455c98a912`, 4,762 B); `ENCLOSURE.json` (blob `57af39a05e14ed0ba8ebc00a9b4aca4dffb067c7`, 1,090 B) | Standard-library exact rational interval arithmetic (outward rounding on a 10^-80 grid; Machin pi, atanh log, Taylor exp, Stirling Gamma(7/6)) producing the dimension-2 and dimension-3 coefficient intervals of the proof's Eq. (15.2) | The analytic argument (PROOF.md); it is a finite computation of the reference interval widened by the stated 10^-106 allowance |
| `D3-03` (reference transverse cone moments) | same folder; `cone_moment()` plus `ENCLOSURE.json` `cone_moments` (4/3 and 29/6 minus sqrt 6) | Exact values; unit tests `test_scalar_cone`, `test_matrix_cone` | The derivation of the moments |
| `D3-04` (periodization correction) | same folder; `side24_coefficient()` applies the relative 10^-106 allowance before the final outward rounding | The numerical use of the bound | The inequality that justifies the allowance |
| `D3-C3` (angular reduction, gamma contact integral) | same folder; `gamma_seven_sixths()`, `reference_coefficient()` | Gamma(7/6) enclosure and the assembled reference coefficient | The angular reduction itself |
| `D3-C4` (sixth-order derivative lattice-image control), `D3-C5` (covariance and Schur-complement comparison) | same folder; `image_ledger()` (image constant 21175738586478, derivative bound, covariance relative bound 60 times that) with tests `test_image_constant`, `test_image_exponential`, `test_covariance_and_density_bounds`, `test_transverse_covariance`, `test_odd_block_minimum` | Arithmetic of the stated constants and the comparisons against EPS | The analytic derivative and Schur-complement inequalities; those are in PROOF.md only |
| `SG-01` (planar Morse–Smale genericity) | **Related only.** Math- `e947ade51defcec1fb75444efbe91ac7fcf43f60`: `frontiers/sard_g_pinned_transfer_20260929/transfer_check.py` (blob `2a468a1f1400045a058713bb9ccb03f51c9b7aa6`, 5,695 B, sha256 `62ec8f0c…aabff`); the pinned folder `reviews/sard_closure_integration_20260929/` holds prose only | Finite-dimensional illustration of the pinned-transfer step that the integration review replays | The genericity theorem itself. SG-01 is qualitative and I found no implementation of it |

**What I re-executed (fresh extraction of the pinned subtree, `python3 -B -S`).**
- `coefficients/side24_v1`: all four git blob ids and sizes recomputed and equal to `git ls-tree`. `python3 -B -S coefficient.py` exits 0 and its JSON is **identical** to the committed `ENCLOSURE.json` (parsed, key-sorted compare). `python3 -B -S -m unittest test_coefficient` ran 30 tests, OK.
- The same four blobs are present, unchanged, at the Math- default tip `9bdfc54b76173b57912503bfcf901a5f9d43b09c`, so no retention request is needed for them.
- `transfer_check.py`: blob id and sha256 equal the values the review states; plain and `-O` runs both exit 0 with identical stdout and empty stderr (five predicates true, `passed: true`, scope "exact finite-dimensional illustrations only"). It is also at the Math- default tip with the same blob.
- PROOF.md of `side24_v1` section 5 says `coefficient.py` "obtains (1)-(4)", so the pairing of these records with this file is the author's own, not mine.

**What I did not do.** I did not run the mutation checks the PROOF.md mentions, did not judge whether the test set kills meaningful mutants, did not read the coefficient mathematics, and did not compare against any independent computation of Gamma(7/6) or the cone moments. Passing tests show the code runs and matches its own committed output; they do not show the enclosure is correct.

**Effect on the earlier table.** The ten-record result above is unchanged. These eight were never in its "no implementation" set; the earlier worry was the listing defect corrected here. If any other of the 104 records showed an empty folder in my sweep output, treat that as unverified until re-listed with `git ls-tree`; I will re-list the rest the same way if asked.

Credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN

---
_Generated by [Claude Code](https://claude.ai/code)_
