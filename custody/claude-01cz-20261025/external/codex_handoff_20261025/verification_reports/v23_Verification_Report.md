# v23 independent release verification

**Passed for the exact candidate within code, custody and offline replay scope.** This audit does not add a mathematical peer review or a formal certificate.

Candidate: `Universal_Law_v23_RN3_Recovery_Addendum.zip` (12,833,946 bytes).

SHA256: `014c732bf40c8433f0c318580032fc57aa71e3b054549ec0a3aad539d4c8c437`

## Verified execution and custody

- All 70 regular ZIP members have safe unique paths and valid CRCs. All 69 SHA256SUMS rows and 68 nested inventory rows match, with complete membership.
- Exact recovered RN3 ZIP: 2,706,421 bytes, 78 members, all 77 payload hashes and 22 original Python sources checked.
- Exact official wheel: 10,125,938 bytes, 126 entries/113 regular files, all 112 hashed RECORD rows and complete coverage checked. All 112 payloads independently match the previous installation. Stored PyPI metadata/acquisition hashes match.
- All 15 author-review inputs and 12 frozen review artifacts remain exact. The mathematical review is bound to SHA256 `b8c04adfb1cba7302adc8f0ec1f15efed5010e0d26426c46fbce1fa8a09f0ff2`.
- Two full offline profiles passed in 24.160 s and 22.792 s. Each ran RN3 normal/`-O` (13,604 cells, 294 audit checks, 24 full-Gram laws per mode), H3 normal/`-O` plus two rejecting controls, and the separate 512-bit spectral H3 checker. Fresh H3/spectral receipts exactly equal the frozen reviewed receipts.
- All 19 framework tests and 38 external code/safety probes passed. Both output directories are distinct and 0700; all 40 first-run regular files retain their hashes after repeat and rebuild.
- The rebuilt ZIP is byte-identical. Complete source/manifests and canonical data remain unchanged, including sealed v19/v20/v21/v22. v22 remains `82536fa4ed628a8f96cd4b8734a867c50f69b1d28cb55d810053cb449448754c`.

Execution used CPython 3.12.14/Linux x86_64, hidden original `/workspace` and host site packages, private network/PID namespaces and disk-backed temporary storage. The verified wheel alone supplied python-flint 0.9.0 / FLINT 3.6.0 / GMP 6.3.0 / MPFR 4.2.2 from a private temporary prefix. No old venv or network was used. The explicit interpreter shim adds `-S -B` and preserves `-O`; original mathematical scripts stay byte-exact. Child resources and wall time are bounded, process groups are terminated on timeout, and logs use exclusive regular files. This remains a trusted-host reproducibility runner.

## Numerical scope and remaining limits

At the reviewed d=2, L=24, r=1/20, b=6/5, gap mark 1/6, axis-pin rung, the fresh source-bound receipts reproduce **Z>0.0077592917375327858**, exceeding the old imported 0.0077592917375327855 floor, and **I_far < 2.83312 r³**. The integrated region is **[-12,12]² outside the disk of radius 5**, with area 24²−25π. The larger 5≤|y|≤17 Euclidean annulus is the enclosure cover only. These statements retain the exact-law and trusted Arb assumptions of the separate review.

No all-small-r/radius band, other orientations/marks/dimensions, near annulus, old five-factor fourth-order composition, full theorem assembly or kernel certificate is established. The new H3 proof is not recovered original H3 code and does not authenticate separate historical ledger identities. The 104 canonical records and 108 prerequisites remain unchanged; new admissions, complete formal certificates and organizational-independence credit remain **zero**.

## Fixed initial defects and transparency

The initial wrapper's output-link traversal and intermediate payload-internal output were confirmed using disposable targeted fixtures deliberately stopped before native numerical execution. They are not replay PASS records. Final code refuses these paths, protects existing output targets and rechecks the complete source payload before PASS. Startup/resource handling was also replaced. Initial/intermediate code and outcomes remain separate from final candidate evidence; the implementation's first spectral-layout setup failure is preserved in the product history.

The H3 runner deliberately executes two recorded in-memory negative mutants; their nonzero failures do not claim the mutated physical theorem is false. Source files and reviewed code remain unchanged. The private reviewer layout receives the fresh, hash-bound author receipt without editing the checker. No passing finite script is treated as a new mathematical peer review, canonical admission or formal certificate.

Other platforms/runtimes, arbitrary scripts and adversarial same-user behavior remain unqualified. This audit checked preserved PyPI metadata rather than independently repeating HTTPS acquisition. Full commands, hashes, initial/final fixtures and limits appear in Verification_Report.json and the portable evidence archive.
