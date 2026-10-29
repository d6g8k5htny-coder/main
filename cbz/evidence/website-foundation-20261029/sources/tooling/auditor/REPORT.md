# CBZ-0 finite auditor handoff

Observed 2026-10-29T17:10:50.208433+00:00. Tool and tests are new, unstaged files; no existing product, Git, remote, or settings writes were made by this worker.

G0 PASS: 43 behavioral tests pass; 43 tests pass under optimized Python; all 12 deliberately source-mutated scanner variants fail through assertions. The declared threshold is exactly 1/1, checked by integer cross-multiplication (12/12). Three additional direct negative controls invoke the selftest CLI on genuinely mutated scanners (disabled placeholder detector, ignored hidden files, suppressed coverage failures); each returns FAIL/nonzero. Receipts: G0.json, optimized-tests.log, selftest-negative-controls.json.

G1 Main INCOMPLETE (exit 2): 2,827 tracked working-tree files enumerated and read; zero active lexical findings; nine explicit coverage gaps (eight archive/container files and one notebook). No claim of total Layer A closure, mathematical soundness, nonvacuity, theorem alignment, transitive axiom closure, or complete regression is made. Root owns the joint semantic gate and full G5 after source freeze.

Tool SHA256: `d18f74984b6a5d4c2834119b9ee804d6e71942f306a7801035acdb04f66a7d47`. Tests SHA256: `347fb2ee14f6ac69e67604f22cd29995bf18724fe3f12d75b29faf61e728247b`. G0 and G1 bind the exact scanner bytes; G0 also binds the tests. G1 inventory SHA256: `17c04c072480944d1316bd223bd48a9c441cf6578a769ec153c78d2a4ef21dda`.

Run from the repository root:

```sh
python -B tools/cbz_audit.py selftest
python -B tools/cbz_audit.py scan --root .
python -B -O -S -m unittest discover -s tests -p test_cbz_audit.py
```

Optional `--registry PATH` and `--quarantine PATH` schemas are documented in `python -B tools/cbz_audit.py --help`. Manifest JSON rejects duplicate keys, fractional/nonfinite JSON numbers, unknown or duplicate paths, wrong roles and mismatched SHA256. All mathematical threshold decisions use exact integers. A valid quarantine annotates rather than deletes hits; missing independent isolation/dependency evidence remains visible.

The tool deliberately distinguishes static_result from result. static_result may PASS for clean supported source, but result remains INCOMPLETE until the external joint gate supplies and evaluates real semantic evidence. Empty trees cannot yield a Layer A PASS. Unsupported proof formats, archive containers, LFS pointers, unreadable/nonregular files, symlinks, submodules, index conflicts, and untracked proof/archive candidates are explicit gaps.

Demonstrated capability coverage:

| Scope | Positive and negative coverage |
|---|---|
| Lean lexical checks | All four placeholder/trust keywords, eight trust tokens; nested comments, escaped/ordinary/raw strings, Unicode names and quoted identifiers; unterminated forms and interpolated strings become gaps. |
| Enumeration and provenance | Hidden paths/spaces, complete tracked inventory, subdirectory normalization, exact source hashes; absent files, invalid UTF-8, symlinks/ancestors, nonregular files, submodules, conflicts and LFS pointers remain visible. |
| Proof/archive candidates | All 16 declared source/artifact suffixes and manifest-added sources; magic/suffix archive recognition; ordinary noncandidate data is retained without fabricated proof findings. |
| Certification classification | Explicit Python/Lean certifiers show floats; rational arithmetic and strings do not; unclassified/explicit UI floats do not become mathematical violations; unsupported or unparsable certifiers are gaps. |
| Quarantine/manifests | Exact matching annotations preserve hits; changed hashes, traversal, duplicates, missing rationale/attempts or owner != none fail. |
| Gate truth | Active findings fail; empty scans remain incomplete; CLI emits JSON/nonzero; mutated scanner rejection and exact finite mutation threshold are observed. |

Finite mutation operators: disable_placeholder_detector, disable_trust_detector, ignore_hidden_files, suppress_coverage_failures, suppress_active_findings, disable_float_detector, ignore_quarantine_hash, accept_empty_semantic_closure, disable_archive_detection, ignore_untracked_proofs, misclassify_all_python_as_certification, hide_quarantined_findings.

A selftest-quality correction is preserved in G0-pre-gap-assertion-failure.json: the first new gap-hash test indexed an empty list for two mutants, producing test errors; the gate refused to count those as valid kills (10/12). The corrected test first asserts the expected gap count, so both mutants now fail by assertions. Neither the detector nor threshold was weakened, and no failing test was removed.

Layer B decisions: standard-library-only read-only scanner; no archive extraction subsystem or semantic green-by-regex; exact hash manifests; preserve quarantine hits; external aggregation for kernel/nonvacuity/status/isolation receipts. These choices keep this tool finite and useful without misrepresenting its scope. No mathematical status promotion occurred.

Actionable coverage closures, all currently UNMET:

| Kind | Path | Exact working-byte SHA256 | Required independent closure |
|---|---|---|---|
| archive | `audits/formal_enforcement/2026-09-27/Main188_landed_formal_36358139870.zip` | `4fc0ac1e58a6340802620deab0cc20587b278b0c4e7aa47958a5da3440216aa7` | Inventory every archive member against this outer hash, bind member paths/hashes and independently scan/classify proof/certification content; preserve nested or unreadable residue. Alternatively explicitly quarantine exact bytes with isolation and CLOSED-dependency audit. |
| archive | `audits/formal_enforcement/2026-09-27/Math96_landed_downstream_36359933550.zip` | `43bd4650143bb6a9b59871335c04c1aa6cb324391cd89a5e6509675eb9d63db2` | Inventory every archive member against this outer hash, bind member paths/hashes and independently scan/classify proof/certification content; preserve nested or unreadable residue. Alternatively explicitly quarantine exact bytes with isolation and CLOSED-dependency audit. |
| archive | `audits/formal_enforcement/2026-09-27/Math96_landed_formal_36359933550.zip` | `bc0342134ba8c05d7d45d827961a721b468b8d7bc315bf4a138809b9262e0f4a` | Inventory every archive member against this outer hash, bind member paths/hashes and independently scan/classify proof/certification content; preserve nested or unreadable residue. Alternatively explicitly quarantine exact bytes with isolation and CLOSED-dependency audit. |
| proof | `docs/notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb` | `e348db8e9d667caff2a8da1d85ffd5ad599db34b34e542d9662d0bc1edc84024` | Parse notebook cells, outputs and attachments against this hash; classify mathematical certification/proof content, audit executable cells and statement status; or exact quarantine plus isolation/CLOSED-dependency audit. |
| archive | `docs/reports/atlas-evidence-accessibility-20261026/qualified-observation-5596.json.gz` | `ba152ef91077370c703d92275593b03f63d6b69dfba58221c64d2d775843ab19` | Inventory every archive member against this outer hash, bind member paths/hashes and independently scan/classify proof/certification content; preserve nested or unreadable residue. Alternatively explicitly quarantine exact bytes with isolation and CLOSED-dependency audit. |
| archive | `docs/site/evidence/side24-main-3b39/verified-execution.zip` | `80ac88cf770afbadd89f4801a4692d6b6540a9269dfb916c249d491c2b41f737` | Inventory every archive member against this outer hash, bind member paths/hashes and independently scan/classify proof/certification content; preserve nested or unreadable residue. Alternatively explicitly quarantine exact bytes with isolation and CLOSED-dependency audit. |
| archive | `docs/site/snapshots/38ac44c585feb2f9a480d8e54ad1a7fe56ad35aff19e2eab929861ea81974644/archive.zip` | `497d72cd31401891042fcfa8f75eb9b121a90cf2771bb3d6df43c18974829a57` | Inventory every archive member against this outer hash, bind member paths/hashes and independently scan/classify proof/certification content; preserve nested or unreadable residue. Alternatively explicitly quarantine exact bytes with isolation and CLOSED-dependency audit. |
| archive | `docs/site/snapshots/7c583e28878d2bf2c94bf37325820bd5eb1e7f3f557e8a78dd27881a27f6548c/archive.zip` | `9f8542fd934649ffbda1a154e01576673b91f3bdb40495bb097a52a97eebcd34` | Inventory every archive member against this outer hash, bind member paths/hashes and independently scan/classify proof/certification content; preserve nested or unreadable residue. Alternatively explicitly quarantine exact bytes with isolation and CLOSED-dependency audit. |
| archive | `reviews/zeta_fold_slice_c_openai_20261026/fresh/raw/rs_cheb_coeffs.npz` | `7b86066629cb2be07d0c1f98cb8ed74357b62ddd2480ad7d9d063c1be9ed10fc` | Inventory every archive member against this outer hash, bind member paths/hashes and independently scan/classify proof/certification content; preserve nested or unreadable residue. Alternatively explicitly quarantine exact bytes with isolation and CLOSED-dependency audit. |

Archive gaps are not automatically mathematical violations. The NPZ coefficient container and gzip observation record need explicit purpose/dependency classification before any such conclusion. A filename, historical audit label, or existing green receipt is not a substitute for checking its actual contained bytes.

Required semantic receipts remain: kernel_and_declared_axioms, status_index_and_nonvacuity, statement_alignment_and_unweakened_bounds, test_preservation_and_regression, certification_scope, dependency_and_dynamic_syntax_review. Quarantine adds quarantine_isolation_and_closed_dependencies. Root can independently account for these gaps in a joint gate while preserving the original G1 INCOMPLETE result and exact boundaries.

Source ownership ends at tools/cbz_audit.py and tests/test_cbz_audit.py. Supporting evidence and this tool registry entry are under /workspace/cbz0/auditor. Root must stage/integrate these new files and rerun the final joint gates after all source and governance changes are frozen.
