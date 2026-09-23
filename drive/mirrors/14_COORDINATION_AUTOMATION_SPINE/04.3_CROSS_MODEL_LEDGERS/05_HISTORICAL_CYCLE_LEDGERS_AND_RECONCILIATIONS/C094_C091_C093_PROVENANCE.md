# C094 C091 → C093 Inter-Release Provenance Audit

| File | Classification | C091 SHA | C093 SHA | Cause / disposition |
|---|---|---|---|---|
| `Q0_C091_PROOF_GATE_REDUCTION.md` | CONTROL-CHARACTER-REPAIR | `128ab070189f` | `10e1f4e1fa73` | hidden form-feed interpretation after LaTeX \frac caused split 'rac' lines; C093 repaired the rendered equations; semantic equations restored; no theorem-status change |
| `C091_CORRECTION_LEDGER.md` | IDENTICAL | `b0f806b3ff5c` | `b0f806b3ff5c` | byte-identical across frozen bundles; no ledger action beyond identity record |
| `q0_c091_canonical_contract.json` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |
| `q0_c091_contract_checker.py` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |
| `q0_c091_contract_check_report.json` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |
| `q0_c091_gate_reduction.py` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |
| `q0_c091_gate_reduction_report.json` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |
| `periodized_bf_matrix_transfer.py` | PORTABILITY-EDIT | `91cb008e45df` | `3b1c44449f52` | hard-coded /mnt/data output/module paths replaced by paths relative to __file__; formatting-only changes where present; algorithm and declared mathematical result unchanged |
| `periodized_bf_matrix_transfer_report.json` | IDENTICAL | `75f423ed03bd` | `75f423ed03bd` | byte-identical across frozen bundles; no ledger action beyond identity record |
| `gaussian_corridor_certificate.py` | PORTABILITY-EDIT | `d8bd3f12b341` | `46f373df22bb` | hard-coded /mnt/data output/module paths replaced by paths relative to __file__; formatting-only changes where present; algorithm and declared mathematical result unchanged |
| `gaussian_corridor_certificate_report.json` | DIAGNOSTIC-REEXECUTION | `b6a70cfa846b` | `8b874396913a` | SciPy multivariate-normal orthant diagnostics are numerical algorithm outputs without a frozen random integration state; re-execution changed only diagnostic CDF leaves; all Chernoff candidates, covariance inputs, counterexample labels, invalid-product verdicts, and H4 replacement gates are unchanged |
| `palm_weighted_gaussian_chernoff.py` | PORTABILITY-EDIT | `e2a308393bcd` | `6a32d466d028` | hard-coded /mnt/data output/module paths replaced by paths relative to __file__; formatting-only changes where present; algorithm and declared mathematical result unchanged |
| `palm_weighted_gaussian_chernoff_report.json` | IDENTICAL | `1a46553e7b57` | `1a46553e7b57` | byte-identical across frozen bundles; no ledger action beyond identity record |
| `bf_two_critical_value_gap.py` | PORTABILITY-EDIT | `d0324cdc9b9a` | `0a0b5e24bebe` | hard-coded /mnt/data output/module paths replaced by paths relative to __file__; formatting-only changes where present; algorithm and declared mathematical result unchanged |
| `bf_two_critical_value_gap_report.json` | IDENTICAL | `acf6691a68d9` | `acf6691a68d9` | byte-identical across frozen bundles; no ledger action beyond identity record |
| `q0_c091_corridor_product_audit.png` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |
| `q0_c091_bonferroni_budget.png` | C091-ARCHIVE-ONLY | `—` | `—` | C093 closeout selected a provenance subset; the complete file remains immutable in the verified C091 bundle; not overwritten or lost; C091 bundle remains the authority for this artifact |

## Same-size corridor report

- changed scalar leaves: **12**
- only orthant diagnostic fields changed: **True**
- adjudication object unchanged: **True**
- maximum relative diagnostic shift: **4.174e-07**

The exact Gaussian/Chernoff inputs and every terminal adjudication remain unchanged. The varying fields are explicitly labeled diagnostics in the report; they are not theorem inputs.

## Governance conclusion

All changed same-name files are now ledgered. The C091 bundle remains the immutable authority for its own bytes; the C093 variants are new-release portability or repair variants, not silent overwrites.