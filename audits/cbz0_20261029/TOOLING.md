# CBZ-0 tooling registry

Every tool built or changed for Clean Baseline Zero. Each entry says what it does, how to
run it, how it is itself verified (Layer A8: an auditor must be auditable), and which CBZ-0
gate it serves. A tool without a passing self-test is not listed as usable.

## lean_layer_a_scan — Lean Layer A source scan

- **Where:** `Math-` `tools/lean_layer_a_scan.py`; tests `tests/test_lean_layer_a_scan.py`;
  workflow `.github/workflows/layer-a-lean-scan.yml`.
- **Serves:** G0 (auditor self-test), G1 (Layer A static audit); CBZ-0 criteria 4 and 9.
- **What it does:** lexes Lean 4 source, skipping line comments, nested block and doc comments,
  strings and character literals, and reports any code identifier whose name or dotted
  component is forbidden: `sorry`, `admit`, `sorryAx` (A1); the `axiom` command,
  `native_decide`, `ofReduceBool`, `trustCompiler`, `implemented_by`, `extern`, `unsafe` (A2);
  `Float`, `Float32` (A3). «Quoted» identifiers are checked too. An unterminated comment, string
  or quoted identifier is a lex error (exit 2), never a silent pass.
- **Run:** `python3 -I tools/lean_layer_a_scan.py --root <repo>` scans every git-tracked `*.lean`
  outside `.lake/`. Pass paths to scan only those. `--json` gives a machine-readable report.
  Exit 0 = no findings, 1 = findings, 2 = lex error.
- **Self-verification:** `python3 -B -S tests/test_lean_layer_a_scan.py` (and with `-O`).
  Covers every forbidden name in code; the same text in comments, docstrings and literals;
  substring non-matches (`sorryCount`, `admits_bound`); line numbers across multi-line
  comments and strings; malformed input; CLI exit codes; default scope (tracked files only,
  `.lake/` excluded); the repository's own tree (G1). A mutation test rewrites the scanner
  15 ways (drop each forbidden token, break nesting, ignore escapes, freeze line numbers,
  swallow lex errors, return 0 on findings, include `.lake/`, drop «» names). Each mutant
  must fail the suite on a real test failure, and an unmutated harness control must pass,
  so an import failure cannot pass for a kill.
- **Limits (honest boundary):** static only. It cannot see an axiom or `sorry` reached through
  an imported dependency, a macro that expands to one, or `decide` that is sound but slow.
  Those need a kernel build plus `#print axioms` (G2–G4), which Math- `formal/gate.py`
  provides only for the `formal/` package. Lean raw strings `r#"…"#` are not lexed specially.
- **Maintenance owner:** none; self-contained, standard library only.
- **Observed result (first run):** 0 findings, 0 lex errors in all 162 tracked Lean files of
  Math- and all 4 of main.

## make_inventory — Phase 1 inventory renderer

- **Where:** `main` `audits/cbz0_20261029/make_inventory.py`; inputs `audits/cbz0_20261029/branches.tsv`, `audits/cbz0_20261029/prs.json`,
  `audits/cbz0_20261029/gates.json`, `audits/cbz0_20261029/heads.json`.
- **Serves:** Phase 1 exit criterion (INVENTORY.md with zero unknown entries); G10 evidence.
- **Run:** `python3 -I audits/cbz0_20261029/make_inventory.py audits/cbz0_20261029` rewrites `audits/cbz0_20261029/INVENTORY.md` from the inputs.
- **Self-verification:** none beyond determinism: it only formats observed inputs and computes
  counts. It certifies nothing about Layer A, so A8 does not apply.
