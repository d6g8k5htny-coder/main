# Independent scanner review

Reviewed frozen scanner working bytes at SHA256 `679988ace9817638867f84ada19d7171fc0b5749f313f5fb5b2a45a26e6d2fde`, on repository base `27c89b05bd2503c580b692f57146d608a7f0f668`. Candidate paths were untracked. The author confirmed this freeze before findings were delivered. Author repairs subsequently changed working bytes; this review is not a verdict on those successor bytes.

Two P1 false negatives were observed using official Lean 4.34.1 and CPython 3.11.16. Both valid source fixtures compile with exit0 and expose a forbidden axiom through `#print axioms`, while the scanner CLI with `--fail-on all` exits0 with zero findings and lexical errors.

1. `macro-string.lean`: builtin `throwError` interpolation containing a nested ordinary quoted string hides an explicit `sorry` term. `def f : CoreM Unit := throwError "bad {let x := "safe"; (sorry : Nat)}"`. Actual axiom set includes `sorryAx`. The primary ordinary-string scan stops at the nested quote before the interpolation hole closes; its later string then hides the placeholder. Balanced-hole postprocessing does not see the complete region.
2. `numeral-command.lean`: `def harmless := 0axiom hiddenPremise : False` followed by `theorem bad : False := hiddenPremise` is valid Lean. Actual axiom set for bad is `[hiddenPremise]`. The numeric scan consumes arbitrary alphanumeric suffixes and therefore swallows the axiom keyword in `0axiom`. Numeric token boundaries must follow Lean rather than generic isalnum.

Exact source fixtures, compiler output, scanner JSON and scanner output are adjacent. The preliminary `numeral-adjacency.lean` attempt failed Lean elaboration/parsing; it is retained as a failed probe and is not the valid counterexample. No production source, tests, scanner working bytes, refs or remote metadata were edited by this reviewer.

Both concrete findings were sent directly to the auditor author, root and receipt reviewer. The author confirmed the first and began source repairs; no final successor-source PASS is claimed. Kernel and semantic coverage remain separate from this finite lexical review.
