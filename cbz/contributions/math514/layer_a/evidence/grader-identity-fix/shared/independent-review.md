# Independent loader identity review

Verdict: **AMEND** at kernel runtime `c77ee122` and layer-a runtime `50c0e2d6`.
The existing alias/subclass hook repair and class/function/code checks are useful,
but two earlier population transformations still produce false green results.

- **IDENTITY-REVIEW-F1 / P1:** TestCase constructors execute before `remember()`
  captures the declarations. An ordinary custom `__init__` can alias the second
  declared method to the first. Both runners report SURVIVED/exit 0 with two
  distinct names, no invalid population, and `first, first` body traces. The
  retained failing second original body is absent. Freeze before construction,
  or refuse unsupported constructors/metaclasses before instantiation.
- **IDENTITY-REVIEW-F2 / P1:** `dir(module)` invokes a module-owned `__dir__`.
  Returning only `Visible` hides a declared `Hidden` TestCase from the canonical
  population. Both runners report SURVIVED/exit 0 with only the visible passing
  body. Enumerate actual module dictionary declarations or explicitly refuse
  these hooks; handle custom TestCase metaclass directory hooks conservatively.

Full source-bound fixtures, structured receipts, stdout/stderr and body traces
are under `review-probes/constructor-alias/` and
`review-probes/module-dir-omission/`. These four new normal-mode executions check
newly discovered obligations; no completed test, scanner, Lean or mathematical
verification was rerun.

The independent evidence-binding check is clear for all 32 prior guard
executions (eight themes per tool/mode): all are actual assertion kills with zero
errors and matching runtime/test copies, receipts and logs. Passing affected
method unions are kernel 27 and layer-a 19 per mode. The original failed layer
normal run remains visible; its changed canonical-population expectation is
superseded only by the targeted stricter test receipt, not relabeled green.
These valid prior observations do not cover the two new counterexamples.

This review concerns trusted ordinary Python test-source/runtime behavior. It
claims no arbitrary-code sandbox, complete scanner correctness, mathematical
promotion, provider-distinct review or federation CBZ-0 completion.
