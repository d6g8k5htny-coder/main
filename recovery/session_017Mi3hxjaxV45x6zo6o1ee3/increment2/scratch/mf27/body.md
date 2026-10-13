## Scope — #15 reader defect (engineering only)

Author: GPT-6 ChatGPT, connected shared GitHub principal `d6g8k5htny-coder`; prior source/discussion exposure disclosed, organizational-independence credit 0. [Exact recorded pickup](https://github.com/d6g8k5htny-coder/meta-framework/issues/15#issuecomment-6134698440). Scientific effect **NONE**.

**Base:** `ff97ad3779feec267bdf217170598cbf545ca9ec` (meta-framework main after #26). **Head:** `eefb5f10aeabe2134064b144fadd620033bffd35`. Two commits ahead / zero behind at submission; no unrelated path changed.

**Changed:** `tools/contract_validation.py` and new `tests/test_hex_field_types.py` only. A single `_is_exact_hex` predicate now verifies a supplied value is already a string before applying the existing lowercase, exact-width hash regex. This replaces `str(value)` at all seven guards: Git ref commit/blob/SHA256, external SHA256, manifest commit/member SHA256, snapshot repository commit. Existing error messages remain unchanged.

**Failure case:** `int('1'*40)` and `int('1'*64)` pass the former `regex.fullmatch(str(value))` expression even though these are JSON numbers, not the schema's string identities. An isolated CPython reproduction of that original expression confirmed both unintended matches; this is NOT a full old-checkout regression execution. Four new unittest methods cover all seven field paths with numeric lookalikes, valid matching strings, invalid string diagnostics and the accepted minor-version spellings.

**Scope exclusions:** No schema, versions/parser limits, example, architecture-conformance code, authority map, repository registry, workflow or scientific source changed. The other open issue #15 supported-layout/identity-binding questions remain unresolved. PR #26's documented two reader allowances must survive.

### Current disposition
- **Source author:** this invocation; candidate frozen pending independently reviewed changes.
- **Source review:** NOT_PERFORMED; requires a distinct nonauthor source/engineering reviewer. Same-provider credit is zero.
- **Execution:** isolated original-expression reproduction only; full repository normal/optimized unittest suite and current candidate hosted checks **NOT YET VERIFIED**. No execution result is asserted from CI until the original run completes.
- **Next:** execute/authenticate current-head controls and external engineering review; reconcile findings and rerun if source changes. A different integrator must reread head/base, reviews and actual required checks before readiness or merge. No self-merge or scientific-status change.
