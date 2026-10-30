# G11 auditor — bounded census, not certification

## Purpose

`tools/cbz/g11_census.py` reads tracked blob bytes from one Git revision and reports review triggers for the owner-defined G11 tooling-hygiene criterion: Python stubs, code-comment markers, no-op workflow run steps, and tool documentation/exercise references. It is a triage instrument, not a proof, a Layer A certificate, or a federation-wide G11 admission.

## Commands

From the repository root, using only the Python standard library:

```sh
python3 -B -S -m unittest tests.test_g11_census
python3 -B -S -O -m unittest tests.test_g11_census
python3 -B -S tools/cbz/g11_mutation_selftest.py
python3 -B -S -O tools/cbz/g11_mutation_selftest.py
python3 -B -S tools/cbz/g11_census.py --rev <commit> --json census.json
```

The census exits 0 for no detected production finding, 1 for a detected production finding, and 2 when the requested Git revision cannot be read. A zero exit is only the result of this implementation over its stated scanned paths and rules.

## Soundness boundaries

- Stub exemptions are bound to imported `abc.abstractmethod`, `typing.overload`, `typing_extensions.overload`, `abc.ABC`, `typing.Protocol`, `typing_extensions.Protocol`, actual built-in exception classes, and the named `http.server` request-handler bases. Name suffixes and locally defined look-alikes do not grant exemptions.
- Python test/tool exercise evidence uses parsed imports and recognized subprocess invocations, not arbitrary comments or string mentions. JavaScript test imports are read from import declarations after comment stripping. Workflow exercise evidence is read from run commands, not YAML comments.
- YAML comments are scanned for marker words. Empty and comment-only `run` blocks are reported as no-op steps.
- The mutation self-test preserves the original mutation inventory and admits a kill only when the clean full 33-test baseline succeeds and a single designated test body fails at its exact designated assertion line with one AssertionError, zero errors, and zero skips. The harness propagates the parent's optimized mode to child tests and also has a mutation against its own attribution predicate.

## Limits

This scanner does not establish Python runtime behavior, shell/YAML semantics in general, JavaScript parsing correctness, completeness of every possible test/execution mechanism, or semantic absence of placeholders introduced by metaprogramming. The comment lexer for JavaScript/shell is intentionally conservative. Findings and misses require source-aware review. A green component workflow is not G11 gate acceptance, Layer A compliance, proof closure, scientific acceptance, or CBZ-0.
