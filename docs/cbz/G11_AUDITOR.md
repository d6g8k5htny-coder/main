# G11 auditor — bounded census, not certification

## Purpose

`tools/cbz/g11_census.py` reads tracked blob bytes from one Git revision and reports review triggers for the owner-defined G11 tooling-hygiene criterion: Python stubs, code-comment markers, no-op workflow run steps, and tool documentation/static-reference checks. It is a triage instrument, not a proof, a Layer A certificate, or a federation-wide G11 admission.

## Commands

From the repository root, using only the Python standard library:

```sh
python3 -B -S -m unittest tests.test_g11_census
python3 -B -S -O -m unittest tests.test_g11_census
python3 -B -S tools/cbz/g11_mutation_selftest.py
python3 -B -S -O tools/cbz/g11_mutation_selftest.py
python3 -B -S tools/cbz/g11_census.py --rev <commit> --json census.json
```

The census exits 0 for no detected production finding, 1 for a detected production finding, and 2 when the requested Git revision cannot be read. In-scope tracked symlinks and other non-regular scanned entries fail closed with exit 2. A zero exit is only the result of this implementation over its stated scanned paths and rules.

## Soundness boundaries

- Stub exemptions are bound to imported `abc.abstractmethod`, `typing.overload`, `typing_extensions.overload`, immediate `typing.Protocol`/`typing_extensions.Protocol` interface methods, and the named `http.server` request-handler `log_message` override. ABC inheritance alone and exception-class inheritance do not exempt a method. Nested outer-class context, name suffixes, local look-alikes, and lexical bindings shadowing an imported decorator/base do not grant exemptions.
- Python reference evidence is limited to module-level statements and `unittest.TestCase` methods, following only locally reachable helper calls and literal branches; uncalled helpers, `if False`, runtime-dependent branches, and loop bodies are not credited. Workflow references ignore here-document bodies and are read from run commands, not YAML comments. Python/shell `-c` payloads and JavaScript template-literal text are not credited. These are static references only, never proof of runtime execution, and cannot satisfy a runtime-exercise requirement.
- YAML comments are scanned for marker words. Empty and comment-only `run` blocks are reported as no-op steps.
- The mutation self-test preserves the original mutation inventory and admits a kill only when the clean full 38-test baseline succeeds and the exact designated module/class/method reports `FAILED` with one AssertionError at its exact designated assertion line, zero errors, and zero skips. The harness propagates the parent's optimized mode to child tests and mutates its own parser and each attribution guard.

## Limits

This scanner does not establish Python runtime behavior, shell/YAML semantics in general, JavaScript parsing correctness, completeness of every possible test/execution mechanism, or semantic absence of placeholders introduced by metaprogramming. The comment lexer for JavaScript/shell is intentionally conservative. Findings and misses require source-aware review. A green component workflow is not G11 gate acceptance, Layer A compliance, proof closure, scientific acceptance, or CBZ-0.
