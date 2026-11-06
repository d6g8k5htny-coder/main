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

- Stub exemptions are bound to source-resolved imports, not decorator or base names. An imported `abc.abstractmethod` exempts only a method in an immediate class with a trusted imported `abc.ABC` base. Imported `typing.overload`/`typing_extensions.overload` exempts only declarations in one contiguous same-name overload group followed by a concrete, non-stub implementation. Immediate imported `typing.Protocol`/`typing_extensions.Protocol` interface methods and the named `http.server` request-handler `log_message` override remain contextual. ABC inheritance alone and exception-class inheritance do not exempt a method. Nested outer-class context, unpaired overloads, name suffixes, local look-alikes, and lexical bindings shadowing an imported decorator/base do not grant exemptions.
- Python reference evidence is limited to module-level statements and classes whose base resolves to the imported `unittest.TestCase` identity (including explicit aliases), following test methods, their `setUp`/`tearDown`/class lifecycle methods, locally reachable helper calls, literal branches and the two sides of a simple `name is [not] None` cache guard; a local class merely named `TestCase` is not credited. The scanner resolves explicit local path and command bindings built from `__file__`, `pathlib.Path`, selected `os.path` functions and environment fallbacks, and credits them only at a trusted `subprocess`, `os.system`, `runpy.run_path` or `importlib.util.spec_from_file_location` call, or at a reachable wrapper proven to forward its argument to one of those trusted APIs. Reachable rebinding invalidates wrapper forwarding, unresolved local names do not inherit same-named module path values, and augmented assignment replaces or clears a stale static value. Forwarded process commands receive the same interpreter command-mode filtering as direct process calls. Unknown sequence elements retain position: an unknown prefix refuses attribution, while a dynamic suffix after the statically selected tool cannot shift that already selected executable. Resolved starred lists are expanded; unresolved starred prefixes refuse. Boolean `or` is left-to-right and refuses an unknown operand before a proven fallback. JavaScript `new URL(..., import.meta.url)` bindings are credited only when an identifier expression is passed to the directly called named import from `node:child_process`; quoted text, property names, quoted call text and property calls are excluded. Uncalled helpers, `if False`, other runtime-dependent branches, loop bodies, fake same-named APIs and mere path mentions are not credited. Workflow references ignore here-document bodies and are read from run commands, not YAML comments. Python/shell `-c` payloads and JavaScript template-literal text are not credited. These are static references only, never proof of runtime execution, and cannot satisfy a runtime-exercise requirement.
- Class methods remain scoped to their class and cannot register same-named
  bare module wrappers. Statically known conditional commands select only the
  taken branch; unknown alternatives receive attribution only when every branch
  selects the same tool. A JavaScript URL binding is credited only if no
  intervening assignment precedes the selected child-process call. Every
  component passed to `pathlib.Path` participates in resolution; an assignment
  through a command element or slice invalidates the root aggregate; and a
  function parameter shadowing a named child-process import blocks attribution
  within that function.
- YAML comments are scanned for marker words. Empty and comment-only `run` blocks are reported as no-op steps.
- The mutation self-test has 119 cases. It includes adverse controls for bounded absolute-parent traversal and multi-component path construction; direct and forwarded command-mode filtering; aggregate invalidation after command element assignment; unreachable and rebound wrappers; function-local imports; destructuring; local shadows; JavaScript strings, properties, callee identity, parameter shadowing and post-binding reassignment; augmented assignment; unresolved and resolved starred command positions; Boolean short-circuit order; lexical callee and class-method shadowing; conditional-command alternatives; quoted JavaScript call text; proven command prefixes with dynamic suffixes; `unittest` lifecycle reachability; and trusted loader-wrapper forwarding. Two original cases remain dropped as equivalent mutants: exception bases confer no stub exemption, so neither a decoy `…Error` base name nor an ignored shadow of a builtin exception name can change a report. It seals the exact source slice containing `MUTATIONS`, `TARGETS`, `NEW_MUTATIONS` and `EXPECTED_MUTATION_NAMES` with test-side SHA-256 `21958cb885d59b8cd0b4bcbc3ea6880eb889f5a170e54742da20a885787c5149`, then compares complete literal case tuples—not names alone—against the harness source. A kill is admitted only when the clean full 67-test baseline succeeds and the exact designated module/class/method reports `FAILED` with one AssertionError at its exact designated assertion line, zero errors, and zero skips. The harness propagates the parent's optimized mode to child tests and applies each mutation to its declared source, including all harness-target cases.

## Limits

This scanner does not establish Python runtime behavior, shell/YAML semantics in general, JavaScript parsing correctness, completeness of every possible test/execution mechanism, or semantic absence of placeholders introduced by metaprogramming. The comment lexer for JavaScript/shell is intentionally conservative. Findings and misses require source-aware review. A green component workflow is not G11 gate acceptance, Layer A compliance, proof closure, scientific acceptance, or CBZ-0.
