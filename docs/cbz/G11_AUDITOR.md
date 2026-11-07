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
  within that function. Interpreter `-c`/`-m` mode switches are considered only
  before the statically selected script; later tokens are program arguments.
  Element or slice writes invalidate a wrapper parameter's forwarding proof,
  and helper-local invalidation leaves an explicit unresolved binding rather
  than falling back to a module aggregate. JavaScript parameter analysis reads
  binding positions, so a default initializer's imported callee remains a
  reference rather than being misclassified as a bound parameter.
- Recognized Python interpreter options (`-W`, `-X` and
  `--check-hash-based-pycs`) and the declared shell option subset consume their
  operands before script selection. An element or slice write invalidates all
  statically resolved identity aliases of its command list. A JavaScript
  function parameter also shadows a same-named outer URL binding within that
  function. Supported command-list mutator calls (`append`, `clear`, `extend`,
  `insert`, `pop`, `remove`, `reverse`, `sort`, `__delitem__`, `__iadd__`,
  `__imul__`, and `__setitem__`) invalidate the root binding and every resolved
  identity alias. A declaration of the named child-process callee within a
  function shadows that import across the function, and a `new URL` binding
  declared in a function is visible only to calls in that same function. These
  are deliberately bounded parser rules, not general command or JavaScript
  semantic execution.
- A write or supported mutator call through an alias of a wrapper parameter
  invalidates that parameter's shared forwarding provenance. Reassigning a
  registered wrapper binding invalidates later calls through that name.
  JavaScript declarations in any enclosing function shadow a same-named
  imported child-process callee in nested functions.
- Wrapper discovery stops after an unconditional `return` or `raise`, so an
  unreachable later process call cannot establish forwarding. A reachable
  call inside a deferred function carries the function's actual activation
  point when the scanner checks whether a module wrapper was defined or
  reassigned; lexical line order inside that function is not mistaken for
  module execution order.
- A later same-named Python function definition replaces the earlier wrapper
  registration even when the replacement is non-forwarding. List `+=`
  preserves the statically resolved list object's identity, so mutations
  through pre-existing aliases remain visible. A JavaScript child-process call
  in a literal-false branch or an uncalled named helper is not reachable call
  evidence. Anonymous or otherwise unproven JavaScript function contexts fail
  closed.
- Deferred Python function bodies are ordered at their actual activation
  point, so a later module assignment cannot rewrite the value seen by an
  earlier invocation. JavaScript helper invocations are followed recursively
  to a module-executed root; an uncalled helper chain is refused. A block
  callback is a root only when passed to the locally imported `node:test`
  identity, the registration call is itself reachable, and that identity is
  not shadowed. An arbitrary same-shaped callback API remains untrusted.
- At a supported Python `is None` branch join, bindings written by either
  mutually exclusive branch are invalidated instead of sequentially combined.
  Each alternative first receives an isolated fail-closed binding environment,
  so calls inside one branch cannot inherit assignments from the other.
- Reachable Python helper bodies are expanded at the call position. Their
  binding effects precede analysis of the caller's following statements.
- Declared Python `global` writes update the corresponding module binding.
  Element or slice deletion invalidates the command aggregate and all resolved
  identity aliases. A rebound JavaScript helper cannot supply reachability.
- Deletion through a forwarding wrapper invalidates its forwarded command
  proof. Declared globals are excluded from helper-local static shadows, and a
  JavaScript helper call resolves the nearest lexical declaration before it can
  inherit module-helper reachability.
- Python helper effects in call arguments precede attribution of the enclosing
  call. Reassigned helpers cannot retain their former bodies. Deferred
  JavaScript helpers validate module URL bindings at reachable activation
  points.
- Each Python call argument retains the static value observed at its evaluation
  point, while later in-place mutation invalidates the shared aggregate.
  Assignment RHS helpers execute before the assignment target is rebound.
- List and tuple elements retain their static values in left-to-right
  evaluation order before later element effects can rebind source names.
- A declaration in a containing module-level JavaScript lexical block shadows
  a same-named imported child-process callee.
- Captured sequence values propagate through supported list concatenation.
- Ordinary helper traversal stops after unconditional `return`/`raise`,
  including termination selected by a literal branch.
- Destructured declarations in a containing JavaScript lexical block shadow a
  same-named imported child-process callee.
- Named operands in supported list concatenations retain their evaluation-time
  values before later operand effects can rebind the source name.
- Selected Python branches stop traversal after definite termination inside
  the branch.
- Semicolonless JavaScript lexical declarations shadow imported child-process
  callees under ASI.
- Repeated sequential calls to a Python helper are each expanded; an active-call
  identity guard bounds direct recursion without suppressing later calls.
- A helper's declared `global` reassignment invalidates the module-level helper
  identity used by later calls.
- Literal Boolean `and`/`or` short-circuiting excludes operands that cannot
  execute; an unknown predecessor remains fail-closed.
- Literal conditional expressions traverse only the arm selected by a proven
  condition; unknown conditions establish neither arm.
- Async helper bodies are not credited from coroutine creation alone. General
  await/coroutine activation analysis is outside this bounded scanner.
- Async forwarding wrappers and generator-function bodies are not credited
  from coroutine/generator object creation alone.
- Comprehension bodies require statically established iteration; generator
  expression bodies require consumption. Chained comparisons stop after a
  proven false prefix.
- JavaScript `const`/`let` URL bindings are available only within the lexical
  brace range containing their declaration; `var` retains function semantics.
- YAML comments are scanned for marker words. Empty and comment-only `run` blocks are reported as no-op steps.
- The mutation self-test has 174 cases. It includes adverse controls for bounded absolute-parent traversal and multi-component path construction; direct and forwarded command-mode filtering, including option operands and switch order relative to the selected script; aggregate and identity-alias invalidation after direct, helper-local and wrapper-parameter element writes, element deletion or supported list-mutator calls; wrapper-parameter aliases, reassigned or redefined wrapper bindings, unreachable calls after wrapper and ordinary-helper termination, repeated sequential helper calls with an active recursion guard, and deferred-caller activation/value order; call-position, argument-value, argument-evaluation, sequence-element and supported-concatenation ordering; assignment RHS ordering; global writes, global helper-identity invalidation and active identities of nested helpers; declared-global local-shadow exclusion; unreachable, rebound and lexically shadowed wrappers and JavaScript helpers; deferred JavaScript binding activation; function-local imports; destructuring; local shadows; JavaScript strings, properties, callee identity, URL-binding and binding-position parameter shadowing, simple and destructured local/enclosing/module-block declarations including ASI; local URL scope, default initializers and post-binding reassignment; augmented assignment with list identity; unresolved and resolved starred command positions; Boolean, conditional-expression, chained-comparison and comprehension reachability; synchronous-only helper/wrapper execution with unawaited coroutine and uniterated generator bodies left uncredited; lexical callee, class-method and URL-binding scope; conditional-command alternatives and isolated `is None` branch environments/joins; quoted and unreachable JavaScript call text; recursive helper reachability and imported `node:test` callback roots with reachable unshadowed registration; proven command prefixes with dynamic suffixes; `unittest` lifecycle reachability; and trusted loader-wrapper forwarding. Two original cases remain dropped as equivalent mutants: exception bases confer no stub exemption, so neither a decoy `…Error` base name nor an ignored shadow of a builtin exception name can change a report. It seals the exact source slice containing `MUTATIONS`, `TARGETS`, `NEW_MUTATIONS` and `EXPECTED_MUTATION_NAMES` with test-side SHA-256 `2286112d27097569d494039eebf771e8a92a9cb995cc7fa1c2f723a2f41b9063`, then compares complete literal case tuples—not names alone—against the harness source. A kill is admitted only when the clean full 123-test baseline succeeds and the exact designated module/class/method reports `FAILED` with one AssertionError at its exact designated assertion line, zero errors, and zero skips. The harness propagates the parent's optimized mode to child tests and applies each mutation to its declared source, including all harness-target cases.

## Limits

This scanner does not establish Python runtime behavior, shell/YAML semantics in general, JavaScript parsing correctness, completeness of every possible test/execution mechanism, or semantic absence of placeholders introduced by metaprogramming. The comment lexer for JavaScript/shell is intentionally conservative. Findings and misses require source-aware review. A green component workflow is not G11 gate acceptance, Layer A compliance, proof closure, scientific acceptance, or CBZ-0.
