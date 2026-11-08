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
  condition; unknown conditions establish neither arm and apply possible
  binding, aggregate, import and wrapper effects from both arms before later
  attribution.
- Async helper bodies are not credited from coroutine creation alone. General
  await/coroutine activation analysis is outside this bounded scanner.
- Async forwarding wrappers and generator-function bodies are not credited
  from coroutine/generator object creation alone.
- Comprehension bodies require statically established iteration; generator
  expression bodies require consumption. Chained comparisons stop after a
  proven false prefix.
- JavaScript `const`/`let` URL bindings are available only within the lexical
  brace range containing their declaration; `var` retains function semantics.
- Forwarding-wrapper discovery applies the same proven literal
  conditional-expression reachability rule as direct execution traversal.
- Wrapper and helper reassignments declared `global` or `nonlocal` invalidate
  the destination binding at actual execution order; `nonlocal` resolves only
  to the nearest enclosing function scope.
- Lambda construction visits positional and keyword defaults, which execute at
  construction, but does not visit the deferred lambda body.
- Wrapper forwarding is decided only after callee, positional and keyword
  argument effects have executed; a mutation in a later argument can therefore
  invalidate an earlier forwarded aggregate.
- Wrapper discovery applies chained-comparison short-circuiting. Unproven
  exception handlers and `else` bodies are not execution evidence; `try` and
  `finally` bodies remain reachable.
- Possible writes in unproven exception handlers and `else` bodies invalidate
  downstream static bindings without crediting those branches. Wrapper
  discovery uses the same boundary and credits only the proven `try` and
  `finally` paths.
- Empty eager comprehensions and unconsumed generator expressions cannot prove
  forwarding. A statically nonempty one-generator eager comprehension with
  literal-true guards is the bounded positive case.
- A definitely terminating `finally` terminates the enclosing statement
  sequence. Python `nonlocal` static writes target the nearest real enclosing
  binding, and a destructured JavaScript lexical declaration shadows an outer
  URL binding at the child-process call.
- Unknown Python branches invalidate possible writes without crediting either
  branch, and a handler-free `try` whose body terminates also terminates the
  enclosing sequence. Decorated helpers and non-string executable positions
  are left unresolved rather than inheriting stale execution provenance.
- JavaScript helper and URL attribution requires the actual unshadowed helper
  identity, a trusted `node:url` `fileURLToPath` import when conversion is used,
  and a URL binding delivered directly or as an immediate array element to the
  child-process call. Nested named helper chains remain reachable when their
  own declaration is the resolved identity.
- Unproven exception effects invalidate active helper and forwarding-wrapper
  identities as well as static values; exception target names are effects too.
  Assignment expressions evaluate their value and then update the destination
  binding. Nested aggregate mutation recursively invalidates reachable list
  aliases, while replacement of one outer element does not rewrite a separately
  bound child.
- Eager comprehension credit requires a statically nonempty supported literal
  iterable and a simple target name. Python interpreter parsing admits only the
  documented script-continuing flags and operand options; attached `-c`/`-m`,
  exit-only, and unknown pre-script options fail closed. Decorated forwarding
  wrappers are not registered because arbitrary decorator results are unknown.
- JavaScript URL bindings count only in child-process command/file position 0
  or as immediate identifiers in the literal argv array at position 1. Options,
  callbacks, later arguments, dynamic arrays and composite expressions do not
  receive URL credit.
- Helper and forwarding-wrapper calls must satisfy their supported signatures
  before their bodies or forwarding positions become reachable. Helper frames
  bind positional, keyword, default, `*args` and `**kwargs` aggregates by
  object identity; tuple `+=` remains a rebind while list `+=` mutates shared
  identity.
- Annotation-only assignments do not invalidate an existing import identity;
  value-bearing annotations and assignment expressions do. Nested helpers
  resolve owned enclosing bindings before module fallback.
- The bounded `unittest` model executes `setUpClass`, then `setUp`, the test,
  and `tearDown` for each lexically discovered test method, followed by
  `tearDownClass`. Inheritance, module fixtures, cleanup stacks, skips, custom
  suites and asynchronous lifecycle behavior remain outside the supported
  domain and receive no general runtime claim.
- JavaScript reachability excludes unbraced literal-false statements and keeps
  concise arrow bodies deferred until a reachable call or trusted `node:test`
  registration. URL declarations must begin in executable code, not quoted
  text. Mutable URL objects lose path provenance when an alias escapes or may
  mutate them; primitive `fileURLToPath` string aliases retain only their
  supported immutable value.
- Assignment targets visit executable subexpressions after their right-hand
  side and before rebinding. An unresolved chained-comparison suffix or eager
  comprehension body/guard contributes possible-effect invalidation without
  receiving execution credit; a proven-false comparison prefix and a proven
  empty/noniterable comprehension remain excluded.
- Chained assignment evaluates its right-hand side once, then interleaves each
  target's executable subexpressions and store in language order. Lambda
  defaults under unresolved branches contribute possible effects while lambda
  bodies remain deferred.
- JavaScript class/object method bodies, including computed-name methods,
  remain deferred until supported invocation is established, and literal `if`
  conditions exclude either dead branch, including dangling-`else` binding.
  Dynamic property-method invocation and general JavaScript parsing remain
  unsupported and fail closed.
- Optional JavaScript named-property calls and computed accesses receive no
  subexpression credit until their optional chain is proven to evaluate.
  Unconstructed instance-field initializers remain deferred. Regular-expression
  literals following `if`, `for`, `while`, or `with` headers are masked before
  child-process call discovery.
- Class definitions under unresolved Python branches retain decorator, base,
  keyword and body construction effects while method bodies stay deferred.
  Helper definition-time parameter effects are mapped to bound caller
  arguments. JavaScript optional-chain suppression continues through repeated
  named-property hops, and computed-name instance-field initializers remain
  deferred until construction.
- Optional-chain suppression continues through computed accesses and
  intermediate calls until the complete continuous chain ends. Semicolonless
  newlines are recognized as instance-field boundaries, so subsequent field
  initializers remain deferred too.
- Helper-effect summaries retain class-definition-time mutations and map them
  through bound caller arguments while method bodies remain deferred. Calls
  after an unsupported JavaScript `await` receive no activation credit, and
  nested computed method names are delimiter-balanced before their bodies are
  classified as deferred.
- Destructuring stores execute leaf-by-leaf from right-hand-side leaf values
  snapshotted before any target effect, while nested mutable values retain
  identity aliasing. Quoted class/object method bodies remain deferred. Nested
  computed instance-field names are delimiter-balanced before their initializer
  is bounded, and assignments inside class static blocks remain executable at
  class definition.
- Nested Python destructuring snapshots each unpacking level only when that
  nested target is reached. Quoted and numeric JavaScript instance-field
  initializers remain deferred until instance construction. Direct unconditional
  termination inside an executable class static block suppresses its unreachable
  tail without suppressing the block's reachable definition-time effects.
- Numeric instance-field names include supported exponent, leading-decimal,
  separator and radix forms. A direct static-block `throw` evaluates its bounded
  operand before the remaining block tail becomes unreachable.
- Omitted mutable-default helper parameters retain their captured aggregate for
  possible mutation propagation. Parameter aliases assigned inside nested
  helper statements map mutations back to the originating formal. Numeric-name
  class/object methods remain deferred, and `static = ...` is classified as an
  instance field rather than the class static modifier.
- A mutable default retains the identity captured at function definition even
  if the name in its default expression is rebound before invocation. Helper
  aliases assigned under a literal branch retain only the taken formal origin;
  aliases under an unresolved branch conservatively retain all possible formal
  origins.
- Class-body discovery skips nested braced class/function expressions inside a
  heritage expression before selecting the outer body. Ordinary and annotated
  assignment right-hand sides are visited before formal-alias rebinding, so
  helper mutation at that evaluation point remains attributed.
- Helper-return expressions and container/subscript aliases conservatively
  retain every formal aggregate origin used to construct them. A possible
  mutation through an attribute of a formal aggregate is outside the bounded
  object model and therefore fail-closes by invalidating all tracked aggregate
  provenance; filename or exception text is never used as a substitute.
- Multiline computed method/field names use balanced delimiter discovery.
  Malformed or unbalanced class structures fail closed rather than granting
  module execution. Valid private names such as `#class` and `#function` and
  reserved words inside import/export specifiers are not expression-keyword
  evidence; paired live calls preserve positive attribution.
- YAML comments are scanned for marker words. Empty and comment-only `run` blocks are reported as no-op steps.
- The mutation self-test has 328 cases. Earlier 319–327-case configurations
  remain historical evidence for their exact trees and do not establish this
  expanded contract. In addition to the earlier adverse controls, three
  designated cases remove helper-return alias propagation, remove
  container/subscript alias propagation, suppress detection of unknown nested
  aggregate mutation, and suppress its direct-branch propagation. It seals the
  exact source slice containing
  `MUTATIONS`, `TARGETS`, `NEW_MUTATIONS` and `EXPECTED_MUTATION_NAMES` with
  test-side SHA-256
  `97820c30570ab5dd2d72842d94495e1b0d695bd1307f43124a25db7535d09691`,
  then compares complete literal case tuples—not names alone—against the
  harness source. A kill is admitted only when the clean full 263-test baseline
  succeeds and the exact designated module/class/method reports `FAILED` with
  one `AssertionError` at its exact designated assertion line, zero errors and
  zero skips. The harness propagates the parent's optimized mode to child tests
  and applies each mutation to its declared source, including all
  harness-target cases. Assignment-order mutants preserve right-hand-side
  traversal and change only ordering.

Exact hosted source `8038863c04cc96acfcfc0d2953ab8575ccce213e` (tree
`de8e2c664116803643cfda2f324756e7716c57af`) was qualified locally at exact-tree
commit `1616b31889a8b242ff6950dbd7af4d437e54648c`: 256/256 controls passed and
321/321 mutations were killed in normal and optimized modes with zero skips.
The mutation receipts are byte-identical, 199,677 bytes, SHA-256
`367a13fed18c0880fcc73590ffb1f615bb48ba1baf76ecc621f427637f31f687`.
The byte-identical censuses are 56,059 bytes, SHA-256
`bcd9605e448a4e3d121f8e9a03733190f376367e5c2ab690b17a90675605ef8f`;
they contain zero production findings, retain 32 tool candidates, exit 1, and
therefore preserve G11 FAIL. The preceding native source
`3f3dc9166a79b7d0bb7ec97b6422621a7da90f85` failed both censuses with
`RecursionError`; that actual failure is retained and not rebound.

## Limits

This scanner does not establish Python runtime behavior, shell/YAML semantics in general, JavaScript parsing correctness, completeness of every possible test/execution mechanism, or semantic absence of placeholders introduced by metaprogramming. The comment lexer for JavaScript/shell is intentionally conservative. Findings and misses require source-aware review. A green component workflow is not G11 gate acceptance, Layer A compliance, proof closure, scientific acceptance, or CBZ-0.
