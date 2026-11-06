This isolated layer-a patch adds a conservative synchronous unittest profile to the published lifecycle grader. The exact baselines are `mutation_test.py` SHA256 e857fde3c3d7caf301e24a77972ee18d1253186613633e1b0876b61f274eebed and `tests/test_mutation_runner.py` SHA256 347135c69335eaa74a62f5b6380057fff24e30eafcf57b8418de69bef1be7f7d. No actual tool, checker, Lean, behavioral suite, or mutation-definition file was edited.

The preserved `red-mixed-generator-false-kill.json` is an actual mixed unittest population run through the published grader: one generator test method never entered its deferred body, while another ordinary method raised AssertionError. The grader nevertheless returned KILLED and recorded both method bodies as entered/finished. The lazy execution sentinel remained false. Its full fixture, grader hash, structured result, and traceback log are retained.

The final policy is deliberately narrow:

- Selected methods must be ordinary synchronous Python functions or bound methods. Population discovery rejects generator, coroutine, and async-generator function shapes. `inspect.unwrap` validates the decorator chain, and every linked layer is checked, including a lazy middle wrapper whose final unwrap target is ordinary.
- An invalid population is refused before fixtures or methods execute. The existing pristine external population binding remains in force.
- A selected synchronous body must return exactly None. Every other return is ERROR, including False, zero, an empty string, a scalar, a generator, a coroutine, an async generator, a custom awaitable, or an otherwise unknown iterable. The grader does not consume or await the object.
- A fresh native coroutine may be closed only in CORO_CREATED state to avoid an unawaited-coroutine warning. The grader never advances it, awaits an async-generator close, invokes a custom close method, or iterates an unknown object.

The implementation uses the existing invalid_population fail-closed channel; it does not weaken assertion provenance or add an interpretation that old receipts had proved this new profile. The two existing test classes (eleven old harness methods, including eight lifecycle methods) and all 57 MUTATIONS entries remain byte-for-byte unchanged. The test module's direct-entry guard remains after the new class.

Four new methods were added:

1. test_mutation_runner.LazyBodyHarnessRules.test_sync_bodies_require_none_and_refuse_other_returns_without_iteration
2. test_mutation_runner.LazyBodyHarnessRules.test_mixed_lazy_function_shapes_are_refused_before_any_body
3. test_mutation_runner.LazyBodyHarnessRules.test_wraps_decorated_lazy_functions_are_refused_at_population
4. test_mutation_runner.LazyBodyHarnessRules.test_sync_lazy_returns_are_rejected_without_advancing_objects

Only these four new methods ran against the final profile, once normally and once under PYTHONOPTIMIZE=1 with `python -B -S -O`; fixture subprocesses inherited optimization. Each mode produced eighteen saved fixture receipts: ordinary None-return pass/kill controls, four scalar/falsey returns, an unknown iterable, three declared lazy method shapes, four decorator-chain shapes, and four synchronous lazy return shapes. All eighteen lazy-execution sentinels remained false in each mode. For the seven refused function/decorator populations, the ordinary-method sentinel also remained false, proving preflight refusal before any body. For runtime non-None returns, the companion genuine assertion body executed; the whole population still graded ERROR.

The requested final None-only profile received its own RED: before the final return check, zero, False, an empty string, a scalar, and an unknown iterable were falsely accepted in mixed assertion populations. `red-none-profile.log` records five assertion failures. The final normal and optimized logs are GREEN.

Three actual final-profile source mutants were cleanly assertion-killed in each mode: disabling the non-None return rejection; replacing identity-to-None checking with truthiness (admitting falsey returns); and iterating an unknown returned object during rejection. These mutate the real isolated grader source and run only the new behavioral control through an unchanged trusted runner. Structured receipts show full outer population execution and assertion failures without runtime-error kills.

The initial implementation rejected known lazy return shapes while allowing other synchronous values. Its completed normal/optimized controls, full source/test snapshot, and five source-mutant receipts per mode are preserved under lazy-types-generation. That generation is explicitly superseded by the final None-only profile. Its pristine function-shape refusal, decorator-chain check, and fresh-native-coroutine disposal controls remain applicable: those code paths and their relevant lazy inputs did not change. Those three completed source-mutant controls per mode are reused. The old type-specific custom-awaitable predicate was removed and is not claimed as a final safeguard. The newly broadened return-profile checks received fresh verification.

Completed verification of the old eight lifecycle methods, original minimal actual checker-source controls, 57 checker mutants, and mathematics was reused rather than rerun. New evidence is limited to the newly identified lazy-body/None-return obligations. This report is not a fresh full G0 pass, a rerun of historical source mutants, or a CBZ declaration. Provider-distinct review and publication remain separate requirements.

Frozen full files are in frozen/, with the minimal two-file patch at layer-a-lazy-body.patch and exact baseline/final hashes and method IDs in frozen-manifest.json. The existing two registered file hashes must be updated upon integration. No production helper module or new registry path is needed. The public command and receipt schema remain unchanged.

Suggested README/SPEC addition: “Mutation grading supports ordinary synchronous unittest methods that return None. Generator, coroutine, async-generator, and wraps-linked lazy method shapes are refused during pristine population discovery. Every non-None body return is ERROR. Returned objects are not iterated or awaited; only a fresh native coroutine may be closed without entering its body. Lazy call-object creation is not evidence that a deferred test body executed.”
