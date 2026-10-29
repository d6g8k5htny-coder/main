# Sequential Math adapter — bounded engineering review

Observed 2026-10-29. Reviewer: separate OpenAI/Codex execution, inherited source exposure; no organizational independence credit and no scientific review.

Reviewed driver SHA-256: `0a81d59c359948adf80032eafe0c838139dd979d5ddc64491db7d2fcbc46474c`.
Exact replay helper SHA-256: `e61370531ad8e12837ca9ae9ec1f615208f9ec0b225967c9622be0ab81ddd43d`.
Supplied pinned upstream source SHA-256: `2d18ec111d73c536b0bdee2533556bb9548c349f386797d3d3837b1f84d55dae`.

## Findings

The two previous engineering findings are **ADDRESSED** at the reviewed driver identity:

- A planted process-monitor exception now terminates the task-owned process before propagating the exception. The independent original probe observed no live direct child afterward. TERM/SIGKILL cleanup is retained for descendants in the task-owned process group; unrelated groups are not selected.
- When an outer source seal is refused after the underlying gate wrote success evidence, the live inner receipt is removed from the active location and preserved as `refused-inner-receipt.json`. The independent original probe observed the refused outer status, no active inner success receipt, and a preserved receipt.

No additional blocking finding was identified in this finite source/control review.

## Observed checks

The existing independent probe was replayed with `--expect-fixed` into `final-review-probes.json` and exited 0. The unchanged engineering fixture suite was independently executed in normal and optimized Python modes: **14/14 per mode, no skips or failures**. Source SHA was checked before and after.

Source inspection confirmed that the generated Lean helper imports `LeanChecker`, initializes the search path, and calls the supplied pinned `replayFromImports` API for exactly the validated module name. The supplied upstream API loads exported/server/private object parts and invokes kernel replay into an import-only environment. This is ordinary named-module replay, not `--fresh` replay of the full dependency closure.

The adapter's replacement is forwarded by `gate.py` into the original helper. It retains sequential builds of every registered module, ordinary default-build confirmation, all registered module replays, original expected-positive/negative phase outcomes, compiled admission controls, axiom/inventory checks, source/dependency observations, and before/after adapter/runtime identities. No Math source or original gate predicate was edited by this reviewer.

## Limits

This review did not execute a heavy Math gate, establish a theorem, assess statement alignment/nonvacuity, repeat official-release custody acquisition, or establish federation CBZ-0. The supplied upstream source was inspected; an installed source-tree pathname was absent, so no installed-source equality is claimed here. Resource guards sample task process-group RSS and elapsed time; they are engineering diagnostics rather than a mathematical bound or a security sandbox against workloads escaping their process group. The final Math outcome must come from an actual completed source-bound execution and its outer receipt; these fixture successes cannot substitute for it.
