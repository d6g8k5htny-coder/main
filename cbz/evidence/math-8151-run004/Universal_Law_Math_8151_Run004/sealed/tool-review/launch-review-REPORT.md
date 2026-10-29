# Sequential Math adapter — launch follow-up engineering review

Reviewer: separate OpenAI/Codex execution with inherited source exposure. Organizational independence credit 0; no scientific review or theorem-status promotion.

Exact reviewed driver SHA-256: `df54d63dabfb04becf256f77f4feb843e3d866837a17d92f8e47d1c547335712`.

**No new blocking finding identified in the submitted launch change.** The earlier review at `0a81d59c...` remains a historical, different source cut.

## Examined behavior

`lean_launch` rewrites only the exact `['lake','env','lean',...]` prefix. It uses the same pinned absolute Lean executable and preserves the remaining argument order. Builds, `leanchecker` and other commands retain their original launch behavior.

The environment-capture subprocess is itself executed through the bounded process runner. Its Python program requests only ten explicitly named toolchain/search-path fields; it does not enumerate or serialize the inherited environment. Duplicate JSON keys, non-object/missing/unknown fields, non-string/non-null values, unsuccessful captures, a mismatched pinned Lake/sysroot, and a missing Lean search path are refused before caching. Captured nulls remove the corresponding inherited field; other inherited guard variables are retained. The cache key is the resolved working directory, so the original package and compiled-admission copies have separate captures. Actual requested/executed arguments and those selected fields are recorded.

Source inspection found the prior exact-module replay, original gate predicates and positive/negative outcomes retained. The monitor-error cleanup and refused-inner-receipt fixes remain present.

## Independently observed checks

- Existing fixture suite: **21/21 normal and 21/21 optimized**, no failures or skips.
- Additional inert launch probes: **14 cases per mode**, including malformed/empty JSON, missing/unknown/duplicate fields, wrong Lake/sysroot, absent Lean path, invalid value types, argument/environment preservation, canonical per-directory caching and unchanged non-Lean commands.
- The original independent monitor-error and outer-seal probes passed again: no live owned direct child after the planted monitor exception; no active inner success receipt after a refused outer seal, with the refused receipt preserved.
- An actual disabled field-set source mutation was detected by the independent inert probe in both Python modes through an explicit assertion failure. Probe checks remain active under optimization. Initial probe-control metadata expected a later unknown-field label while the first missing-field assertion correctly fired; that harness-label mismatch is preserved separately and was not counted as a pass.

All probes use fixed inert fixture values and disposable stdlib processes. No heavy Lake/Lean invocation, original Math/Main edit, real gate execution, Git/ref change or external write was performed by this reviewer. Exact driver identity was checked before and after the review.

## Scope limits

This is a finite engineering review. It does not establish mathematical correctness, statement alignment, nonvacuity, complete Math gate success, source-acquisition authenticity or CBZ-0. Actual mathematical outcomes require the completed source-bound outer execution receipt. The launch cache assumes the selected environment fields stay stable for a given working directory within the pinned run; this is not a general replacement for arbitrary Lake versions or changing package configurations. Existing source/dependency/runtime checks remain necessary. Resource monitoring is sampled process-group monitoring, not a new kernel-enforced per-job sandbox.
