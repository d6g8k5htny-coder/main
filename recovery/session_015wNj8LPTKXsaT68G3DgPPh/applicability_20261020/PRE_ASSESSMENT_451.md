# main#451: pre-assessment of whether C8's R1 applies after a refresh

Written by Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8), the R1 reviewer of record (PASS_ENGINEERING_SCOPED at `8f1d846daeb03990a84a92fa6e42c9903575c412`, [451/6279044289](https://github.com/d6g8k5htny-coder/main/pull/451#issuecomment-6279044289)). Read-only. Credit 0. Scientific effect NONE.

It was assessed against current `main` `b78134c75e733febc6bd2becac4459036ef8c0ad` at about 00:20Z on 20 October, before any refresh existed. It is not a verdict on any new head. After a refresh, it must be re-validated against the actual head and base. If `main` has moved again, the base delta must be assessed again.

## 1. Byte identity (mechanical; `delta_readback.sh`)

A simulated refresh, a local commit with parents `b6f9c104` and `b78134c7`, gives:
- tree `f2a9cdf0`, equal to a fresh `git merge-tree --write-tree`;
- 192 delta paths, +12972/−0, all under `reviews/zeta_fold_slice_c_rerun_20261019/`;
- a raw diff identical to R1's reviewed diff, `0a7e5145..8f1d846d`: the same paths, modes and before and after blobs;
- no delta path changed on `main` since the reviewed base.

## 2. Review applicability (C8's judgment)

**Base changes.** `0a7e5145..b78134c7` changes:
- `.github/workflows/semantic-memory-experiment.yml` and `tests/test_semantic_memory_benchmark.py` (#448);
- `AGENTS.md` (#449);
- three new files under `incoming/h3-closure-restoration-20261018/` (#446).

None of these is under the record directory or under `experiments/zeta_fold_exploration/`.

**Dependencies of R1.** Each object is the same at `0a7e5145`, `efc2fe90`, `b78134c7` and `b6f9c104`:

| Path | Object |
|---|---|
| the consumed packet `experiments/zeta_fold_exploration` | tree `bca06db8` |
| `tools/navigation_check.py` | `f2dec4c5` |
| `tools/workspace_landing_check.py` | `07ccd323` |
| `.github/workflows/navigation.yml` | `c4e15729` |
| `.github/workflows/workspace-landing.yml` | `e37310e9` |
| `.github/workflows/public-intake.yml` | `d1b421ea` |

**Interactions with R1's findings:**
- **(a) Manifest, PASS:** stands. Its hashes cover the record files and the unchanged packet.
- **(b) Archives, NOT CHECKED:** unchanged. It stays not checked.
- **(c) Replay, PASS:** stands, because the packet is unchanged.
- **(d) 50/14/2, PASS:** stands, because the record bytes are unchanged.
- **Notes N1–N3:** unchanged.
- **Base-change workflows:** neither the semantic-memory workflow change nor the `AGENTS.md` change touches the record or the packet, and no changed workflow runs over the record directory.

**Conclusion (provisional):** if byte identity holds at the actual H′ and `main` is still `b78134c7`, R1 applies to the composition. R2 (the full different-provider replay) and the README's §9 (NOT_PERFORMED) are unchanged. Fresh hosted checks are still required.

## 3. Hosted checks (reported separately after a refresh)

At H′: `verify`, `public-intake` and `public-shop` (strict), plus the PR's other checks. Each must be reported with its own event, workflow and checked commit.

## Ownership and restriction

C8 integrates nothing. Before any read after a refresh, C8 confirms current ownership, takes a bounded pickup pinned to the exact head and base only if no one else owns that assessment, and otherwise coordinates.

A release by 014K or Claude017 ends that worker's reservation. It does not clear a permission denial or authorize another session to perform the denied action (Dylan, 20 October about 00:14Z, in C8's session).
