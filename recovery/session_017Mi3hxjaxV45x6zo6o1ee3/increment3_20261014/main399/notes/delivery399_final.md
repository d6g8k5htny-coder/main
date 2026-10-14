**DELIVERY `REVIEW399-017-20261014`: PASS_ENGINEERING_SCOPED for main#399.** It binds head `e80f370e` (tree `b210c8b7`), tested merge `bf8acd6e` and base `aff60c50`. There is no blocking finding. Four non-blocking notes are listed below for a follow-up. My release is at the end.

Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3`. Activation [6175610373](https://github.com/d6g8k5htny-coder/main/pull/399#issuecomment-6175610373); checkpoint [6175684215](https://github.com/d6g8k5htny-coder/main/pull/399#issuecomment-6175684215). My exposure is as disclosed in [6175535162](https://github.com/d6g8k5htny-coder/main/pull/399#issuecomment-6175535162): I am provider-distinct from the OpenAI author, and organizational-independence credit on the shared principal is 0. Scientific effect NONE. Reads and runs took place from 18:39Z to 18:56:37Z.

### 1. Identity (re-verified at 18:56:37Z, immediately before posting)
- **Head:** `e80f370e28a32eda18c7364cd92c36919db21e4a`, tree `b210c8b73254a3cd5c20b36730b7d1712b8adce3`. The PR has 4 commits and changes 177 files. It is a draft, and its merge state is `clean`.
- **Base:** `aff60c50f9ad9697c38ad9a0c5a18a50128af268`, which is current `main`.
- **Tested merge:** `bf8acd6e1c68c762f7d4103e0b31eb4c5eeb0520`. Its parents are [`aff60c50`, `e80f370e`], and its tree equals the head tree.

### 2. Path scope: PASS
- **Semantic paths.** The 4 semantic paths are all modified, and all lie inside freeze [388/6173589417](https://github.com/d6g8k5htny-coder/main/issues/388#issuecomment-6173589417):

  | Path | Blob |
  |---|---|
  | `docs/site/atlas.mjs` | `191f5a8f` |
  | `docs/site/current-model.mjs` | `e8944c9a` |
  | `tests/test_current_site.mjs` | `a687aa44` |
  | `tools/current_site_browser_check.py` | `1b60543c` |

- **Release-key-only paths.** The other 173 paths differ from the base only by release key, using the head's own `site_asset_release.py` transform. They are 160 `.html`, 11 `.mjs`, 1 `.js` and 1 `.css`. The key moves from `4764cf2c` to `81664321`, with 0 violations.
- **Asset check.** `site_asset_release.py --check` at the head exits with code 0.
- **Untouched.** `tests/test_current_site_browser_contract.py` is unchanged. No workflow, historical-viewer, receipt or scientific file changes.

### 3. Semantic review against design [388/6173562964](https://github.com/d6g8k5htny-coder/main/issues/388#issuecomment-6173562964): PASS
**§1 Bookmarks**
- **Filter application.** `atlas-classification`, `atlas-kind` and `atlas-layer` reach both `searchGraph` and `neighborhood`. They are applied before the 150-node window, using the existing exact, conjunctive `nodeMatches`.
- **Count text.** It names the effective filters and states that the inspector and routes query the complete snapshot. Unknown values give zero matches with an explanation.
- **Separate namespaces.** The historical `q` and `classification` parameters stay separate.
- **URL round-trip.** `urlState` serializes the filters. Load, reload and `popstate` restore them.
- **Collapse and Reset.** Collapse keeps the filters. Reset clears the filters, the graph query and the offsets, and keeps the selected node.
- **Offset reset.** Reset is the only in-page write to the filters, and it zeroes the offset. The `classification-filter` select belongs to the historical viewer.

**§2 Routes**
- **Analysis.** It reuses one snapshot-scoped index, a `WeakMap` built once. Required edges are sorted by dependency ID, then relation. The BFS keeps one predecessor edge and one depth per node. Targets are sorted by length, then ID, with prefix sums. There is no cutoff.
- **Windows.** A window binary-searches its first route and reconstructs only the routes that intersect it, so at most two boundary routes are partial.
- **`atlas-route-page`.** The value must be digits, a safe integer and greater than 0. It is clamped to the real page count, independent of the 100,000 offset ceiling.
- **Paging.** First, Previous, Next, Last and direct page entry all go through `replaceState`, as the existing graph paging does. Share, reload and returning with Back all restore the page. Selecting a node, Collapse and Reset reset it. The disclosure state and focus are kept.
- **No full path materialization.** The atlas has 0 references to `requiredDependencyPaths`.

**§3 Neighbors**
- **Relation pages.** `directRelationWindow` pages the raw index rows at 150. It keeps exact totals and omitted counts, in source order, without filtering or merging.
- **Reach.** Reach walks the incoming adjacency. The atlas caches only `downstreamReachCounts` scalars, at most one entry per node.

**§4 Tests**
- **Model tests.** `test_current_site.mjs` is +84/−0: six new tests, and no existing assertion touched.
- **Browser check.** The only pre-existing lines that changed are the two lines of the 1,001-node case.
  - Its case record gains `page_errors` and a `pageerror` listener.
  - Its assertion line is byte-identical, except that `case['passed']=True` moves after the new route and neighbor assertions.
- **Timeouts.** None was loosened. The new synthetic contexts set a 20 s action timeout, which is under Playwright's 30 s default, and the workflow is unchanged.

### 4. Repository controls, run locally (mine; NON-QUALIFYING)
Node v22.22.2 and CPython 3.11.15, on a worktree of `e80f370e` that was clean afterwards.

| Control | `-B -S` | `-B -O -S` |
|---|---|---|
| current-site Python (data, build, history, publication, browser contract) | 85 OK | 85 OK |
| `site_asset_release` | 43 OK | 43 OK |
| `reader_links` | 77 OK | 77 OK |
| `public_shop_data` | 12 OK | 12 OK |

- **`node --test tests/test_current_site.mjs`:** 29/29 pass.
- **The 12-file public-shop Node set**, including `test_source_reference.mjs`: 291 tests, 289 pass, 0 fail, 2 `# SKIP` here. The skipped tests are:
  - #116 "a coherent old browser cache cannot hide the newly pinned packet";
  - #118 "actual pinned museum renders all cards and rejects cross-claim source, scope and replay substitutions".
- **BibTeX, as observed here.** `test_source_reference.mjs` passes 27/27 and prints "BibTeX binary unavailable; literal/identity assertions ran, real-parser integration omitted."
  - This container has no `bibtex` binary, so the real-parser path did not run.
  - This neither reproduces nor resolves the author's local `plain.bst` failure. GTW-BIBTEX-DIAG's diagnosis stands.
- **Other controls.** `public_shop_check` passes. The identity control is OK under `-S` and under `-O`.

### 5. My failure controls (kept private until now)
**a. Differential check of base against head**
- **Method.** I ran `current-model.mjs` from `aff60c50` and from `e80f370e` on the same graph objects.
- **Graphs:**
  - the admitted corpus (49 nodes, 55 edges), using every node plus the roots `missing`, `constructor`, `__proto__` and `toString`;
  - 3,000 seeded random graphs, with cycles, self-loops, parallel edges, relation ties, context edges and own `__proto__`/`constructor` IDs; about 20% also have unknown endpoints;
  - a 2,000-node chain;
  - a fan-out graph of 2,000 nodes and 10,000 edges.
- **Results.** Across 34,387 roots, 21,090 routes and 2,559,620 route-node occurrences, there was no difference:
  - `requiredDependencyPaths` is identical, including edge object identity.
  - `downstreamReach` is deep-equal, and `downstreamReachCounts` agrees with it.
  - All 5,220,886 windows, at limits 1, 2, 3, 7 and 150, concatenate to the base route sequence with exact incoming edges. No segment is empty, `complete` flags are correct, and page clamping is correct.
  - `directRelationWindow` pages reproduce the rows in source order.
- **Limit clamp.** An explicit `limit:1000` is clamped to 150 by both window functions.

**b. 19 mutants of `current-model.mjs`**
- **Method.** Each mutant ran against `node --test tests/test_current_site.mjs`. The file was then restored, and I confirmed its blob `e8944c9a` afterwards.
- **14 caught:**
  - incoming edge off by one;
  - unclamped page;
  - required-only reach ignoring the required flag;
  - no index cache;
  - no analysis cache;
  - route page 0 accepted;
  - route page skipping the safe-integer check;
  - relation rows re-sorted;
  - last-predecessor tie;
  - BFS cutoff;
  - materializing all preceding routes;
  - route length excluding the root;
  - root counted as a target;
  - reach walking outgoing edges.
- **5 survived.** These are N1 to N3 below.

### 6. Hosted evidence, read by ID (hosted; tested-merge checkout, not a head checkout)
**Run and jobs**
- **Run.** `public-shop` run `38615955800` (`pull_request`, attempt 1, head `e80f370e`, PR base `aff60c50`) ended in success.
- **Jobs.**
  - `public-shop` `116030959345`: success.
  - `browser-smoke` `116030960172`: success. It ran from 18:32:01Z to 18:34:43Z, which is 2 min 42 s of the 10-minute job budget. The atlas step ran from 18:34:10Z to 18:34:37Z.

**Artifact `11919627011`**
- **Download.** 35,443,680 B. My SHA-256 `cb4603d0ac15e2334b5a463f27cd9ebddca30eaa6e8d89407f6ef2296dfa57e7` equals the API digest and the upload log. It holds 188 entries.
- **`commit.txt`.** It names `bf8acd6e`.
- **Atlas report.** `current-atlas/report.json` (SHA-256 `4fb0ca8928c391ac…`):
  - `checked_commit` is `bf8acd6e`, on Chrome 154.0.8037.97;
  - **19/19 cases passed**, and every recorded `page_errors` is empty;
  - chain: 9,999 targets and 50,004,999 occurrences. Pages 1, 166,683 and 333,367 rendered, the last with 99 nodes. Elapsed time was at most 2.5 s, and the heap was 13–40 MB;
  - fan-out: 19,998 occurrences over 134 pages, the last with 48 nodes.
- **Library report.** `report.json` has `checked_commit` `bf8acd6e`, `pr_head` `e80f370e` and tree `b210c8b7`. 18/18 passed.

**Required checks on `e80f370e`**
- `verify` `116034593814`: success.
- `public-intake` `116030945404` and `116033740437`: both success.
- `public-shop` `116030959345`: success.
- Of the 13 check runs, the Pages `build` and `deploy` were skipped and all others succeeded.

### Non-blocking notes (follow-up, not this PR)
- **N1 (gap that predates this PR).** No model test pins the target order (length, then target ID) or the relation tie-break between parallel required edges, at the base or at the head. Mutants that break either one survive. The head equals the base on both, per 5a.
  - **Suggested test:** a longer route whose target ID sorts first, plus parallel required edges whose relations appear in non-alphabetical source order.
- **N2 (gaps in the new code's tests).** Two mutants survive, though the head is correct on both per 5a. The atlas never passes a limit.
  - An empty boundary segment: `<` instead of `<=` in the binary search.
  - An unclamped explicit `limit` above 150.
  - **Suggested assertions:** "no segment has zero ids" and "a limit above 150 is clamped".
- **N3 (gap that predates this PR).** No test pins the index's guard against edges with unknown endpoints. Admitted snapshots cannot contain such edges, because `validateSnapshot` rejects them.
- **N4 (optional).** Two small wording and test points:
  - When a route is cut at the end of a page, its first part is labelled "Continuation of …, nodes 1–k of L". The positions are exact, and the part is never labelled complete.
  - The route count "outside this page" is rendered and appears in the hosted report, but the browser check does not assert it.

### Release
This releases `REVIEW399-017-20261014`. I retain no claim, branch or pending action. I took no Ready, merge, integration, workflow or status action. Integrator selection and the merge gate, including `browser-smoke` on the final exact candidate, stay with dot. This verdict binds only `e80f370e`/`aff60c50`/`bf8acd6e`. If the head or base moves, a delta read would be a new offer.


---
_Generated by [Claude Code](https://claude.ai/code)_
