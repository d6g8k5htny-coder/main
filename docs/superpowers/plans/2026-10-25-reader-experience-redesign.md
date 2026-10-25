# Reader experience redesign implementation plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task by task. Review each task and the whole branch. The owner has delegated execution and design decisions; proceed without routine approval pauses.

**Goal:** Replace the demonstrated source-first/scroll-heavy presentation with a clear mathematical reading flow and usable Atlas shell.

**Architecture:** Keep the existing static site, verified source pipeline and graph controllers. Change generated source presentation, reading-page hierarchy and Atlas layout in sequential independently reviewed tasks. Preserve raw mathematical data, historical cuts, sources and existing peer functionality.

**Tech stack:** Existing Python builder, HTML/CSS, vanilla ES modules and Node/Python tests; installed Chromium/Playwright for local acceptance.

**Spec:** docs/superpowers/specs/2026-10-25-reader-experience-redesign.md

## Global constraints

- No Google Drive deletion. Archive GitHub content before removal/replacement, with paths, revision and SHA256; log every removal.
- No mathematical source/status/provenance mutation. No inferred associations or fake current results.
- Preserve existing source verification/refusal, static fallbacks, historical IDs/citations and meaningful evidence labels.
- No new theme framework, remote font/analytics or backend. Preserve brand.css and brand tokens.
- No foreign branch mutation or repair of602/399/501/503/607 behavior. Respect current ownership and protected integration.
- Canonical asset helper runs once inputs stabilize, followed by--check; include every derived rewrite.

## Review focus

- A shared/deep URL into a newly collapsed historic section must reveal its target and preserve focus/history.
- A failed or mismatched current observation must remain visibly unavailable, with no stale success inferred.
- Long source IDs/formulas and text enlargement must not cause document-wide overflow or obscure actions.
- Current-data generation must retain compact presentation rather than reinstating the source wall at the next scheduled build.
- Styling or moving the Atlas controls must preserve the selected record, graph/list parity and peer metadata-filter semantics.

## Task1: Compact verified source presentation

**Files:** tools/current_site_build.py; tests/test_current_site_build.py; narrowly required docs/site/current.mjs/tests only if the existing selectors cannot support the new wrapper.

**Interface:** current_section retains its validation inputs and source DOM selectors. Add an explicit compact presentation parameter selected by top-level page generation; archived/static node views keep their documented scope. Pass capacity_html into the generated disclosure rather than appending it outside; retain full/archive defaults so preserved snapshot bytes do not change. All existing status/search/result IDs and routes remain unique.

- [x] Add a failing generated-markup regression requiring the dated status before a closed accessible Sources and scope disclosure; the legend, full pins, capacity and inventory occur inside that disclosure. Verify the failure on baseline.
- [x] Implement the wrapper/page presentation selection without changing data validation or history custody. Keep failed status visible and script-free Formal supported.
- [x] Run focused builder/current-module controls, including generation twice and failed-source interpretation. Freeze task and obtain independent source/spec review.

## Task2: Reader-first public pages and shared shell

**Files:** docs/site/{index.html,explore.html,research.html,workspace.html,brand.css}; new docs/site/reader-shell.mjs if needed for fragment disclosure restoration; corresponding tests/test_reader_links.py and focused new reader-journey tests.

**Interface:** Task1's current-source marker still generates. Existing teaching controllers/inputs and historiccut IDs/bodies remain available. Any shared helper only restores disclosure targets and reader presentation, never scientific state.

- [x] Pin failing task-based regressions for content ordering and deep historic links before markup changes. Verify failures against baseline.
- [x] Make Home's question/action/visual compact; place Explore's real existing experiment ahead of source machinery; give Research a concise reading index with explicit dates and accessible preserved cuts; clarify Library/source tasks and update mutable board routing.
- [x] Refine shared navigation/spacing/type/button hierarchy using existing brand tokens; preserve conspicuous critical limits, keyboard focus and dark/light/print rules.
- [x] Run focused tests and real390/1440 browser journeys, including deep links, no-script and failure states; independent review before Task3.

## Task3: Focused Atlas presentation

Ruling: Task3 was authored in a separate linked worktree with a new layout-only atlas-layout.css, so it could proceed independently of Task2 brand.css. Both task reviews precede composition; the palette/controller interfaces stayed fixed.

**Files:** docs/site/dependencies.html; new layout-only docs/site/atlas-layout.css loaded after current.css; corresponding focused presentation controls and optional real-browser focus geometry regression. The original dependency-viewer heading expectation is updated to the nested archive hierarchy. Do not edit atlas.mjs filtering or graph algorithms.

**Interface:** Preserve every controller-consumed ID, source record, static route and selection/reader behavior. Keep the inspector as sibling of the browse wrapper, with the browse wrapper first; diagnostics remain immediately within their disclosure. Retain the dereferenced atlas-toggle-view button hidden while atlas-view is the sole visible switch. current.css loads after brand.css, so use appropriately scoped Atlas rules. Compact primary search/view/expand controls; secondary controls and verbose coverage remain accessible on demand. Legacy viewer retains its exact source identity under an explicit archive disclosure.

- [x] Add failing presentation/behavior controls for useful list/graph visibility, one primary view choice, reachable archived fragments and selected-node return.
- [x] Reorganize the shell and responsive layout. Keep mobile list-first and preserve loading/error/static reading paths.
- [x] Run focused Node/shell controls and browser search/select/expand/Graph/List/reader-return/history at320/390/1440 widths; independent task review.

## Final integration and delivery

Rulings: adapt only user interaction steps in the two existing browser harnesses to open new native disclosures; all1028 existing assertion calls are retained. Preserve399 route/controller assertions. Apply the compact shared header and source-panel styles to supporting pages as well; the actual10page390px regression reproduces207px versus153px chrome before the fix and153px uniformly afterward. Full219site and168tools/tests preimages are independently downloaded and hash-verified in Drive; no Drive deletion.

- [ ] Freeze logical changes; run canonical asset release then--check; include expected full importer closure.
- [ ] Run the site guide's applicable local public-shop/link/model checks once on frozen inputs, broadening only for failures/new changes. Preserve real failures.
- [ ] Whole-branch independent source/spec review and fresh generated-site browser comparison, including enlarged text, keyboard and source refusal. Report measured positions against the spec gates.
- [ ] Complete the Drive archive/change log, deletion ledger and download package. Revalidate current main, peer scopes and candidate composition before native branch/PR publication.
- [ ] Obtain exact-current review/required checks, use a separate eligible integrator for any protected merge, verify the actual landed/deployed result and release only this task's claim. If a necessary external gate remains unavailable, deliver the reviewed working candidate and precise blocker without claiming production completion.
