# Parallel reader tools — 3 October 2026

Goal: complete two disjoint, already-authorized roadmap improvements without duplicating scientific metadata or another writer's work.

Source: `docs/UI_UX_REVIEW_20261003.md` and the existing Explore/Cite interfaces at main `8a48aba827af261bc0f0c8afa662fcc8dd70cd53`. Native scope claim: main229 comment5973643548. The owner's standing approval covers implementation and guarded integration; external human review is not a prerequisite.

Global constraints: static local tools; no backend, external fonts, dependencies or scientific-status changes. Keep exact source provenance and teaching/verification boundaries. C121 retains Research/index content, `tests/test_reader_links.py`, `tools/public_shop_browser_check.py`, and generated asset keys until concrete handoff. Implementers must not mutate those paths or publish branches. Root serializes integration and refreshes live refs.

## Task 1: Meaningful curvature presets

Add keyboard-operable ordinary buttons to the existing curvature controls for Maximum (s=-2,R=1), Saddle (s=0,R=1), Singular boundary (s=-1,R=1), and Minimum (s=2,R=1). Preserve the reset. Presets must use the same calculation/draw/state path as the sliders so readouts, diagram, accessible explanation, URL history, share link and export citation all describe the same state. Never label a singular Hessian a strict maximum/minimum. Manual slider changes need no persistent preset selection if it would mislabel a custom state. Preserve no-JavaScript explanations and disabled controls.

Write meaningful regressions first in `tests/test_explore_models.mjs`, testing actual production wiring or a small extracted controller with real callbacks. Test every preset's eigenvalues/classification, unknown preset refusal, input/state commits and consistency with current export state. Production scope: `docs/site/explore.html`, `explore.mjs`, `explore-models.mjs`, `explore.css` only as needed. Use existing styles and avoid a new module if unnecessary.

## Task 2: BibTeX source-reference template

Extend the existing locally validated source reference with BibTeX for its exact repository/commit/path and optional supplied digest. This is a source-reference template, not discovered bibliographic metadata: no inferred author, year, DOI, publication type, proof title, review or acceptance. Use an honest generic source description and `@misc`. A documented placeholder citation key is acceptable; explicitly tell readers to replace it before combining entries. Properly escape BibTeX special characters while preserving valid Unicode and URL identity. Avoid TeX command injection from accepted source paths. Do not confuse escaping with source validation.

Expose readable/selectable BibTeX output and a Copy BibTeX button alongside text/JSON. All copy formats use the existing single-operation lock, stale-output invalidation, clipboard-denied/manual-copy fallback, URL restore and Back/Forward contract. An edit clears every generated output until rebuild. Test behavior first in `tests/test_source_reference.mjs`, including Unicode/special characters, no invented metadata, validation refusal, copy concurrency and stale completion. Production scope: `docs/site/cite.html`, `cite.mjs`, `source-reference.mjs`.

## Review and integration

Each task gets one nonauthor spec/code review bound to exact commit/tree and changed files; fixes get focused re-review. Run affected Node tests, then the applicable full frozen-tree checks. Root obtains concrete handoff for shared browser regression wiring and generated release keys; do not claim green historical browser checks validate either new interaction. Publish one reviewed integration PR when shared integration is available, inspect push and PR workflows at current head/tested tree, expected-head guarded merge, verify landed checks/deployment, then read back delivery/release.

If source metadata is absent, record it as absent instead of adding inferred data. This increment does not complete the entire research-object or historical-snapshot roadmap.

## Integration successor

C121 completed and explicitly released the shared harness, reader tests and deterministic release-key paths in native handoff5973901687. Integration claim5973963677 starts from landed main4aadc771e8ffc08ee56bac63991a6ceb282a4292 and includes the separately reviewed recorded-context increment. Retain the dated Research addendum and all16 browser cases. New coverage guards were observed failing before helper wiring; the real hosted run remains the authority for whether native browser controls executed. Generate one combined release after authored site bytes stabilize. Source-bound reviews of the three implementations remain prior evidence; frozen whole-branch review and fresh actual-head checks are separate integration evidence. Scientific effectNONE.
