# Research evidence reader

Owner request: implement beneficial parts of the supplied 182-point UI/UX review. Scientific effect NONE. Native pickup main#246 comment5971685261; base52252bc8ab32df5f7d96620c9965a7e85343447c.

## Intent

Help a visitor move from a recorded object to its scope, evidence, dependency context and exact source. Preserve the static architecture, existing URLs, dated snapshots, source bytes and scientific classifications. User authorization permits implementation and integration without another approval queue.

## First increment

Extend the existing verified dependency viewer rather than create a competing research database. Display a selected object's recorded scope and notes first; disclose its complete unchanged metadata beneath an audit heading. Show an evidence table with actionable source/review links and explicit “not recorded” values for missing review, reproduction, formal and alignment interfaces. A linked record is not a verification tick or acceptance. No review independence credit is inferred.

Add exact-classification filtering, an all-records view, clear filters, and shareable query/classification URL state. Selection remains independent of filters. Unknown saved classifications visibly refuse instead of silently switching scope. Back/Forward restores both filters and selection without stealing focus. No source status becomes a frontend-authored value.

Use existing colors, keyboard controls, responsive grids and reduced-motion behavior. Evidence has an accessible HTML table. No backend, new dependencies, global vocabulary, progress scores, timeline comparison or invented scientific summaries.

## Scope and limits

The graph contains recorded scope/notes but lacks consistently typed establishes/excludes, review-source identities, reproduction and formal alignment fields. This increment does not infer them from paths or green CI. It labels the date and missing metadata; richer object views require an exact source export with explicit field mapping and review.

## Validation

Model tests for non-inference, safe links, full recorded classes, filtering intersection, and malformed URL state; actual module startup tests for controls and restored state. Hosted browser covers desktop/mobile light/dark, filter URL reload, evidence/scope, audit disclosure, focus, and no overflow. Existing byte validation/refusal and full applicable suites remain required before guarded merge.
