# Conditional proof routes — cumulative-transfer pilot

Engineering projection only. Scientific effect: NONE.
This page describes one added view, not a new claim database, theorem verdict,
or general-purpose theorem prover. The original museum cards, source inventory,
STATUS snapshot, config and manifest retain their existing identities and meaning.

## The first route

`conditionals.mjs` reads the already-indexed cumulative-transfer correction from
`museum.json`. Its exact source is
`reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md` in Math-,
Git blob `044ac5fdaf403a38e33983e31f0ad69f8e76d6d5`, 3272 bytes,
SHA256 `83f653393dc6245980f6848e2bfc65ac9ad4c7d8304b028b88764356fd3825b6`.
The audited commit is `d6628da09384728992dcbe6e921cc28ba85aebb0`.
Both interfaces require that exact commit and matching immutable URLs.
A different 40-character string is not evidence that these bytes belong to it.
The exported projection retains the complete source descriptor as well as its digest.
The checked source fixture under `tests/fixtures/` is an exact reading copy of
those public bytes, not a new proof body or an authoritative source.

The view retains six inputs: the setup/quantifiers, four numbered hypotheses,
and the definition of N. One ALL edge joins those inputs to the conditional
limit. The positive-coefficient and zero-coefficient conclusions remain separate.
The warning that a cumulative asymptotic does not imply a density asymptotic
is retained. The ratio h/(kappa*r^m) -> 1 and one fixed lower comparison constant
are not replaced by pointwise comparisons or a fitted exponent.

Application evaluation is always NOT_EVALUATED in this pilot. Hypotheses are
not declared satisfied merely because their implication has a reviewed source.
The existing source/review card remains the source of quoted review information.
No independence credit, lemma closure or promotion is computed here.

## One projection, two interfaces

The browser reuses the museum module's single verified startup
(`verifiedMuseum`): config-to-manifest digest, central source projection and the
per-page byte cache are fetched and verified once per page fetch and shared by
`museum.mjs` and `conditionals.mjs`, so the route repeats no central request and
reads the cumulative proof bytes from the same cache the claim card verified.
It then checks the audited proof descriptor and its complete byte and Git-blob
identities before parsing any hypotheses. A rejected startup is not retained; a
later caller re-verifies instead of inheriting a stale refusal. A failed load
clears the conditional view and reports unavailable. Source text is rendered
with textContent, not HTML.
The new HTML section is an accessible text dependency view, not a new 3D exhibit.

The read-only CLI uses the same projection function. It checks the existing
config-to-manifest digest, selects the one source descriptor, and verifies a
local source file against it. The CLI makes no network request, does not execute
the source, and does not replay the central index or review semantics. The
browser retains those extra existing checks. A projection supplied via --check
must equal the complete regenerated statement: missing premises, ALL-to-ANY
changes, extra nodes, cycles or status fields are refused rather than adopted.

Run from the main repository root:

```sh
node --test tests/test_museum_conditionals.mjs tests/test_museum_conditionals_startup.mjs tests/test_museum_conditionals_cli.mjs tests/test_museum_identity.mjs
node tools/museum_conditionals.mjs --source tests/fixtures/cumulative_transfer_source.txt > /tmp/conditional-route.json
node tools/museum_conditionals.mjs --source tests/fixtures/cumulative_transfer_source.txt --check /tmp/conditional-route.json
```

`--root CHECKOUT` selects a different local checkout for config/manifest reads.
The emitted JSON is a disposable projection, not a committed second inventory.
The source must remain byte-identical to the named proof. A new mathematical
statement requires an explicitly reviewed update to this bounded projection.
The CLI tests use temporary metadata fixtures around the real proof bytes;
the hosted workflow also exercises the actual committed config and manifest.

## Scope and deployment

This adds a pilot to museum.html, not an exhaustive graph of P0.1, D1, D5,
SARD-G or all research obligations. No OR route is invented. Future alternative
proof routes need their own exact statements and source-bound interfaces;
induction must state its base case and a well-founded decreasing parameter.
A larger graph cannot turn circular support into proof.

The separate Math- scalar and one-endpoint Hessian certificates in PR89 are
author derivations awaiting review. They are deliberately not added to the
museum's accepted result cards. Neither supplies the additional-witness
covariance or determinant-weighted Palm estimate. query- remains a read-only
identity lookup.

Local verification covers the new module, CLI, source integrity, HTML wiring,
a lightweight DOM refusal harness, and a mocked successful browser startup
(`tests/test_museum_conditionals_startup.mjs`): shared config/manifest reuse
with `startMuseum`, claim selection, proof loading, host replacement, and the
fail-closed paths for altered proof bytes, a non-audited descriptor and a stale
manifest. It is not a real-browser, layout, GPU or live
network test, nor a replay of every unrelated repository test. The hosted workflow
records its own results. A draft PR is not a deployed change to the public site.

## Reconnaissance and next mathematical interface

Lean blueprints are an inspected precedent for making dependencies visible:
https://github.com/PatrickMassot/leanblueprint. This implementation installs no
Lean toolchain and creates no kernel-checked proof. The analogy concerns
presentation and source navigation only, not formalization or acceptance.
The full local source/proof identities and limits are the controlling context
for this pilot; literature analogies do not discharge its hypotheses.

For a future D5 multiscale proof, a useful conditional interface is explicit:
if a complete weighted counting partition yields E[N]=sum_j J_j(r)/Z_r,
Z_r>=c_Z*r^2, and J_j(r)<=A*r^5*2^(-gamma*j) uniformly with gamma>0,
then E[N]<=A*r^3/[c_Z*(1-2^(-gamma))]. This is geometric-series arithmetic,
not evidence that these bounds or the partition hold for D5. Every collar,
axis, remainder and law-identification interface must be supplied. A first
moment does not by itself establish the separate elder-event inclusion.
No cumulant hierarchy or new PDE/gravitational interpretation is installed.

## Identity follow-through

The initial offline implementation checked commit syntax, digest and URL
consistency, but could accept correct bytes attributed to a different,
syntactically valid commit if the fixture manifest and URLs were changed
coherently. Three test-first controls exposed that gap (24 old tests passed,
3 new tests failed). The source selector now requires the audited commit; the
exported route also carries the complete repository/path/commit/blob/bytes/hash
and URL descriptor. Changing source revision requires a reviewed projection
update, not relabeling a local proof. This is an author correction, not a
nonauthor review or a change to mathematical status.

A Chromium smoke-test attempt in this continuation was blocked at localhost
navigation with ERR_BLOCKED_BY_ADMINISTRATOR, before any module assertion.
No browser protection was changed; this is a recorded verification gap, not a pass.
