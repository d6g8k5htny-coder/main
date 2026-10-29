# Scoped accessibility audit of frozen F973

The baseline has two actionable findings: overlapping footer link targets, and an invalid labelled graph viewport in the source-failure state. This is **not a WCAG conformance certificate**. The criterion inventory distinguishes confirmed findings, observed checks, partial coverage, human assessment and criteria outside the tested page.

Actual performer: existing OpenAI GPT-6/Codex worker, previously the C150 source/report reviewer and the 26 October audit operator. Prior source exposure, same family/provider/shared principal, organizational credit0, scientific effectNONE; no fresh, blind, human or organizational independence claim. Dylan's personal reading is pending. No product or mapping changes were made.

## Exact target and execution chronology

- Product baseline: Main **f973ead67ec139dcfef9b97f1758523639c835b3**, tree **68977b754d1d85c213cd859fa6cde1a0e06a5f8e**; Math **7d63d08f6cfadfd9c2ef28039a370d741126f9a0**.
- Frozen observation:49nodes,55directed relationships,108excerpts; payload1,897,850B/SHA256 **e2a174d1b0aa3a348dedd5dd6b867aff7346b31a03ace20c39847c4b6ed8c957**. The replay's wall clock is28October; this is not a current public-source claim.
- The original143captured members remain unchanged. Four missing assets were added only in a separate local fixture from exact tracked **F973**, not938: `brand/mark.svg`, `dependency-source/GRAPH.json`, `dependency-source/PROVENANCE.json`, `node-status.json`. All147members are individually bound by `../INPUT_COMPLETE.json`, including supplement Git blobs and SHA-256. No original input was overwritten.
- The original26October hard bound22:45 expired during the pause; its terminal release was not posted then. Preservedrun06 JSON records23completed cases and a22:24:56finish, but the original process-exit/terminal receipt is unavailable. This closeout does not infer it or renew/backdate that stage.
- **New28October execution:**14:38:55–14:39:37UTC,41.980s, actual exit0/no timeout, receipt `EXECUTION_RESULT.json`. It completed23/23cases, including21axe states and2JavaScript-disabled browser-only states;121manual scenario assertions passed. There were0page errors and0missing-asset HTTP responses. Exit0 means the audit executed completely; it does not mean the page passed WCAG.

New pickup388/6471911101, mirror619/6471911557; CP14:55/hard15:20UTC28October/end. Current Main559/formal637 and the parent's bridge/candidate were not tested. Their source/composition/scientific qualification remains separate.

## Findings for the source owner

**F1 — confirmed footer click collision, P2; WCAG2.5.8.** At320/390CSSpx in both themes, `footer p a` rectangles are43.796875px high but successive lines start23.796875px apart, producing20px overlaps. Real mouse clicks two pixels above the bottom of visible “Universal Law” text dispatch to “Library (research workspace)”.1200px geometry has no overlap. Axe flags effective23.8px target size/spacing below24px. These are styled navigation links, not ordinary prose links. The usability defect is independently demonstrated by hit tests and actual click dispatch.

Minimal affected product path: **`docs/site/brand.css:133`**, whose inline-block links use `padding:10px 2px; margin:-10px 0`. Reserve nonoverlapping target space while preserving target size. Parent implementation and regression qualification are required. Evidence: `../manual-01/MANUAL.json`, `../deep-01/DEEP.json`, their footer screenshots, and fresh `run-07-20261028/graph-390-light-axe.json`. The manual click probes are actual26October executions; their bytes are preserved, not relabelled as28October runs. The fresh complete-input scan reproduces the target-size findings.

**F2 — confirmed invalid ARIA structure, P2; automated WCAG4.1.2 finding.** In the explicitly injected local `current.json` fetch-failure state, focus-enabled `#atlas-stage` is a `div` with `aria-label` and no valid named role. Axe reports `aria-prohibited-attr`. Minimal affected path: **`docs/site/dependencies.html`**. Give the intended viewport an appropriate valid named role while preserving the failure boundary. This is not a demonstrated screen-reader speech failure, and is distinct from the old missing-capture-assets setup issue. Evidence: fresh `fetch-failure-320-dark-axe.json` and screenshot.

**A1 — heading-order advisory.** With the foundations disclosure closed, H1 leads to the inspector H3. Axe tags this as best practice, not a definite WCAG AA failure. An earlier H2 such as “Explore claims and evidence” can restore the hierarchy while preserving all `#atlas-inspector h3` consumers. A visible H2 needs only the HTML change; any visually hidden version must remain in the accessibility tree. No generic visually-hidden utility was found in the inspected baseline styles. The source owner chooses the implementation.

**A2 — 3D scroll-list portability followup, not a demonstrated Chromium keyboard failure.** Axe flags the relationship UL at390px. Actual Chromium154 Tab reaches it, its UA focus outline is visible, and ArrowDown scrolls0→40px at320/390 in both themes. Full49node/55relationship text alternatives are present. Explicit named keyboard-scroll semantics in `docs/site/atlas-spatial-shell.mjs` can improve portability; other browser/AT qualification remains open. Do not count this as a failed keyboard interaction in the tested browser.

**A3 — unresolved automated coverage.** Normal labelled generic containers are axe incomplete, including `.atlas-primary-actions` and `.atlas-archive-notices`; they were not silently promoted to definite failures. Contrast checks remain incomplete for pseudo elements and partially overlapping/clipped graph nodes.28representative solid computed-color samples exceed4.5:1, ranging6.28–17.27, but do not settle every painted pixel or graph edge. `../contrast-computed.json` records the calculation and limits.

## What was actually exercised

The23-state matrix covers1200/390light/dark2Dgraph, reader/List and actual WebGL3D;320/390both-theme text spacing;320no-JS; refused pins; injected source-fetch and WebGL failure; forced colors and reduced motion. Native source/formal/relationship disclosures, selection→inspector→source Tab order, skip link, tabs, Go-to Escape, search live-equivalent and the500ms equation announcement were exercised. Real canvas orbit changes pixels and Tab leaves the canvas. Sampled source/disclosure/canvas focus has visible rings; text-spacing reader width remains320/390without horizontal document overflow.

The ordinary matrix renders the selected7node/6edge neighborhood. Separate preserved26October probes render the full49node/55edge dataset and verify all49node/55relationship alternatives, real3D selection and source focus. They are separately attributed. All scripted “manual” scenarios use real browser keyboard/mouse events orchestrated by Playwright; screenshots were visually inspected by the AI operator. No human audition or spoken AT is implied.

Representative screenshots reviewed include the fresh1200light graph,1200dark formal disclosure,320light text-spacing source focus,320dark no-JS and fetch-failure states; preserved320light/390dark footer targets,390dark3D keyboard focus and relationship scroll list. The screenshot paths are recorded in `SCREENSHOT_REVIEW.json`. Other screenshots are retained without a claim of individual visual inspection.

## Coverage and limits

`CRITERIA.csv`/`CRITERIA.json` explicitly classify all55WCAG2.2A/AAcriteria; this is a coverage inventory, not a claim that all55were fully evaluated. `CASES.csv`, `FINDINGS.json`, raw axe JSON and the execution receipt make the findings reproducible. Axe produced50distinct passing rule IDs; its heading-order and scroll-list flags still require the qualifications above.

Not established:200percent text-resize as a separate test; physical orientation/device behavior; actual speech or screen-reader user experience; other browser engines; exhaustive painted contrast/flash analysis; full site navigation, linked static-node destinations or authentication/form workflows. The capture is one Atlas page plus exact captured assets and four supplements, not the entire website. No-JS source/fallback content was inspected; that does not qualify every linked destination. Keyboard orbit alone does not settle WCAG2.5.7's single-pointer nondrag alternative requirement; that criterion remains a human/pointer followup.

Old launch socket-path failure, premature300ms announcement sampling, mobile3D viewport-pausing/probe fallback setup errors, missing static captures, and the extra keyboard probe's invalid assumption that an unmapped node always has an immutable source link are preserved. Corrected successor runs use fresh output roots; no old failure is relabelled as a pass. No source/sandbox/TLS/assertion weakening occurred.

## Tool and operational boundary

Sandboxed `/usr/bin/chromium` **154.0.8037.57**, Playwright **1.62.0**, Python **3.12.14**, Linux6.18.55, local HTTP, software SwiftShader and reduced motion. **axe-core4.10.3**, DEV-only, npm SHA-512 SRI verified; scriptSHA256 **880970c081707360e64f34cea25ff91892f5bc95675b0776925b9709dd8a68bb**, MPL-2.0 license retained. Complete package integrity/license records are in `../tooling/TOOLCHAIN.json`; no production dependency was added. Same-origin DEV script/spacing routes preserve CSP; the fresh replay blocks nonlocal page requests.

No product/source/index/ref/Ready/Main/Drive/settings or AT-role actions. No denied resource was inspected, installed, retried or rerouted, including the protected Playwright cache and APT-preflight paths. Parent source implementation, current637 mapping and SITE/AT/integration roles remain separate. Frozen inventory and completion receipt bind this new closeout only.
