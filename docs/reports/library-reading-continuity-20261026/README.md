# Keep the Library reader’s place

Main c447d102 rebuilt every saved row when another tab saved an item or source-result attachment finished. That removed focused source/reference/identity controls and closed open Saved identity disclosures. The bounded repair reuses rows by the existing literal entryKey, refreshes changed validated metadata/reference fields, and supplies a reachable focus fallback when an item disappears. It preserves the reading-list schema, sources, classifications and all mathematical content.

Root OpenAI Codex authored this engineering repair; actual GPT-6 family, finer variant not independently attested, organizational independence0. Separate engineering review may use the same family. Scientific review requires a different actual family; this release supplies no scientific review or theorem certification.

## Actual verification

- Initial39 browser controls against unchanged Main:33 focus/disclosure failures and6 existing controls passed; zero page errors. Original independent exact-Main audit also reproduced6 failures with2 outside-focus controls.
- Final49 focused browser cases passed at1440/light,390/dark,320/light: saved control roles, source identity after earlier-row removal, open/closed disclosures, validated metadata enrichment, duplicate import, actual removals/clear, queued-write/newer-focus, closed list and tab-only result reattachment. Native same-origin peer-tab storage events and the product UI were exercised. Fixture SHA metadata tests formatting, not authenticated source custody.
- Original reading-list13 and saved-source→Cite3 browser cases passed on final product bytes; zero page errors across these runs. All12 prescribed/focused command groups exited0: configured Node313tests/311pass/2expected museum fixture skips; reading-list/citation30; geometry10; Python12+46+77+14+5+5; integrated public-shop and museum checks passed.
- Canonical asset release/check: `fc8dff7ea5337eeb443c963d07e6c52cc4454f7f3ac6e217862ae6496edb38fd`. Workspace edit is solely the derived reader-module import key; generated observation bytes are unchanged.

`SOURCE.json` pins candidate product files. Browser reports retain actual Git checkout HEAD plus measured working-tree product hashes; the base HEAD is not a claim that modified product bytes equal unchanged Main. Original complete39-case RED and final49-case GREEN have different counts because additional controls were added before implementation. The extended baseline also observed the added duplicate-import/tab-only failures, with3 subsequently corrected singular-count harness errors retained in the private evidence package. One initial repetitive polling run was intentionally interrupted; it is not a complete result. No hidden product exception made the tests pass.

Browser was sandboxed headless Chrome156.0.8078.47 on Linux, SHA25695dc9473b4f904cb3ab0800fc7206081fcae12abc094a088bd741a13f1747b72. Local hash-pinned catalog fixtures replayed; unrelated remote requests refused. Viewport/keyboard checks are not physical touch, Safari, spoken assistive technology, full WCAG or high-node performance qualification. No public deployment or independent review is asserted by these local results; those have their own subsequent receipts.

## Replay

```bash
python3 -m pip install playwright
python3 -B tools/research_desk_continuity_browser_check.py --browser /path/to/chrome --output reading-continuity-evidence
python3 -B tools/research_desk_browser_check.py --browser /path/to/chrome --output reading-list-evidence
python3 -B tools/reading_list_citation_browser_check.py --browser /path/to/chrome --output citation-evidence
node --test tests/test_research_desk.mjs tests/test_reading_list_citation.mjs
python3 -B tools/site_asset_release.py --check
```

Use an installed sandbox-capable Chromium/Chrome executable; no browser binary is bundled. Complete prescribed reader checks are in docs/site/README.md. Exact preimages remain in the independently downloaded full Main #636 Drive archive1UV4f0z71WK1sztMGpAYd5Ko3AxzpXMYe, whole SHA2563479838a82a32c2c40f77f8d928b7859a301cdc274bfdc399c3702944d5e53f5. No Drive file was deleted, replaced or shared differently.

The broader interface still lacks an admitted public worker-service/authentication/ID-mapping contract. This repair advances reliable find→save→inspect→cite reading independently of that specific integration gap.
