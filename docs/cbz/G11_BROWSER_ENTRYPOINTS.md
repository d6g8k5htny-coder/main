# Browser-check entrypoints in the G11 census

## Purpose and scope

The browser-check programs below are retained source-bound regression drivers
for the public reader and Atlas interfaces. Their individual reports and the
feature reports linked from `docs/site/README.md` preserve the real-browser
assertions, source identity, failures and limitations that existed when those
checks ran.

G11 additionally exercises each program's command-line parser and import
boundary in `tests/test_browser_tool_entrypoints.py`. That test invokes the
actual file with `--help` and requires exit 0, empty stderr, a nonempty
`usage:` banner and the selected file's own name. Missing files and a program
that refuses `--help` are explicit negative controls.

This is not browser execution. Parser/import coverage does not requalify an
old browser receipt, prove that a browser scenario passes on the current tree,
establish accessibility or scientific acceptance, or replace the browser
requirements and source-bound commands documented for each feature.

## Registered entrypoints

Atlas drivers:

- `tools/atlas_accessibility_audit.py`
- `tools/atlas_accessibility_regression.py`
- `tools/atlas_accessible_list_browser_check.py`
- `tools/atlas_approachable_browser_check.py`
- `tools/atlas_claim_summary_browser_check.py`
- `tools/atlas_formal_evidence_browser_check.py`
- `tools/atlas_live_controls_browser_check.py`
- `tools/atlas_refusal_browser_check.py`
- `tools/atlas_relationships_browser_check.py`
- `tools/atlas_resilience_browser_check.py`
- `tools/atlas_source_bridge_browser_check.py`
- `tools/atlas_spatial_browser_check.py`
- `tools/atlas_spatial_labels_browser_check.py`
- `tools/atlas_webgl_fallback_browser_check.py`

Reader and research-workspace drivers:

- `tools/compare_experience_browser_check.py`
- `tools/figure_png_browser_check.py`
- `tools/library_discovery_browser_check.py`
- `tools/library_source_search_browser_check.py`
- `tools/persistent_status_browser_check.py`
- `tools/reader_jump_browser_check.py`
- `tools/reading_list_citation_browser_check.py`
- `tools/research_desk_browser_check.py`
- `tools/research_desk_continuity_browser_check.py`
- `tools/research_desk_undo_browser_check.py`
- `tools/research_reading_browser_check.py`
- `tools/snapshot_compare_browser_check.py`
- `tools/source_list_continuity_browser_check.py`
- `tools/website_navigation_browser_check.py`

## Exact verification commands

Run the bounded parser/import contract in all supported interpreter modes:

```sh
python3 -B -m unittest tests.test_browser_tool_entrypoints
python3 -B -O -m unittest tests.test_browser_tool_entrypoints
python3 -B -OO -m unittest tests.test_browser_tool_entrypoints
```

Run the G11 auditor controls, mutations and exact-revision census separately:

```sh
python3 -B -S -m unittest tests.test_g11_census
python3 -B -S -O -m unittest tests.test_g11_census
python3 -B -S tools/cbz/g11_mutation_selftest.py
python3 -B -S -O tools/cbz/g11_mutation_selftest.py
python3 -B -S tools/cbz/g11_census.py --rev HEAD --json g11-census.json
```

The G11 result certifies only the bounded static documentation/reference rules
implemented by that auditor. Real-browser reruns still use each program's
own `--help` output and the feature-specific commands in `docs/site/README.md`
and the corresponding report README.
