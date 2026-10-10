# Public reading experience and research workspace

**Live:** [Universal Law](https://d6g8k5htny-coder.github.io/main/site/).

The public home introduces the mathematics before its implementation. All pages share Home, Explore, Research and Library navigation. Contribution options remain available from every page's footer site map, from Home's About section, and from the workspace's Join the work section.

- **Home (`index.html`):** a short introduction to random landscapes, thresholds and persistence; no source catalog is downloaded on this page.
- **Explore (`explore.html`):** three deterministic teaching models—negative-definite curvature, cubic gap scaling and a finite capacity example. Controls work locally with no remote data service. These illustrate their defined examples and do not certify a continuum result.
- **Research (`research.html`):** dated public reading cuts, newest first (Home and the Research hero link the newest), followed by the established question-to-proof paths. Pinned source citations, scoped formal packets and issue-only proofs retain distinct scope; separately labeled mutable upstream links reach newer work.
- **Library and workspace (`workspace.html`):** the existing exact coefficient viewer, dated status snapshot, hash-verified source inventory and contributor instructions. Old `index.html#board`, `#coefficient`, `#inventory` and `#contribute` bookmarks route here; without JavaScript the home still offers the workspace link.
- **Reproduce (`reproduce.html`):** a single pinned coefficient replay, prerequisites, actual expected output and the boundary of the check.
- **Cite (`cite.html`):** object-specific citation guidance and a local immutable-reference builder. It checks input syntax, not source existence or scientific status.
- **Source records (`museum.html`):** the preserved historical source-bound exhibits, review cards and conditional routes. It names its separate snapshot; later proof-index results are not silently imported into old cards.
- **Formal (`formal.html`):** what the original 27 September 2026 Lean lane's 13 scalar companions establish and what stays outside them; an explanation at that source snapshot, not a current formalization inventory or live acceptance register. Its dated "Since this snapshot" section points to later formal packages.
- **Dependencies (`dependencies.html`):** a dated, read-only claim-dependency snapshot derived from frozen source bytes; it does not update claim status or replace the live proof index.
- **Measure (`measure.html`):** a synthetic counts-to-density teaching example with invented inputs; it generates no random field and validates no theorem.

The front and technical layers use the same existing evidence. No second scientific register or server backend is added. Source identities are available in expandable disclosures, while object/class labels and unavailable states remain visible. The original coefficient artifact’s `scientific_acceptance: false` is retained; later scoped review remains separate.

## Verify the reader experience

```sh
# Stage any new or deleted file under docs/site or docs/public-math first (git add <path> / git rm <path>).
python3 -B tools/site_asset_release.py          # rewrites ?site-release= keys after ANY change under docs/site or docs/public-math
python3 -B tools/site_asset_release.py --check  # must print the token and exit 0; commit every rewritten file
python3 -B -m unittest discover -s tests -p test_public_shop_data.py
python3 -B -m unittest tests.test_site_asset_release
node --test tests/test_public_shop_frontend.mjs tests/test_explore_models.mjs tests/test_curvature_export.mjs tests/test_teaching_export.mjs tests/test_pin_export.mjs tests/test_palette_export.mjs tests/test_reader_navigation.mjs tests/test_source_reference.mjs tests/test_measurement_model.mjs tests/test_dependency_viewer.mjs tests/test_museum_frontend.mjs tests/test_workspace_navigation.mjs
python3 -B -S -m unittest discover -s tests -p test_reader_links.py -v
python3 -B -m unittest discover -s tests -p test_museum_data.py && python3 -B tools/museum_check.py && node --test tests/test_museum_geometry.mjs
python3 -B -E -S tests/test_public_shop_check_identity_control.py -v
python3 -B -E -O -S tests/test_public_shop_check_identity_control.py -v
python3 -B tools/public_shop_check.py
```

These are the local steps of the required `public-shop` job, including both W8b public source identity controls. In the node list, two museum tests report SKIP without `MUSEUM_FIXTURE`; `tools/museum_check.py` reruns them with that fixture. The Run locally and museum blocks below are subsets of this list.

On macOS, a symlink alias in the process temporary directory can make the site-asset fixtures compare a lexical temporary path with the canonical path returned by the release inventory. That portability defect is owned in [#294](https://github.com/d6g8k5htny-coder/main/issues/294) and is not repaired here. The example below is a trusted local invocation setup only: it uses the caller's existing writable temporary directory, resolves that directory to its physical path, rejects the checkout and its descendants, and exports `TMPDIR` before starting a new Python process. It creates nothing, hard-codes no workstation path, and is not a production-path or fixture change.

```sh
(
  checkout_dir="$(pwd -P)" || exit 1
  trusted_tmpdir="$(CDPATH= cd -P "${TMPDIR:?Set TMPDIR to an existing trusted temporary directory}" && pwd -P)" || exit 1
  case "$trusted_tmpdir/" in "$checkout_dir/"*)
      printf '%s\n' 'Choose an existing temporary directory outside this checkout.' >&2
      exit 1
      ;;
  esac
  test -w "$trusted_tmpdir" || exit 1
  export TMPDIR="$trusted_tmpdir"
  python3 -B -m unittest tests.test_site_asset_release
)
```

Check real desktop/mobile layouts, keyboard operation, light/dark appearance and source-unavailable behavior separately. The pure-model tests are not browser or scientific verification.

