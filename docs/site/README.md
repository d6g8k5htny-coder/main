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


## Latest public work is a reading cut

`research.html#latest-work` is static editorial navigation checked at its displayed
UTC time, except its closing **Check newer work** route list (`#newer-work`), which
carries its own "as read" date. It links the landed soft-model chain at Math `07320089`, the separately
landed cap formalization at `0fda855b`, exact public C81–C84 issue-comment proofs and
reviews (model and source-bound actual/contact scopes kept distinct), the finite numerical work at main `1e1c9a1c`, and the existing catalog and
query sources. The named boundaries are part of each reading path. This is a
curated selection plus full upstream navigation, not a complete artifact catalog,
scientific-status register or automatic synchronization service.

The 18:00 UTC refresh ([PR248](https://github.com/d6g8k5htny-coder/main/pull/248))
added the conditional actual-bar cards `#actual-bars` and `#strict-bar-coefficient`
to this cut. Later dated cuts were added above it: `#reading-addendum` (3 October 2026,
20:01 UTC), `#pair-endpoint-rate` (4 October, 00:18 UTC), `#shrinking-bin-sampling`
(4 October, 06:20 UTC, [PR266](https://github.com/d6g8k5htny-coder/main/pull/266)),
`#reading-cut-20261007` (7 October, 18:37 UTC) and `#reading-cut-20261009` (9 October, 00:05 UTC; the landed finite Gaussian H0 lifetime source and its program, read at main `0be54aa2`). Home, the Research hero and the Library
link the newest cut. `#latest-work` keeps its 18:00 UTC heading, wording and pins; its
later additions are a forward pointer to the 20:01 addendum and two labeled
reviewer-lineage notes beside its 3 October review links, which the 7 October cut
discloses, and the maintained contents of its closing `#newer-work` route list
(below), which is navigation, not cut text. Two further labeled source-identity notes close the 20:01 and 00:18 cuts
with the SHA-256 identities their pinned links lacked, and one dated
`<p class="cut-pointer">` inside the 7 October cap card records that Math #193 was
reopened eight minutes after that cut. The 9 October 2026 cut (`#reading-cut-20261009`) restates all five in its source disclosure, so none
is the only record.

Exact-commit citations preserve the reading cut. The **Check newer work** section
(`#newer-work`) intentionally follows mutable upstream branches and discussions. It is
maintained navigation, not cut text: it carries an "as read" date, and when the
coordination board or task queue moves, its routes and the matching Join-the-work card
in `workspace.html` are updated in place without re-pinning any cut
(`NewerWorkRoutes` in `tests/test_reader_links.py` checks them). Comment links
identify the public record but comments remain editable; their reviews record the
reviewed identity. A missing source is unavailable evidence. No browser fetch or
execution of those sources is claimed. The Library, status and museum snapshots
retain all their original pins and are not refreshed by this section.

The existing four light/dark desktop/mobile source-card browser cases now also
exercise the Home → Latest public work route, native keyboard fragment focus,
repeated source-identity disclosure, Back/Forward navigation, no document overflow
and the static route with remote source requests refused. Each case retains a
latest-entry screenshot. Actual source identity readback is a separate engineering
check, not something established by a screenshot or link.

### Adding a dated reading cut

Insert the new `<section id="…" class="latest-work" tabindex="-1" aria-labelledby="…">` with its own `<time datetime="…Z">` immediately before the current newest cut. Never edit an earlier cut's text; the closing **Check newer work** route list (`#newer-work`) is maintained navigation with its own as-read date. In the same change:

1. Point Home's "Read the latest public work →", Research's hero "Latest public work" and the Library's latest-work links at the new id. Give the new cut one uniquely named link to the cut below it; exact link names must stay unique. Update the `<time datetime>` and visible date beside the Home and Library entry links to the new cut’s timestamp (`tests/test_reader_links.py` pins them). The Home About link line links Source records, Dependencies and Formal, and its disclosure links Measure; keep ‘dated’/‘synthetic’ in those labels. In the Research hero, add one row for the new cut at the top of the “Dated reading cuts” list (its question heading and the `<summary>` text of its source disclosure) and update the “Newest reading cut” `<time>`; if the cut introduces a status token not listed under `#status-words`, add a quoted-provenance row for it (where it is quoted from, never a definition).
2. In `tests/test_reader_links.py`, update the entry fragments, add the cut's pins to `EXPECTED_READING_URLS` and `EXPECTED_HASH_BINDINGS` (the parser joins the text of all `<code>` elements in an `<li>` into one string and binds that string to each pinned link in that `<li>`, so keep an item's `<code>` text equal to its intended digest), update the pinned-link total and the `p.latest-boundary` count, and add a byte-identity pin for the cut that is no longer newest. Four digests pin the page below the newest cut: the 7 October cut whole (cut-pointer included); `#shrinking-bin-sampling` whole; the cuts digest from `#pair-endpoint-rate` up to `#lifetimes`, with lineage notes and the contents of the `#newer-work` aside stripped (its opening tag and its place as the last child of `#latest-work` stay pinned); and the paths digest from `#lifetimes` up to `#further-reading`. A cut that stops being newest gets its own byte-identity pin, as the 06:20 cut has. Outside the pins stay the contents of `#newer-work` (maintained routes, checked by `NewerWorkRoutes` in `tests/test_reader_links.py`), the Further-reading section and the footer.
3. In `tools/public_shop_browser_check.py`, update `check_latest_work_flow` and `check_reading_addendum_flow`: entry focus, datetime, card and pinned-link counts.
4. Keep `research.html` script-free, its IDs unique and every local fragment resolvable.
5. Append the new cut, with its date and UTC time, to the list of later dated cuts earlier in this section, so this guide does not stop at an older cut.
6. Rerun the release tool and the public-shop steps above.

Rules for future cards, not a retrofit of pinned cuts: each card carries one visible primary pinned proof link on its face, with digests, reviews and manifests in the disclosure, and the test’s EXPECTED lists enumerate card-face and disclosure links separately. Each card ends with a `<dl class="card-status">` of up to three rows — “Landed?” (landed at Math or main `<commit>` / open proposal, quoting the PR state at the cut), “Read by” (provider · verdict token quoted from the linked review · scope) and “Scientific effect” (quoting the source’s own declaration) — reusing the 7 October labels “Who read it:” / “Still candidates:” / “Still informal:” verbatim where they apply. A display equation inside a future card is authored as `<p class="display-formula"><code>…</code></p>`, which `home.css` renders as a block; kickers and eyebrows are uppercased by CSS, so a future cut may author them in sentence case. Pinned cuts are not retrofitted.

## Keep or share a Library search

The Library's search, repository and path filters are saved in the current URL
without adding a browser-history entry for every keystroke. Refreshing or
returning with Back/Forward restores the filters. **Link to this search** gives a
bookmarkable link directly to the catalog; **Clear filters** restores its initial
50-row view. Filtering never changes source identities or recorded review scope.

Search links use `q`, `repository` (`main` or `Math-`) and `path`. Empty values are
omitted; the first repeated value is used, and an unknown repository is treated
as all repositories. Unrelated parameters and section bookmarks are preserved in
the current address but omitted from the explicit search link. Filters are visible
in URLs and browser history. Links use the catalog’s pinned source records, not a
live proof index; the search URL does not freeze future catalog updates. If history updates are unavailable, filtering and the explicit
link still work; if source verification fails, controls stay unavailable.

The Library's static text names its bytes. `#status-rendered-from` states the SHA-256 and byte count of `status.json` and of the `STATUS.md` snapshot that `config.json` pins, and `#proof-link` carries the pinned proof URL that `app.js` also assigns; `tests/test_reader_links.py` pins both to `config.json` and to the payload bytes, so refreshing `status.json`, `config.json` or the SIDE24 pins means updating those sentences and the three `<noscript>` routes in the same change. A refused source prints `Unavailable: … No result inferred.` followed by a link to the pinned source on GitHub, built only from a pin whose repository, commit and path pass the displayed-identity guard that `core.verifiedBytes` applies to every pinned source (owner-fixed repository, 40-hex commit, no empty, `.` or `..` path segment, each segment URL-encoded).

## Browser evidence in CI

The existing `public-shop` workflow also runs a separate read-only `browser-smoke`
job. Its optional [pinned Playwright dependencies](../../tests/browser-requirements.txt)
use the [official Python package](https://pypi.org/project/playwright/1.62.0/)
with the Ubuntu 24.04 runner’s packaged stable Chrome and its enabled sandbox.
The browser version/executable hash and runner-image identity are recorded; the
browser is runner-bound rather than a fixed Playwright download. The runner serves
only `docs` on an ephemeral
loopback port. It does not publish a preview or use account credentials.

`browser-smoke` is not a required status check: as read on 7 October 2026, ruleset 23798639 requires only `verify`, `public-intake` and `public-shop`. Inspect its current-run report before integrating a site change.

The `public-shop-browser-<run-id>` artifact contains a JSON report and screenshots
for 1200×900 and 390×844 in light and dark mode, plus an explicitly refused
inventory request. It tests URL round-trips, actual Back/Forward navigation,
keyboard reset/focus and document overflow against the real app and pinned public
sources. Four additional source-card cases follow the Research reading path,
check exact review-comment links and retained qualifiers, and repeatedly open
and close the original-quote disclosure with the keyboard. They also capture
the Home, Explore, Cite, Reproduce and Formal entry viewports and check for
document overflow. The report binds the actual tested checkout separately from
the PR head. Two reader-tool cases follow the shared Research links at desktop
and narrow viewports, exercise the synthetic normalization/error boundary, and
verify the dependency viewer's pinned counts, review-source search, saved link,
invalid-link refusal and source-snapshot label.
Two rendered-layout cases read what Chromium drew rather than the stylesheets.
Focus rings are checked by painted pixels; everything else by computed styles
and laid-out boxes. On the museum page, `#museum-state` and
`#conditional-route-status` each keep one height at 1280 px from their first
text (verification held at the manifest request) to their last. At 1280, 390 and
320 px, in the light and the dark scheme, each conditional-route summary and
link, and the exhibit figure of each of the four views, is focused and
screenshotted, then blurred and screenshotted again: in the ring band, 4 to 7 px
outside the element (outside each line of a link that wraps), changed pixels
must run along at least 97% of every side and fill every outer corner (for a
target taller than the viewport, the viewport grows in height for these
screenshots). Its computed ring must also be solid 3px at a 4px offset, in a
colour with non-zero alpha and at least 3:1 against the background composited
behind it, with no ancestor whose overflow is not visible cutting the ring box.
No summary's ring box may cross any other text line in the card, above or below
it, and at 390 px an opened quote has no inline margin. The optional Three.js
request is refused, so every runner checks EC-014's 2D figure; the 3D canvas's
own ring is not checked. On Dependencies, five nodes with long ids, paths or
file names, at 320 and 390 px with and without a WCAG 1.4.12 text-spacing
override, keep the page within the viewport. No text line may be cut. Walking
outward from it, an ancestor that clips (overflow hidden or clip, paint
containment, content-visibility auto) must hold it inside its padding box, and
an ancestor's clip-path inset() inside that inset rectangle; any other clip-path
is reported as unmeasured. An overflow auto or scroll region makes its text
reachable on that axis, and from there the region's own scrollport must be held
by every ancestor further out and by the viewport. Every Lane and Record state
word stays on one line inside its cell, and the evidence table fits its scroll
region. Under the override the table may be wider only if that region scrolls to
its full width and a keyboard reader can use it: Tab reaches it from the
previous stop, its ring passes the same painted, contrast and clipping checks as
the others, arrow keys scroll it to the ta