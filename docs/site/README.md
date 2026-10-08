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
python3 -B tools/public_shop_check.py
```

These are the local steps of the required `public-shop` job. In the node list, two museum tests report SKIP without `MUSEUM_FIXTURE`; `tools/museum_check.py` reruns them with that fixture. The Run locally and museum blocks below are subsets of this list.

Check real desktop/mobile layouts, keyboard operation, light/dark appearance and source-unavailable behavior separately. The pure-model tests are not browser or scientific verification.


## Latest public work is a reading cut

`research.html#latest-work` is static editorial navigation checked at its displayed
UTC time. It links the landed soft-model chain at Math `07320089`, the separately
landed cap formalization at `0fda855b`, exact public C81–C84 issue-comment proofs and
reviews (model and source-bound actual/contact scopes kept distinct), the finite numerical work at main `1e1c9a1c`, and the existing catalog and
query sources. The named boundaries are part of each reading path. This is a
curated selection plus full upstream navigation, not a complete artifact catalog,
scientific-status register or automatic synchronization service.

The 18:00 UTC refresh ([PR248](https://github.com/d6g8k5htny-coder/main/pull/248))
added the conditional actual-bar cards `#actual-bars` and `#strict-bar-coefficient`
to this cut. Later dated cuts were added above it: `#reading-addendum` (3 October 2026,
20:01 UTC), `#pair-endpoint-rate` (4 October, 00:18 UTC), `#shrinking-bin-sampling`
(4 October, 06:20 UTC, [PR266](https://github.com/d6g8k5htny-coder/main/pull/266)) and
`#reading-cut-20261007` (7 October, 18:37 UTC). Home, the Research hero and the Library
link the newest cut. `#latest-work` keeps its 18:00 UTC heading, wording and pins; its
later additions are a forward pointer to the 20:01 addendum and two labeled
reviewer-lineage notes beside its 3 October review links, which the 7 October cut
discloses. Two further labeled source-identity notes close the 20:01 and 00:18 cuts
with the SHA-256 identities their pinned links lacked, and one dated
`<p class="cut-pointer">` inside the 7 October cap card records that Math #193 was
reopened eight minutes after that cut. The next dated cut restates all five, so none
is the only record.

Exact-commit citations preserve the reading cut. The **Check newer work** section
intentionally follows mutable upstream branches and discussions. Comment links
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

Insert the new `<section id="…" class="latest-work" tabindex="-1" aria-labelledby="…">` with its own `<time datetime="…Z">` immediately before the current newest cut. Never edit an earlier cut's text. In the same change:

1. Point Home's "Read the latest public work →", Research's hero "Latest public work" and the Library's latest-work links at the new id. Give the new cut one uniquely named link to the cut below it; exact link names must stay unique. Update the `<time datetime>` and visible date beside the Home and Library entry links to the new cut’s timestamp (`tests/test_reader_links.py` pins them). The Home About link line links Source records, Dependencies and Formal, and its disclosure links Measure; keep ‘dated’/‘synthetic’ in those labels. In the Research hero, add one row for the new cut at the top of the “Dated reading cuts” list (its question heading and the `<summary>` text of its source disclosure) and update the “Newest reading cut” `<time>`; if the cut introduces a status token not listed under `#status-words`, add a quoted-provenance row for it (where it is quoted from, never a definition).
2. In `tests/test_reader_links.py`, update the entry fragments, add the cut's pins to `EXPECTED_READING_URLS` and `EXPECTED_HASH_BINDINGS` (the parser joins the text of all `<code>` elements in an `<li>` into one string and binds that string to each pinned link in that `<li>`, so keep an item's `<code>` text equal to its intended digest), update the pinned-link total and the `p.latest-boundary` count, and add a byte-identity pin for the cut that is no longer newest. Three digests pin the page below the newest cut: `#shrinking-bin-sampling` whole; the cuts digest from `#pair-endpoint-rate` up to `#lifetimes`, with lineage notes stripped; and the paths digest from `#lifetimes` up to `#further-reading`. A cut that stops being newest gets its own byte-identity pin, as the 06:20 cut has. Only the Further-reading section and the footer stay outside the pins.
3. In `tools/public_shop_browser_check.py`, update `check_latest_work_flow` and `check_reading_addendum_flow`: entry focus, datetime, card and pinned-link counts.
4. Keep `research.html` script-free, its IDs unique and every local fragment resolvable.
5. Append the new cut, with its date and UTC time, to the list of later dated cuts earlier in this section, so this guide does not stop at an older cut.
6. Rerun the release tool and the public-shop steps above.

Rules for future cards, not a retrofit of pinned cuts: each card carries one visible primary pinned proof link on its face, with digests, reviews and manifests in the disclosure, and the test’s EXPECTED lists enumerate card-face and disclosure links separately. Each card ends with a `<dl class="card-status">` of up to three rows — “Landed?” (landed at Math `<commit>` / open proposal, quoting the PR state at the cut), “Read by” (provider · verdict token quoted from the linked review · scope) and “Scientific effect” (quoting the source’s own declaration) — reusing the 7 October labels “Who read it:” / “Still candidates:” / “Still informal:” verbatim where they apply. A display equation inside a future card is authored as `<p class="display-formula"><code>…</code></p>`, which `home.css` renders as a block; kickers and eyebrows are uppercased by CSS, so a future cut may author them in sentence case. Pinned cuts are not retrofitted.

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
On the museum page at 1280 and 390 px, each conditional-route summary and link,
and the exhibit figure of each of the four views, takes a 3px focus ring at a
4px offset that no clipping ancestor cuts on any side; each summary's ring ends
above the next summary's first text line, and at 390 px an opened quote has no
inline margin. On Dependencies, five nodes with long ids, paths or file names,
at 320 and 390 px with and without a WCAG 1.4.12 text-spacing override, keep the
page within the viewport and every Lane and Record state word on one line; the
index stays within 320 px under the override. A failure names the element, the
side or the word.
A setup artifact without a successful report is not a completed browser run.

Inspect the screenshots before accepting visual quality. This is headless Chromium
with mobile-sized viewports, not physical touch-device, all-browser, full
accessibility or deployed-Pages certification. The original source validation and
required hosted checks remain separate and unchanged.

Without the runner's `/opt/google/chrome/chrome`, a local pre-check can install `tests/browser-requirements.txt` in a scratch environment. That installs only the Python packages: if the environment has no Chromium, download one with `python -m playwright install chromium`, which places it under Playwright's browser cache (`~/.cache/ms-playwright` on Linux, or `PLAYWRIGHT_BROWSERS_PATH` when that is set). In a disposable worktree, edit `tools/public_shop_browser_check.py` so that both its `executable` path and its launch use a local Playwright Chromium binary (`executable_path=…`, with `chromium_sandbox=False` only if the host requires it). Never commit that edit. Run it in that worktree with `python -B tools/public_shop_browser_check.py --output <dir>`; `--output` is required, and the directory receives `report.json` and the screenshots. Such a run is local evidence only: it is not the CI browser record and does not meet the contract's packaged-Chrome and sandbox conditions. Its `report.json` records `checked_commit` and `tree` from the checkout, which identify the base checkout, not the program that ran. To cite such a run, keep the unmodified report, the exact invocation, the modified harness bytes with their SHA-256 (or the original blob id with a replayable patch and the resulting SHA-256), and the executable and sandbox setting used; say that the harness was modified, and do not present the report as the tool's own output. A run without that retained material is an informal observation, not reproducible execution evidence.

## Run locally

From the repository root:

```sh
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000/site/`. Internet access is required to read the immutable public GitHub inputs. If a hash or fetch fails, the affected view reports unavailable and infers no result.

```sh
python3 -B -m unittest discover -s tests -p test_public_shop_data.py
node --test tests/test_public_shop_frontend.mjs
python3 -B tools/public_shop_check.py
```

The interaction harness tests loading, searching, exact decimal selection and pinned query identity; it is not a visual browser test. Local-server preview was unavailable in the editing environment. The baseline live Pages site was checked in a real browser on 2026-09-26: all 2,138 original catalog records loaded and the exact SIDE24 source bytes verified. This baseline check does not validate a later museum deployment.

Reproduce the exports with `python3 -B tools/public_shop_data.py --check`. The exporter uses fixed source identities, refuses changed bytes or unfamiliar tables, and preserves verbatim cells. Updating the snapshot requires an explicit source-bound engineering PR; this viewer never decides a review disposition.

## Notebook

The [SIDE24 notebook](../notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb) exposes the same immutable input and exact values, with an optional approximate chart. See [notebook instructions](../notebooks/README.md).

## Publication on main only

GitHub Pages is configured as **Deploy from a branch → main → /docs**. The entry `docs/index.html` points to `site/`; `docs/.nojekyll` serves static files. GitHub Pages serves `docs/404.html` for any missing path under `/main/`: a static, script-free page with the brand header and the footer site map, linking `site/style.css` and `site/brand.css` by absolute path without a `?site-release=` key. The release tool computes that cache-busting key over `git ls-files` of `docs/site` and `docs/public-math` and rewrites it only inside the `.html`, `.css`, `.js` and `.mjs` files under `docs/site`; `docs/404.html`, like `docs/index.html`, lies outside that inventory, so neither page carries a key. The verified destination is [https://d6g8k5htny-coder.github.io/main/site/](https://d6g8k5htny-coder.github.io/main/site/). GitHub's branch publishing supports `/` or `/docs`, so `docs/site` is the app directory, not a selectable publishing root.

See the [deployment and review settings record](../PUBLIC_SHOP_SETUP.md). Pages, the public Project, the four profile pins, and the main/Math- branch rules have installation receipts. A public site does not establish branch protection or mathematical acceptance.

## Claim cards and exhibits

Open [the verification museum](museum.html). Its small `museum.json` display projection is reproduced by `tools/museum_data.py`; it is not a replacement for the 2,138-row public inventory. The generator preserves all eleven bullets in Math-'s “Reviewed scoped results”, the reconciled D1 row of STATUS's ACCEPT table, and the two remaining STATUS AMEND rows verbatim. An EXACT_COUNTEREXAMPLE retains that label; publication of a source does not make its claim accepted. Where a review is a GitHub comment, the displayed digest identifies the pinned index pointer, not mutable comment bytes.

The museum binds Math sources to `d6628da09384728992dcbe6e921cc28ba85aebb0` and main source/packet membership to `71400b94f6cb354a8cf7aba73ffede2138a64efa`. The existing board/inventory pin `a26f744be7597e3f0c0543c34ead23049bbac657` and SIDE24 import pin `9d7b6802424fb4715b31999066aafca8ee2f3cca` remain explicit historical identities. Neither changes the README's Math checkout or query checkout. No source fetch selects an AMEND branch.

Claims use claim/scope, source/review, and replay columns. Verified scope quotes have a small readable presentation for inline links, code and bold text. The complete original Markdown remains in **Exact source quote**. Relative links use the quoting index or STATUS file’s pinned repository and commit; quoted mutable links remain mutable. Only recognized research GitHub destinations become links, and unsupported syntax stays literal. These display links do not byte-verify their targets. Missing recorded replay commands are disclosed. The separate engineering strip and CI links do not attest to mathematical acceptance. D5's reviewed fixed-annulus region and its open pin-neighborhood/microdisk work remain separate objects.

- EC-014 illustrates the two-pin frame and six data coordinates. Camera motion is presentation, not a replay of its Jacobian argument. Imported self-labels are not adopted.
- D4/D5 diagrams show only their stated fixed regions, with OPEN complements. Diagram dimensions are schematic, not certified constants.
- P15 illustrates the source's stated finite realized-family example, with no unrestricted prize claim.
- D2 has a **fixture absent** card: no committed lifetime/barcode samples were found in the audited defaults. No synthetic proof data or live Gaussian sampler is supplied.
- The packet list contains two landed packages: the identity replay at main `71400b94f6cb354a8cf7aba73ffede2138a64efa` and the SIDE24 chart submitted for task #141 at main `a12c178c0f857a130cf434e9efd44233a038195b`. Each retains **REVIEW_REQUIRED**, scientific effect **NONE**, and **packet — not STATUS**. Unmerged PRs remain excluded.

```sh
python3 -B -m unittest discover -s tests -p test_museum_data.py
python3 -B tools/museum_check.py
node --test tests/test_museum_geometry.mjs
```

Existing intake enforcement remains `.github/workflows/public-intake.yml` and `tools/public_intake_check.py`: trusted `pull_request_target` code, no packet execution, incoming-only files, bounded identities, and a documented credential-pattern scan. The scan is not comprehensive. Task claims remain the human protocol in CONTRIBUTING; display inclusion never replaces review.

`museum_check.py` reads the pinned public bytes, checks regeneration, and exercises the actual committed manifest through the display code with the declared HTML containers. It also tests refusal of cross-claim proof, scope and replay substitutions. It is an integration harness, not a browser layout or GPU test. `museum_data.py --check` remains available for export-only reproduction.

The museum requests its mutable local `config.json` and `museum.json` with
`cache: 'no-store'`. Immutable remote source requests keep their existing caching
behavior. The manifest must still match the config's byte count and SHA-256 before
any cards render; a mixed deployment reports unavailable. This reduces browser
cache staleness, but is not an atomic deployment or a guarantee that an intermediary
cannot return an older coherent pair. A fresh browser read on 2026-09-26 verified
both landed packet cards after the earlier one-packet cached observation; that
observation and its correction are recorded on [main #154](https://github.com/d6g8k5htny-coder/main/issues/154).

The [implementation record](IMPLEMENTATION.md) maps W1–W12, coordination exclusions, and validation boundaries.

## Reproduce this source snapshot

The museum and exact lookup use these immutable checkouts. These commands
belong to the technical guide; the public home has no setup requirement.

```sh
git -C Math- checkout --detach d6628da09384728992dcbe6e921cc28ba85aebb0
git -C query- checkout --detach c88768bb11efd1f7d6bda188f13064bedec54a06
```

The source-identity gate checks these commands against `museum.json` and
`config.json`. Updating the reading interface does not update either source pin.
