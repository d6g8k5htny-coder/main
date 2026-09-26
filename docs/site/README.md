# Public research shop

This static viewer makes the existing public work easier to read. It does not own scientific status.

**Live:** [Open the public research shop](https://d6g8k5htny-coder.github.io/main/site/). Coordinate a bounded contribution on the [Public contributions board](https://github.com/users/d6g8k5htny-coder/projects/1/views/1).

- **Board:** a mechanically exported, byte-bound `STATUS.md` snapshot. Counts refer to the selected rows, with full scope and limits. Historical observations are dated separately; Pages does not fetch a mutable ref.
- **SIDE24 explorer:** exact decimal endpoints from pinned public JSON; approximate bar lengths are NON-CERTIFYING. No lifetime curve, random field simulation, or new bound is generated.
- **Inventory browser:** reads the existing `docs/public-math/sources.json` and its fourteen hash-verified shards. No second inventory is created.
- **Contributions:** GitHub issues and incoming-only result PRs. The viewer has no upload endpoint, credentials, write token, or backend.

The original enclosure's `scientific_acceptance: false` is preserved. Later scoped review is quoted separately from the status snapshot; neither is silently substituted for the other. AMEND mathematics remains AMEND.

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

GitHub Pages is configured as **Deploy from a branch → main → /docs**. The entry `docs/index.html` points to `site/`; `docs/.nojekyll` serves static files. The verified destination is [https://d6g8k5htny-coder.github.io/main/site/](https://d6g8k5htny-coder.github.io/main/site/). GitHub's branch publishing supports `/` or `/docs`, so `docs/site` is the app directory, not a selectable publishing root.

See the [deployment and review settings record](../PUBLIC_SHOP_SETUP.md). Pages, the public Project, the four profile pins, and the main/Math- branch rules have installation receipts. A public site does not establish branch protection or mathematical acceptance.

## Claim cards and exhibits

Open [the verification museum](museum.html). Its small `museum.json` display projection is reproduced by `tools/museum_data.py`; it is not a replacement for the 2,138-row public inventory. The generator preserves all eleven bullets in Math-'s “Reviewed scoped results” and the three STATUS AMEND rows verbatim. An EXACT_COUNTEREXAMPLE retains that label; publication of a source does not make its claim accepted. Where a review is a GitHub comment, the displayed digest identifies the pinned index pointer, not mutable comment bytes.

The museum binds Math sources to `d6628da09384728992dcbe6e921cc28ba85aebb0` and main source/packet membership to `71400b94f6cb354a8cf7aba73ffede2138a64efa`. The existing board/inventory pin `a26f744be7597e3f0c0543c34ead23049bbac657` and SIDE24 import pin `9d7b6802424fb4715b31999066aafca8ee2f3cca` remain explicit historical identities. Neither changes the README's Math checkout or query checkout. No source fetch selects an AMEND branch.

Claims use claim/scope, source/review, and replay columns. Missing recorded replay commands are disclosed. The separate engineering strip and CI links do not attest to mathematical acceptance. D5's reviewed fixed-annulus region and its open pin-neighborhood/microdisk work remain separate objects.

- EC-014 illustrates the two-pin frame and six data coordinates. Camera motion is presentation, not a replay of its Jacobian argument. Imported self-labels are not adopted.
- D4/D5 diagrams show only their stated fixed regions, with OPEN complements. Diagram dimensions are schematic, not certified constants.
- P15 illustrates the source's stated finite realized-family example, with no unrestricted prize claim.
- D2 has a **fixture absent** card: no committed lifetime/barcode samples were found in the audited defaults. No synthetic proof data or live Gaussian sampler is supplied.
- The packet list contains only the one package in the pinned main snapshot. Later or unmerged PRs are excluded until a reviewed engineering refresh. Every row says **packet — not STATUS**.

```sh
python3 -B -m unittest discover -s tests -p test_museum_data.py
python3 -B tools/museum_check.py
node --test tests/test_museum_geometry.mjs
```

Existing intake enforcement remains `.github/workflows/public-intake.yml` and `tools/public_intake_check.py`: trusted `pull_request_target` code, no packet execution, incoming-only files, bounded identities, and a documented credential-pattern scan. The scan is not comprehensive. Task claims remain the human protocol in CONTRIBUTING; display inclusion never replaces review.

`museum_check.py` reads the pinned public bytes, checks regeneration, and exercises the actual committed manifest through the display code with the declared HTML containers. It also tests refusal of cross-claim proof, scope and replay substitutions. It is an integration harness, not a browser layout or GPU test. `museum_data.py --check` remains available for export-only reproduction.

The [implementation record](IMPLEMENTATION.md) maps W1–W12, coordination exclusions, and validation boundaries.
