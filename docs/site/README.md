# Public research shop

This static viewer makes the existing public work easier to read. It does not own scientific status.

**Live:** [Open the public research shop](https://d6g8k5htny-coder.github.io/main/site/). Coordinate a bounded contribution on the [Public contributions board](https://github.com/users/d6g8k5htny-coder/projects/1/views/1).

- **Board:** a mechanically exported, byte-bound `STATUS.md` snapshot. Counts refer to the selected rows, with full scope and limits. API observations are dated separately.
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

The interaction harness tests loading, searching, exact decimal selection and commit currency; it is not a visual browser test. Local-server preview was unavailable in the editing environment. After deployment on 2026-09-26, the live Pages site was checked in a real browser: all 2,138 original catalog records loaded and the exact SIDE24 source bytes verified.

Reproduce the exports with `python3 -B tools/public_shop_data.py --check`. The exporter uses fixed source identities, refuses changed bytes or unfamiliar tables, and preserves verbatim cells. Updating the snapshot requires an explicit source-bound engineering PR; this viewer never decides a review disposition.

## Notebook

The [SIDE24 notebook](../notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb) exposes the same immutable input and exact values, with an optional approximate chart. See [notebook instructions](../notebooks/README.md).

## Publication on main only

GitHub Pages is configured as **Deploy from a branch → main → /docs**. The entry `docs/index.html` points to `site/`; `docs/.nojekyll` serves static files. The verified destination is [https://d6g8k5htny-coder.github.io/main/site/](https://d6g8k5htny-coder.github.io/main/site/). GitHub's branch publishing supports `/` or `/docs`, so `docs/site` is the app directory, not a selectable publishing root.

See the [deployment and review settings record](../PUBLIC_SHOP_SETUP.md). Pages, the public Project, the four profile pins, and Math-'s proof-vault branch rules have been activated. Main's tightened review settings remain intended configuration pending separate verification; a public site does not establish branch protection or mathematical acceptance.
