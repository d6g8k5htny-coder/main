# Public research shop

This static viewer makes the existing public work easier to read. It does not own scientific status.

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

The interaction harness tests loading, searching, exact decimal selection and commit currency; it is not a visual browser test. Browser preview from the editing environment was unavailable because its browser could not reach the local server.

Reproduce the exports with `python3 -B tools/public_shop_data.py --check`. The exporter uses fixed source identities, refuses changed bytes or unfamiliar tables, and preserves verbatim cells. Updating the snapshot requires an explicit source-bound engineering PR; this viewer never decides a review disposition.

## Notebook

The [SIDE24 notebook](../notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb) exposes the same immutable input and exact values, with an optional approximate chart. See [notebook instructions](../notebooks/README.md).

## Publish on main only

Use GitHub Pages **Deploy from a branch → main → /docs**. The entry `docs/index.html` points to `site/`; `docs/.nojekyll` serves static files. GitHub's branch publishing supports `/` or `/docs`, so `docs/site` is the app directory, not a selectable publishing root.

See [settings and task-board setup](../PUBLIC_SHOP_SETUP.md). Repository files do not by themselves enable Pages, pin profile repositories, install branch protection, or create a Project. Their activation must be verified in GitHub settings.
