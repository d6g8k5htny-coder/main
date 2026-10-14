# Public-site presentation

Follow the [current workflow](../../governance/OP-WORKFLOW-20260930.md).
The public home and Explore lead with understandable mathematics and useful
interactions. Keep contributor operations in the workspace and link detailed
source records where they help a reader investigate the material.

For visual changes, read [the shared brand guide](../BRAND_STYLE.md). Use `brand.css` and `brand/tokens.json`; do not add another theme framework or external font/analytics dependency. Keep the approved navy, field-blue and restrained-gold identity with its complementary light view.

The source workspace is a read-only view of its records. Preserve source-bound
content, JSON hashes, exact numeric strings, refusal behavior and distinct class
labels/border patterns. Educational models may be interactive when their scope
is explicit. Presentation changes do not change scientific acceptance; never
regenerate historical proof evidence as a cosmetic operation.

After any change under `docs/site` or `docs/public-math` (HTML, CSS, modules, JSON or Markdown, including this file), stage new or deleted files, then run `python3 -B tools/site_asset_release.py` and `python3 -B tools/site_asset_release.py --check` from the repository root, and commit every rewritten file. Then run the required public-shop steps in [the site guide](README.md#verify-the-reader-experience); `node --test tests/test_public_shop_frontend.mjs` includes the brand controls. Check mobile/desktop light/dark layouts and keyboard focus separately. A local screenshot is not a live deployed-source replay or a full accessibility certification.

The owner's current request authorizes these presentation and workflow changes.
Keep the historical stop record unchanged; the stop itself is no longer in effect
([owner decision](../../governance/OWNER_DECISION_20261005_CURSOR.md)).
