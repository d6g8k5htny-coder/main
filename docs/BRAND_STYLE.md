# Universal Law visual identity

Scientific effect: **NONE**. Public presentation inspired by Dylan Roy's approved navy, electric-blue and gold profile seal. This guide is not a source, review, or scientific-status registry.

## Palette and typography

`site/brand/tokens.json` records the colors actually used in `site/brand.css`. Deep navy is the foundation; field blue identifies navigation; gold is a restrained identity accent. Warm ivory text supports the dark view. The light view uses white/blue-gray surfaces, deep navy text and darker gold. Georgia is used for editorial headings; the system sans-serif stack is used for reading and controls. No remote font, analytics, animation, or visual-effect dependency is added.

The seal remains the detailed research illustration. `site/brand/mark.svg` and `banner.svg` are small decorative companion assets; their contours are not simulation data, persistence calculations, or scientific evidence. They contain no equations, theorem counts or acceptance badges. Use the approved profile picture unchanged.

## Presentation scale (not a status register)

One type ladder, recorded so later passes reuse it rather than re-derive it: h1 clamp(38–68px) Georgia 400 · h2 clamp(28–42px) Georgia 400 · Home and Research card titles 26px Georgia · h3 21px sans-serif 600 · h4 16px · small-caps labels (brand line, eyebrows, card and result kickers, the tools kicker, breadcrumbs, the dated cut-list label) 12px with .16em tracking, uppercased by CSS so source text may be sentence case · body 16px · card prose 15px · notes 14px · code, hashes and exact decimals never below 13px, except in quoted source identities and claim-column blocks, which are 12px. Table headers and a few record labels (the museum claim-column headings and engineering-strip labels, the Dependencies metadata and path labels) still use 11px with their own tracking until a later pass brings them to this scale. Running prose is held to a 78ch measure. Radii form one family, s 6px · m 8px · l 18px · pill, declared as `--ul-radius-*` layout variables in `brand.css`; `brand/tokens.json` records colours only. Scope notes (`.boundary`, `.latest-boundary`) carry the neutral edge colour; amber stays reserved for AMEND/open objects and absent fixtures. Dark mode intentionally uses gold for the primary button, the eyebrows and the focus ring: an identity decision, not a status colour, recorded here so it is not relitigated page by page. Print keeps the evidence: navigation, tools, buttons and empty search fields are hidden, and a filled search field prints its query beside the result count it produced; tables, code, hashes and source links print in dark text on white with the light-scheme state colours in both colour schemes; a collapsed disclosure prints “collapsed; contents not printed” after its summary instead of silently vanishing.

## Meaning must survive styling

ACCEPT-scoped retains a solid green/teal boundary; AMEND/open retains a dashed amber boundary; engineering-only retains a double boundary; illustrations retain a dotted boundary. All labels remain literal. Gold is a brand accent, not an acceptance color. Exact hashes, scoped source text, refusal messages and open obligations must remain legible. Charts keep their original data and geometry; only the coefficient chart's display palette is restyled. Scientific modules, manifests and review/source data are not changed.

## Accessibility and display

Core text/link/state colors are tested at 4.5:1 against all three principal surfaces in both color schemes. Control boundaries and focus rings are tested at 3:1. Keyboard skip links, explicit focus outlines, fluid grids, mobile navigation, reduced-motion, forced-color and print rules are included. These are bounded checks, not a claim of full WCAG certification. Technical identities may use accessible disclosures; scope labels and failure states remain visible. Do not remove evidence merely to make mobile screenshots fit.

GitHub's own navigation chrome and light/dark mode belong to each visitor's appearance preferences. Repository files cannot enforce this site's CSS across github.com. README images use self-contained SVG artwork on a stable navy background and meaningful alt text; do not inject CSS or tracking widgets into Markdown.

## Reuse and maintain

- Website: load `brand.css` after the existing page styles, and retain CSP restrictions.
- README: use a local copy of `banner.svg` or an exact-commit raw GitHub link, with alt text. Keep the repo's real heading, source links and scope below it.
- Never recolor preserved historical evidence, raster plot sources, negative-control artifacts or proof PDFs as a branding operation.
- Never add branding that claims independent review, completed theorem closure, broad physical laws or live agent counts.
- Keep repository names, permissions and required workflows unchanged in a branding pass.
- The September 27 stop is no longer in effect ([owner decision](../governance/OWNER_DECISION_20261005_CURSOR.md)); this guide does not itself start any agent.

## Checks

Run `node --test tests/test_brand_identity.mjs` and the existing public-shop/museum tests. A successful palette test is not a rendered-browser audit; record the tested browser, viewport, light/dark state and data availability separately. Network refusal screens must remain honest and readable.

Reference standards: W3C WCAG 2.2 contrast minimum and focus criteria; GitHub Docs, Managing your theme settings. No conformity beyond the observed tests is implied.
