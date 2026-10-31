# Historical phase1 design (superseded after Important E1)

# Atlas fallback geometry and accessibility evidence

Readers must be able to continue in Graph or List when optional 3D fails. On
Main `79bdde89a3daf7eaf5d4481961eb87433dd002d5`, a captured-public browser audit
found three overlapping graph-card pairs after WebGL initialization failure.
Each overlap is 41.734375px, and a real mouse click on covered metadata selects
the next claim. The measured card is 205.734375px tall; hidden-stage measurement
has positioned its next sibling only 164px below it.

Use the existing controller's layout-only `onChange({write:false})` callback
after the spatial shell restores Graph/List visibility. It remeasures cards
without writing URL/history or adopting a source. Preserve selection, filters,
disclosures, graph relationships, source refusal and all evidence identities.
The only substantive production hunk is `atlas-spatial-shell.mjs` `fail()`.

Alternatives considered: changing the graph layout algorithm would spread this
small recovery fix into shared controller code; changing hidden-stage CSS would
alter accessibility and layout semantics. Both are unnecessary because the
existing layout-only callback already provides the needed behavior.

Verify actual card geometry and mouse selection after WebGL unavailability,
renderer-module failure and real successful rendering followed by context loss.
Exercise desktop/phone widths, both themes, forced colors, Graph/List, history,
source pins and refused bookmarks. A visible fallback alone is insufficient.

Retain the separate measured contrast adjudication and actual version-specific
Orca records. Preserve raw axe incompletes and failed/inconclusive protocols;
never replace them with a conformance or human-audition claim. No new dependency,
formal mapping, scientific status, workflow or security setting is introduced.

The owner delegated priority selection, design and execution. These are the
author's bounded implementation choices; they are not owner-specified details.
Current finite task: main#388/6483670651, operative scope #388/6483977068.


## Final E1/current-base amendment

The original callback was a full render, not layout-only. Nonauthor review
correctly held 2b0/native7dd because it detached a focused metadata summary.
The final design, prospectively scoped in388/6484965413, changes two semantic
production hunks: atlas.mjs accepts graphOnly and reflows only current visible
Graph IDs/edges; spatial fail requests write:false,graphOnly:true after display.
Inspector/List DOM, focus, disclosures, history, sources and Root682 logic stay
intact. A thirteenth real context-loss focus/nextTab test is RED on b43 and GREEN
on product2d64/baseeb79. All13 focused,16 spatial,10 refusal cases and13 release
groups pass; complete two-pass generation is equal. Current source review,
native gates and actual deployment are separate delivery steps. Preserve original
phase1 records and all GitHub preimages before native refresh.
