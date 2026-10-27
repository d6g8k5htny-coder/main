# Current-source result continuity

The Library should let a reader inspect source evidence, reveal more matching
files and continue reading without closing that evidence or losing keyboard
position. This is a presentation repair to the existing admitted-source reader.

## Evidence and choice

On Main `c1678a556ab487954abbdc2456686bb64ddc7114`, the included `proof`
query has 224 matches. A real sandboxed Chromium probe observed that expanding
50 to 100 results removed the first open acquisition disclosure, and expanding
200 to 224 hid the focused button and left focus on BODY. The shared renderer
uses `replaceChildren` for both a changed query and a larger result window.

Three approaches were considered: rebuilding rows and restoring disclosure
state by identity; a general keyed renderer; and appending the next batch for
an unchanged query. Choose the last: the accepted snapshot is immutable within
this load, ordering already comes from `filterSources`, and only explicit
Show more needs continuity. A new rendering framework would add unnecessary
state and touch the separately owned loader contract.

## Required behavior

- Keep the existing 50-file batch size, filtering, source ordering, counts,
  literal source text, pinned URLs and acquisition annotations.
- For Show more on the same query and admitted observation, retain all existing
  row objects, open acquisition details and source-save controls, then append
  only the newly revealed rows. Never duplicate a row.
- If Show more owns focus when activated, focus the first newly revealed
  source link. Apply this on each batch, including the terminal partial batch.
  If that row has no link, focus the row with a programmatic `tabindex=-1`.
  If another control owns focus, do not move it.
- Input changes, Clear search and history restoration replace results and reset
  the 50-file window as before. A different query must never append to an old
  query's rows, even if changed programmatically before Show more.
- Snapshot refusal keeps search/save controls disabled. Static and no-JavaScript
  source routes remain available. No new dependencies, schema or storage.

## Boundaries and delivery

Change only the source-results renderer/Show-more handler in `current.mjs`.
Preserve fetch/admission/timeout/acquisition/model/importer/registry contracts,
mathematical records, source payloads, historical evidence and scientific status.
Reconcile actual #501 renderer hunks before product edits; its wider publication
work remains owned there. Asset URL keys are derived by the existing release
tool, not manually chosen. Preserve other agents' actual changes on integration.

Use real-browser RED/GREEN checks, related Library/reading-desk checks and the
required site checks. Record actual browser, widths/themes, source identities and
limits; DOM checks are not spoken assistive-technology or mathematical review.
User standing delegation authorizes execution without another design approval.
Same-family separate engineering review is allowed under the 25 October owner
decision; scientific effect NONE and organizational-independence credit 0.
