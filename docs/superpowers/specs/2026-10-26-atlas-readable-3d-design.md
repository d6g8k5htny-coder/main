# Read complete claim identifiers in the 3D Atlas

The next bounded reader release makes visible 3D labels readable without
changing mathematical records, graph positions, camera controls or selection.
The existing renderer ellipsizes identifiers and compares fixed-height boxes;
its labels can conceal the distinguishing end of a claim identifier.

Use complete literal identifiers in wrapping native buttons. Measure their
actual dimensions in one read phase, then place nonoverlapping rectangles
within the viewport. Prefer the selected claim, then frontmost nodes, with a
stable identifier tie break. Try positions immediately above, below, left and
right of the node; preserve a gap around each label's own node anchor. Cull
labels when space runs out. The complete accessible node/relationship list,
source inspector and actual node selection remain available. This does not
promise that every identifier can fit in the diagram simultaneously, or that a
selected label cannot cover another projected node in a dense view.

Alternatives considered: shorten identifiers further (loses the identity readers
need); add a separate label-control mode or force layout (adds state or changes
spatial navigation); measure and wrap the current labels (chosen by the engineer
under the user's implementation/Continue authorization). No new dependency.

Keep all label text inert, preserve lazy renderer loading, reduced motion,
resource cleanup and context-loss fallback. Do not add controller state, URL
parameters, new status authorities, inferred edges or formal mappings. The
separately owned #637 amendment still requires its qualified review.

Validate actual WebGL rendering and uncovered canvas picking, complete selected
IDs, measured bounds/intersections, phone/desktop light/dark, reset/focus/history,
keyboard and the existing failure/refusal/static fallbacks. Compare the same
frozen graph before/after, including render and interaction costs. Report the
number of labels that fit rather than presenting culling as coverage loss.
Correct the previous report's prose byte figure to its existing raw15,359B value;
preserve the original bytes first. No mathematical acceptance follows.

Own scope and custody:388/6406487067. Original289-file preimages are privately
archived/readback verified before product edits. Separate engineering review,
separate integration, current required checks and deployed-source replay remain
release requirements. Preserve all other owners and worktrees.
