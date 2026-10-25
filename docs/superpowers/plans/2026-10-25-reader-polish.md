# Reader navigation polish implementation plan

**Goal:** Make the existing public UI easier to scan and navigate, preserving its mathematics and source-bound behavior.
**Spec:** Dylan's current UI-polish instruction; [UI map](../../UI_MAP.md) records the existing structure, not this task list.
**Architecture:** Keep the existing static HTML, local brand stylesheet and native links. Change only the shared navigation presentation and footer grouping. Atlas spatial/controller work remains in its separate lane.
**Tech:** HTML, CSS, Python Playwright, existing Node/Python source checks.

## Constraints and decisions

Keep every existing footer href and link label, scientific statement/classification/source pin, current-source generated block and historical record. No new dependencies, theme or script. Use the existing navy/blue/gold tokens and visible keyboard focus. Same-family separate engineering review is allowed; this task supplies no scientific review.

Approach chosen by the implementer under the owner's existing authorization: make a small shared-shell pass. A new navigation framework would add runtime behavior; CSS-only footer wrapping would retain the dense link sentences. Nested static lists let the existing links form clear groups without either cost.

## Separate edit checklist

- [x] Record the source/UI map and fresh mobile/desktop screenshots. Keep action tracking here.
- [x] Restore at least 44px primary/tool navigation targets, and balance the five primary links into three/two rows below 360px.
- [x] Group the footer's existing Learn, Research and Records/tools links into readable nested lists: three desktop columns, two compact phone columns with the longer group below. Preserve link order, text and hrefs; retain compact print links.
- [x] Re-key changed assets/importers; run required local public-shop checks and meaningful focused Playwright layout/navigation/keyboard controls. Inspect before/after light/dark screenshots, narrow reflow and print visibility; verify unchanged scientific/source content and footer destinations.
- [ ] Obtain a separate engineering review, address material findings, and deliver source + evidence with unresolved limitations. Preserve Drive archives without deleting or overwriting existing files.

## Review focus

At 320px and with increased text spacing, no header/footer link is clipped or offscreen. Keyboard focus remains visible and natural link order is unchanged. Supporting pages and static 404 navigation still work. Footer links remain available with JavaScript disabled and in print. A nested list does not alter a scientific source, classification or refusal state.
