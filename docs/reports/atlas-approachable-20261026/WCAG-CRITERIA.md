# Sanitized WCAG2.2 criterion coverage table

Engineering evidence dispositions only. No copied raw UI prose/source-risk excerpts. Postfix successor: original pre-repair table and failure remain retained separately.

| SC | Criterion | Evidence disposition | Scope/limit |
|---|---|---|---|
| 1.1.1 | Non-text Content | PARTIAL_OBSERVATION | Names and text alternatives sampled; complete equivalence not established. |
| 1.2.1 | Audio-only and Video-only (Prerecorded) | NOT_APPLICABLE_SCOPED | No relevant content/action in sampled reader; not site-wide. |
| 1.2.2 | Captions (Prerecorded) | NOT_APPLICABLE_SCOPED | No relevant content/action in sampled reader; not site-wide. |
| 1.2.3 | Audio Description or Media Alternative (Prerecorded) | NOT_APPLICABLE_SCOPED | No relevant content/action in sampled reader; not site-wide. |
| 1.2.4 | Captions (Live) | NOT_APPLICABLE_SCOPED | No relevant content/action in sampled reader; not site-wide. |
| 1.2.5 | Audio Description (Prerecorded) | NOT_APPLICABLE_SCOPED | No relevant content/action in sampled reader; not site-wide. |
| 1.3.1 | Info and Relationships | PARTIAL_OBSERVATION | Semantic structure sampled; generic-label manual items remain. |
| 1.3.2 | Meaningful Sequence | PARTIAL_OBSERVATION | DOM/AX reading order sampled. |
| 1.3.3 | Sensory Characteristics | PARTIAL_OBSERVATION | Named and keyboard instructions sampled. |
| 1.3.4 | Orientation | NOT_TESTED | Not tested in this scoped audit. |
| 1.3.5 | Identify Input Purpose | NOT_APPLICABLE_SCOPED | No relevant content/action in sampled reader; not site-wide. |
| 1.4.1 | Use of Color | PARTIAL_OBSERVATION | Text and pattern alternatives sampled. |
| 1.4.2 | Audio Control | NOT_APPLICABLE_SCOPED | No relevant content/action in sampled reader; not site-wide. |
| 1.4.3 | Contrast (Minimum) | PARTIAL_OBSERVATION_REPAIRED_A11Y01 | Original failure retained; postfix dark9.9395/light13.2187:1, scoped contrast axe clean. Other contrast incomplete cases remain. |
| 1.4.4 | Resize Text | NOT_TESTED | Not tested in this scoped audit. |
| 1.4.5 | Images of Text | NOT_APPLICABLE_SCOPED | No relevant content/action in sampled reader; not site-wide. |
| 1.4.10 | Reflow | PARTIAL_OBSERVATION | 320CSS list/reader; actual browser zoom not established. |
| 1.4.11 | Non-text Contrast | NEEDS_MANUAL_REVIEW | Complete control/graphic contrast not measured. |
| 1.4.12 | Text Spacing | PARTIAL_OBSERVATION | Specified text-spacing override sampled. |
| 1.4.13 | Content on Hover or Focus | NOT_APPLICABLE_SCOPED | No relevant content/action in sampled reader; not site-wide. |
| 2.1.1 | Keyboard | PARTIAL_OBSERVATION | Native disclosure/list/View and real3D access sampled. |
| 2.1.2 | No Keyboard Trap | PARTIAL_OBSERVATION | Canvas/reader/disclosure exit sampled. |
| 2.1.4 | Character Key Shortcuts | PARTIAL_OBSERVATION | Focused-control shortcuts inspected; no site-wide shortcut audit. |
| 2.2.1 | Timing Adjustable | NOT_TESTED | Not tested in this scoped audit. |
| 2.2.2 | Pause, Stop, Hide | PARTIAL_OBSERVATION | Reduced-motion/user-driven rendering sampled only. |
| 2.3.1 | Three Flashes or Below Threshold | NOT_TESTED | Not tested in this scoped audit. |
| 2.4.1 | Bypass Blocks | PARTIAL_OBSERVATION | Skip link and landmarks sampled. |
| 2.4.2 | Page Titled | PARTIAL_OBSERVATION | Document title observed. |
| 2.4.3 | Focus Order | PARTIAL_OBSERVATION | Bounded keyboard/focus trace only. |
| 2.4.4 | Link Purpose (In Context) | PARTIAL_OBSERVATION | Source/navigation link names sampled. |
| 2.4.5 | Multiple Ways | PARTIAL_OBSERVATION | Alternative reader routes sampled; whole-site coverage incomplete. |
| 2.4.6 | Headings and Labels | PARTIAL_OBSERVATION | Headings and labels sampled. |
| 2.4.7 | Focus Visible | PARTIAL_OBSERVATION | Computed focus styles and focused captures sampled. |
| 2.4.11 | Focus Not Obscured (Minimum) | NEEDS_MANUAL_REVIEW | All overlays/scroll/focus states not covered. |
| 2.5.1 | Pointer Gestures | NEEDS_MANUAL_REVIEW | Physical touch/gestures not tested. |
| 2.5.2 | Pointer Cancellation | NEEDS_MANUAL_REVIEW | Source-level cancellation inspection; full runtime not sampled. |
| 2.5.3 | Label in Name | PARTIAL_OBSERVATION | Native name/label relationships sampled. |
| 2.5.4 | Motion Actuation | NOT_APPLICABLE_SCOPED | No relevant content/action in sampled reader; not site-wide. |
| 2.5.7 | Dragging Movements | NEEDS_MANUAL_REVIEW | Complete single-pointer camera alternatives not established. |
| 2.5.8 | Target Size (Minimum) | PARTIAL_OBSERVATION | Footer Equivalent exception verified; not full target-size acceptance. |
| 3.1.1 | Language of Page | PARTIAL_OBSERVATION | Page language observed. |
| 3.1.2 | Language of Parts | NOT_TESTED | Not tested in this scoped audit. |
| 3.2.1 | On Focus | PARTIAL_OBSERVATION | Bounded focus sample. |
| 3.2.2 | On Input | PARTIAL_OBSERVATION | Explicit-control input sample. |
| 3.2.3 | Consistent Navigation | NOT_TESTED | Not tested in this scoped audit. |
| 3.2.4 | Consistent Identification | PARTIAL_OBSERVATION | Atlas-only identification sample. |
| 3.2.6 | Consistent Help | NOT_TESTED | Not tested in this scoped audit. |
| 3.3.1 | Error Identification | NEEDS_MANUAL_REVIEW | One source-refusal state; not a full form/process audit. |
| 3.3.2 | Labels or Instructions | PARTIAL_OBSERVATION | Native field/control instructions sampled. |
| 3.3.3 | Error Suggestion | NEEDS_MANUAL_REVIEW | One refusal-recovery guidance sample. |
| 3.3.4 | Error Prevention (Legal, Financial, Data) | NOT_APPLICABLE_SCOPED | No relevant content/action in sampled reader; not site-wide. |
| 3.3.7 | Redundant Entry | NOT_APPLICABLE_SCOPED | No relevant content/action in sampled reader; not site-wide. |
| 3.3.8 | Accessible Authentication (Minimum) | NOT_APPLICABLE_SCOPED | No relevant content/action in sampled reader; not site-wide. |
| 4.1.2 | Name, Role, Value | NEEDS_MANUAL_REVIEW | DOM/AX names/expanded state sampled; existing actual-AT issues open. |
| 4.1.3 | Status Messages | NEEDS_MANUAL_REVIEW | DOM status structure sampled; actual speech untested; existing AT issues open. |
