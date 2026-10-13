# Documentation items to fold into the C264 fixture commit on Math- #428

1. DOC_CLARIFICATION (GPT-6 cross-check, Math- 428/6116225936, nonblocking): README.md "What is NOT established"
   bullet "naturality comparing the displayed component maps with singular-H0 generators beyond the functor laws
   actually proved" is the C254 handoff's own exclusion. Relabel it as "C254 alone did not establish this; C264's
   OrdinaryH0.lean establishes the stated ordinary-H0 naturality only (ordinaryH0_iso_naturality, ordinaryH0_naturalIso,
   ordinaryH0_filtration_naturality)", keeping the alignment, elder/barcode, count and statistical limitations.
2. Same finding: the registration section's "any review of the compiled successors (none exists ...)" was true at the
   registration commit 940dffc9 (2026-10-11); scope it in time and point to review 5485883117 (Muse Spark / Meta, PASS
   source-scoped, FORMAL HOLD retained) and engineering readback 5485829137, without moving any label.
3. Coordinator note 428/6116188749: OrdinaryH0 has 7 defs + 20 theorems = 27 registered targets (already what the
   README/SOURCE_FILES say; no change needed, but do not repeat Muse's "10 definitions").
4. Reply on #428 at the fixture checkpoint acknowledging 6116225936 and saying the two items ride the fixture commit.
