# Library temporal comparison implementation plan

**Goal:** Let readers inspect what changed between exact retained versions without confusing pin movement with content or mathematical change.
**Spec:** ../specs/2026-10-25-library-temporal-compare-design.md
**Architecture/tech:** Separate nativeHTML/CSS/module, existing hash/schema contracts, strict versioned comparison model, Node and real Playwright.
**Constraints:** No existing loader/producer/graph/Atlas/scientific/workflow edits; use actual producer semantics; preserve sources; no Drive deletion. Engineering same-family separate reviewer allowed, scientific requires anotherfamily.

- [ ] Read authoritative current-model/history schema contracts; record exact APIs and bounds. Write failing tests before model code.
- [ ] Implement bounded metadata/registry/archive acquisition and explicitschema1/schema2 validation; literalnode/edgemultiset/sourcepin/contentdiff. Negative failures leave no substitute comparison.
- [ ] Add standalone source selectors, shareURL/nativehistory, honest scope/results/errors and bounded result pagination, with static fallback and existingbrand.
- [ ] Actual real archive and browser checks, mobile/light/dark/keyboard/noJS/refusal; visually inspect screenshots. No fabricated data or certification.
- [ ] Freeze owncandidate; complete canonical asset release and applicablepublicshop checks. Separate engineering review, protected integration, archive and observedlive replay.

**Review focus:** Untrusted manifest paths/bounds; mismatched hash/metadata; schema2 missing inventory mistaken as deletion; JS boolean/number equality; duplicate edge multiplicity; stale late async result; sourcepin changes conflated withcontent.
**Ownership:** Root coordinates, separate implementation worker owns newfiles only. Library reading-list/Atlas/Orca source stages remain intact. Evidence outside source under /workspace/research/library-temporal-compare-20261025.
