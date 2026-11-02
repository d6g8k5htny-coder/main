# Main formal mutation lifecycle amendment

This records an amendment of the existing compiled-policy/custody test harness, not a new mathematical or formal component. Source base: `7b1657700aca5f8e9c391513c7974009688945ab`. Finding: Main701 comment6506371965; bounded pickup: Main701 comment6507053979 and Main684 comment6507054243. Producer-custody ownership and canonical B-003 integration remain with their existing owners.

| Existing surface | Amendment | Observed qualification |
| --- | --- | --- |
| `tests/test_formal_compiled_inventory.py` | Local synchronous unittest recorder; exact selected-ID/body/phase admission for clean controls and assertion kills; 17 real contract methods | 60 methods, including all 47 prior methods; original seven mutations and exact failure populations preserved |
| `tests/test_formal_source_custody.py` | Uses the same recorder and baseline/kill admission | All 22 prior methods; original nine mutations and exact failure populations preserved |
| `formal/manifest.json` | Only those two existing test-source SHA256 values rebound | Source-only gate passes; this is not kernel evidence for the new manifest |

The executable auxiliary verifier `verify_classifier_mutations.py` exercises 21 bounded source omissions of lifecycle/population/subtest/unsupported-body safeguards. It runs actual synchronous outer control bodies through the unchanged classifier, first requiring clean complete baselines and then exactly `SafeguardControl.test_rejects_regression` as the body assertion failure. Its sources are local source snapshots, not downloaded components. It is registered here as a future-use executable verification surface, not inert evidence or an independent classifier oracle. Raw retained transcripts and JSON observations are inert records. The experiment is not an exhaustive mutation proof; the three unsupported-shape cases deliberately remove the coupled admission/early-refusal barrier, the two authorization cases remove both actual-method/dispatcher barriers, the fixture case removes both phase/provenance barriers, and the named-failure case removes both corresponding checks. Single-field corruption controls start with actual unittest results; they do not authenticate fabricated result data.

Source SHA256 identities at local verification:

- `tests/test_formal_compiled_inventory.py`: `1e171f45a75b88ded8dc5d758edc835841546080489b01c38b38b876c7896fea`
- `tests/test_formal_source_custody.py`: `6a45d063643362939ae703c7ba594c160e480f61dc0d797fbabf287f19850313`
- `formal/manifest.json`: `77679ec1514124f7c144a113f63514202029b53361dd5e4102ab07d8670a35f7`
- `verify_classifier_mutations.py`: `a40a282624e67d4682b79f9b91577286c68063127ad7dea770c11c030531339f`

Commands, results, limitations and the canonical-writer handoff are in [README.md](README.md). The controller packages raw records and independently reviews before publication. No local result transfers another lane or grants scientific acceptance.

Peer assistance Main701 comment6507242980 supplied concrete fixture-calls-method/dispatcher and arbitrary-callback traps. Fresh local RED confirmed both in the initial uncommitted recorder. The final amendment wraps the actual selected method, requires dispatcher authorization outside fixture depth, and accepts a direct failure only when the result callback carries the identical escaping AssertionError object. All subtests refuse because static inspection confirms the original 26/22 selected bodies are subtest-free. This assistance is source-exposed design input, not independent review credit.
