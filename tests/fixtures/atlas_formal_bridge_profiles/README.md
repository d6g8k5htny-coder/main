# Frozen manifest-profile inputs

These are literal data for offline source-bridge tests, never imported or
executed. Manifest8228 is from93869215a82da60f014093e2ac28503865ed0aa3;
manifesta22f and its two documentation files are from
a30e336982ff4a9400182961ae1f8a241ceff927, unchanged at55961773.
Their source paths are formal/manifest.json, formal/README.md and
formal/REVIEW_LANE.md. The8228documentation bytes are already retained in
tests/fixtures/atlas_formal_build_3b39 and are reused without alteration.

SHA256:8228manifest1a09020f0706abd506824f54fca76df9daa533749350b9ab6b818e0bbe9ed744;
a22fmanifest38eefbfdd84658cd0f6e18bd7d714458a7522b1021f73b1075e7972a63b9bcb0;
README137e7c7ed44d732042d803eb4a0ee75f9ddc6475950a2bb6226a7711a44be14b;
REVIEW_LANE54d32104aa012e917ef40b73fa3a40228d023249b339cf30fe4bdb7dcbfed414.

Only exact known profiles are accepted. Fixture custody is not new kernel
execution, semantic alignment or scientific acceptance.

The two `formal_gate_check.py.txt` and `test_formal_gate.py.txt` inputs preserve
control bytes from Main `4ef1867175dbe1072a54a02e95f80c4e2c70a4ff`, matching both
8228/a22f manifests exactly. Tests copy these as inert source data instead of
assuming today's validator bytes still match a historical manifest. They are
never imported or executed. `historical_source_identity` in
`tools/atlas_formal_evidence_build.py` recognizes only the three fixed historical
manifest digests and checks all their bound file bytes, refusing unknown profiles,
symlinks and unbound package files. This is identity reconstruction, not current
source-gate certification or renewed kernel verification. Run its tests with
`python -B -S -m unittest tests.test_atlas_formal_evidence_build tests.test_atlas_formal_source_bridge -v`
(and repeat with `-O`); existing hash-mutation cases are retained.
