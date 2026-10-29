# Batch 3 work in progress (C8), 28 October 2026

Stage `C8-WRAPPERS-REPAIR-C8-20261026`, batch 3: `c8-theorem-a-constants-planar` and
`c8-torus-vanishing-remainder` (re-pickup Math-#322/6464591018). A copy kept here so a
container restart cannot erase it again. Nothing here is reviewed or delivered.

- `batch3_wip.patch`: the batch-3 change against Math- `8dca89cf` (Math-#520's head),
  rebuilt from the session record by `scripts/rebuild_b3.py`. The rebuilt
  `tools/check_mode_replay.py` has git blob `fd4385de39ebc6142ccd16e0ca2c08d585c2155f`, the
  blob the 27 October torus replay ran. The contract built from `runA` is 5,677 B with
  SHA-256 `be7415ca2ca6d2df0bca4eab12026fffd985f8a79a12863e4cfa7a8e0bff52dd` (families digest
  `30f3b2f43f18082b0b60509747e1525fb84afb1aa8e0db0ac8b9b1818697cd1a`), byte-identical to the
  27 October run-1 contract.
- `runA/`: characterization run A of the unchanged checkers on Math- `main` `4a0d51bb`
  (packet trees `95d1459b` and `04fc5584`), 18 children.
- `scripts/`: the characterization, contract-merge, rebuild and replay scripts.

Scientific effect NONE. Credit 0.

## 29 October: the finished commit (not pushed)

- `batch3_257f9f7d.patch` (`git format-patch`): commit `257f9f7d32842506079b4a960c13397ec3c82d11`,
  tree `a3dfcb6ba7721a2ef151adce6acc674c4000b2ab`, parent Math- `8dca89cf` (Math-#520's head).
  Apply with `git am` on `8dca89cf`. Six files: the two workflows, the helper, the reference
  file, the test file and `reviews/c8_check_mode_wrappers_batch3_20261027/SCOPE.md`.
- `runB/`: characterization run B, byte-identical to `runA` on all 18 children; the reference
  file built from runs A and B is SHA-256 `be7415ca…`.
- Tests on that tree (CPython 3.11.17): 15/15 in both modes for each new family and for all
  five. RED against `8dca89cf`: 29 (theorem-A) and 44 (torus) false acceptances per mode.
- Not pushed to Math-: the push of branch `claude/determined-bardeen-mrqy4y-c8-wrappers-b3` was
  stopped by the session's permission classifier at about 17:00Z, and is left for Dylan to decide.
