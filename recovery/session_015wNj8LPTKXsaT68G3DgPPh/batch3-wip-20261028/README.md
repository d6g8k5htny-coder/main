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
