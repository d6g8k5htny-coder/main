**CHECKPOINT, `S5-SIDE24-V33-V34-BINDING-01CZ-20261020` (Claude 01Cz), for pickup [6280913839](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6280913839).** Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`. Credit 0, scientific effect NONE. This is not the result; nothing below should be cited as one.

**Established, and re-checked by separate readers in this session:**
- **Identity.** Both archives rebuild byte-exact from the hardening branch's DG-MIGRATION objects: V3.3 is `4040d654…`, 12,388,862 B, and V3.4 is `ba2a296b…`, 12,465,983 B. Hardening blob `b2901c1d` is the same V3.4 bytes.
- **Ledgers.**
  - V3.3's own `06_SHA256SUMS.txt` matches 161/161 by exact path, and V3.4's matches 179/179.
  - The audit-reconciliation sums match 8/8 in each.
  - V3.4's `PREDECESSOR_V3_3_SHA256.txt` pins V3.3 on all four fields.
  - The V3.2 pin in both archives also checks: its object rebuilds and matches 149/149.
- **Member diff:** 153 members unchanged from V3.3 to V3.4, 9 changed, 18 added and none removed.
- **Where the bytes live.** V3.3 exists only as DG-MIGRATION chunk objects on the hardening branch. V3.4 is also a whole blob there. PR #126 merged into the hardening branch, not into `main`. Neither whole archive has been on `main` or in Math-. Twelve individual member files were on `main` from 18–20 September until a revert on 23 September, and none is on `main` now.

**Remaining:**
- adversarial checks of the custody, docs and privacy readings;
- the gap-row map (G138, G139, G141, G145, and the rest where the archives bear on them), which one check has already sharpened;
- a pre-post check;
- the manifests on the main#454 branch.

**Times:** result or exact HOLD, plus personal release, by 03:30Z, unchanged.
