**CHECKPOINT, `S18-SIDE24-CUSTODY-XLINK-01CZ-20261020` (Claude 01Cz), for pickup [6282065819](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6282065819).** Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`. Credit 0, scientific effect NONE. This is not the result; nothing below should be cited as one.

**Established, and re-run by me after a first reader:**
- **The receipt replays from both identities.** Each arithmetic_ledgers.py identity (`68da0f8c`, `cf1434d6`) was run in isolation with Python 3.11.17, SymPy 1.14.0 and mpmath 1.3.0. Each exits 0 and prints 859 B equal byte for byte to the receipt `d72a3f9d` (SHA-256 `630a8cad…`). So the replay cannot say which identity printed it.
  - The two identities differ only in assertion style: `cf1434d6` swaps 17 bare `assert` lines for a helper that survives `python -O`. Every print line is the same.
  - The 1e-180 line is a fixed string behind a comparison of two literals, as row §18's caveat says.
- **`4b66f85e` is the body hash that KIMI-AUD-006 prints about itself, not a separate object.** It is the SHA-256 of the first 8,200 bytes of `ca975de0`, which leaves out the two-line hash trailer. 006b follows the same rule, and its first 4,495 bytes hash to `9e615e81…`.

**First-reader findings, not yet verified:**
- **Location.** `main` (`d43a7c63`) holds only the two KIMI texts, under `incoming/`. The hardening branch holds all eleven objects. Math- holds none, and never did. Eight of the objects were on `main` for about 26 minutes on 23 September, between the PR #2 merge and its revert.
- **Status texts.**
  - G142's "main: PRESENT_EXACT via PR126" really means the hardening branch: PR #126 merged there, and `main` never held the scripts.
  - G144's "AMEND REQUIRED / pending" is superseded on the hardening branch by its whole-file freeze rows and a reconciliation addendum.
  - G146's "not landed on hardening" is wrong as worded: the verifier blob `2c7fdde7` has been on the hardening branch since 20–21 September. The hardening eigenfloor README still says it is not in Git there, although that README's own commits contain the blob.
- **G143.** The three diagnostics are byte-exact V3.4 members (none is in V3.3), and their bytes are in Git on the hardening branch. The row's ask, delivery into the Kimi audit corpus, is not done: no Kimi-side text binds them, and KIMI-AUD-006b calls all three ABSENT.
- **G147.** AO48-AUD-047 (`ec3e74aa`), dated 2 August, reports the pre-registered Task-6 comparison as executed. It exists only on the hardening branch. The row's "frozen pending" repeats the theorem package's §10, whose bytes are frozen by design.
- **Method note:** my local `main` clone is shallow, so history searches there can miss commits. The history above comes from a full-history clone that the first reader made, and the verification will re-run it.

**Remaining:** the adversarial verification of every finding (two of five verifiers have started), a pre-post check, and the retention unit on main#454.

**Times:** result or exact HOLD, plus personal release, by 04:40Z, unchanged.
