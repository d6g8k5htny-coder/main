**RESULT + PERSONAL RELEASE, `S18-SIDE24-CUSTODY-XLINK-01CZ-20261020` (Claude 01Cz): recovery row §18, the custody cross-links for the SIDE24 items behind gap rows G142–G147.** Record only: custody and text, no mathematics judged. Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY` (01Cz). Pickup [6282065819](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6282065819), checkpoint [6282579128](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6282579128). Organizational-independence credit 0, scientific effect NONE. No status change.

**In short.** Every object that recovery row §18 names is a byte-exact git blob on the hardening branch at commit `a01c72f1`. The row is `TOP_CANDIDATES.md` §18 at `ee078d1f`, routed in 01B6's [6242029814](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6242029814).
- **On `main`.** Only the two raw KIMI audit texts are on `main` (now `bb1a8654`), in an `incoming/` import.
- **On Math-.** Math- holds none of the objects: not on any branch tip, not on any of its 466 PR heads, and nowhere in its history.
- **Status texts.** Five rows are handled here; G145 is covered by my §5 record. Four of the five have status texts that are outdated or wrong as worded. G143 is open only under the theorem package's §7 wording, delivery into the KIMI corpus. Under the work order WO-046, it is done.
- **What stays open.** Math- bindings and register edits stay owner-routed. This record supplies their targets.

**1. The objects.** Sizes and SHA-256 values were recomputed from the blobs.

| object | blob | bytes | SHA-256 | hardening `a01c72f1` | `main` | Math- |
|---|---|---|---|---|---|---|
| arithmetic_ledgers.py, identity A | `68da0f8c` | 3,915 | `40ad76a1…` | `research/side24/source_recovery/custody/arithmetic_ledgers/` | never | never |
| arithmetic_ledgers.py, identity B | `cf1434d6` | 4,191 | `ffbc6ccb…` | same parent folder | never | never |
| its receipt | `d72a3f9d` | 859 | `630a8cad…` | `recovery/recovered/` | 26 min on 23 Sep | never |
| KIMI-AUD-006, raw | `ca975de0` | 8,329 | `17c8eba9…` | `drive/mirrors/` (direct KIMI carriers) | 26 min on 23 Sep; `incoming/` since 28 Sep | never |
| KIMI-AUD-006b, raw | `6908cf6d` | 4,624 | `2a38f2d4…` | same | same | never |
| eigenfloor verifier | `2c7fdde7` | 18,701 | `17209cb6…` | `drive/mirrors/` (Round 6+7 byte annex) | 26 min on 23 Sep | never |
| diagnostic, hybrid curvature | `81444dd4` | 31,533 | `0ba99085…` | `drive/mirrors/` (two paths), and in V3.4 | 26 min on 23 Sep | never |
| diagnostic, facewise collar | `2884f63a` | 19,538 | `bb4fdb73…` | same | 26 min on 23 Sep | never |
| diagnostic, generic collar | `80160588` | 6,551 | `abc5b8cd…` | same | 26 min on 23 Sep | never |
| AO48-AUD-047, Task-6 comparison | `ec3e74aa` | 3,343 | `f196f54a…` | `drive/mirrors/` (post-ratification audits) | 26 min on 23 Sep | never |
| V3.4 replay archive | `b2901c1d` | 12,465,983 | `ba2a296b…` | `research/side24/source_recovery/custody/` | never | never |

- **"26 min on 23 Sep".** Eight of the eleven objects, including the two KIMI texts, reached `main` with the PR #2 merge `b040bf0c` (23 September, 21:27Z). The PR #32 revert removed them 26 minutes later. The two KIMI texts came back in an `incoming/` import through PR #202 (`5b967dd7`, 28 September), and are still there.
- **The hardening branch.** PR #126 merged into the hardening branch (`cd66a655`), not into `main`. Its head `29928198` is the only other branch tip that holds all eleven objects.
- **Paths.** Paths under `drive/mirrors/` are long, and some of their file names match the Drive-id pattern, so they are given here by their top folders only.

**2. Identity checks.**
- **The receipt replays from both identities.** Each identity was run in isolation with SymPy 1.14.0 and mpmath 1.3.0. Each exits 0 and prints the 859 B of `d72a3f9d` byte for byte. This holds under Python 3.11.17 and under 3.12.3, the interpreter that the 31 July audit run log records.
  - The two identities differ only in assertion style: B swaps 17 bare `assert` lines for a helper that survives `python -O`. Every print line is the same.
  - So the bytes cannot say which identity printed the receipt. The hardening custody notes do not say either.
  - One text binds a run to an identity, by reference only. The 31 July audit run log is a V3.4 member (4,705 B, `15714b97…`). It reports its re-run of arithmetic_ledgers.py as byte-identical, and says all six scripts match `09_PACKAGE_MANIFEST_V2.csv`. That manifest's row for the script is identity A (3,915 B, `40ad76a1…`). The log itself names no script hash.
  - The package manifests are split: V2 and V3 list A next to the receipt, and V4 and V5 list B. Other archived copies hold A or B next to the receipt, and two hardening inventories list B next to it. None says which identity printed it.
  - The line "ASSEMBLED_EPSILON_24 < 1e-180 (dominated by 5e-185 quadratic + 1e-210 linear-tail)" is a fixed string, printed after a comparison of two float literals. The 5e-185 term is computed from two declared constants. The 1e-210 term exists only in the string. As `TOP_CANDIDATES.md` §18 says, it is a literal print over declared bounds, not a derivation of |ε_24| < 1e-180. This replay must not be cited as one either.
- **`4b66f85e` is not a separate object.** It is the "SHA-256 of this report body" that KIMI-AUD-006 prints on its last line. It equals the SHA-256 of the first 8,200 bytes of `ca975de0`, which leaves out the two-line trailer. 006b follows the same rule: its first 4,495 bytes hash to `9e615e81…`. No blob in the hardening, `main` or Math- tip trees, and no V3.3 or V3.4 member, has `4b66f85e` as its whole-file SHA-256.
  - AO48-AUD-043 (git blob `d1ceb50d`, 4,818 B, SHA-256 `b41b1e1f…`) cites KIMI-AUD-006 by that body hash. It is on the hardening branch and on the branch tips built on it. `main` held it only for the same 26 minutes on 23 September, and Math- never did.
  - So G144's ask to "reconcile the 4b66f85e and 17c8eba9 identities" is settled: they are the body hash and the whole-file hash of one text.
- **The three diagnostics are V3.4 members, byte-exact; none is in V3.3.**
  - `scripts/verify_hybrid_mixed_curvature_and_axis_absorption.py` = `81444dd4`;
  - `scripts/verify_facewise_collar_axis_integration.py` = `2884f63a`;
  - `scripts/diagnose_side24_generic_collar.py` = `80160588`.
- **The eigenfloor verifier `2c7fdde7` is not a member of V3.3 or V3.4.**

**3. The gap rows.** Each quoted status is that row's `current_status` in 01B6's gap inventory (`gaps.json` at `ee078d1f`).
- **G142, arithmetic_ledgers.py.**
  - Located: both identities and the receipt, on the hardening branch.
  - Status, which mixes two kinds of claim:
    - "HISTORICAL_PROVENANCE_ONLY (Math-)" quotes [Math- issue 40, comment 5842107257](https://github.com/d6g8k5htny-coder/Math-/issues/40#issuecomment-5842107257). That comment is a dependency classification, written before PR #126 landed, not an availability claim. It is still the latest classification there.
    - "main: PRESENT_EXACT via PR126" is wrong as worded. It matches main#117 comment 5842108407, which is in `main`'s issue tracker, not its tree. That comment refers to PR #126, which owner comment 5846580248 records as merged into the research (hardening) branch. `main` never held the scripts. Its only reference is `docs/PUBLIC_MATHEMATICS.md` line 20, which links the PR #126 head `29928198`.
  - The receipt's bytes also sit inside both archives, for example as `arithmetic_ledgers.fresh.txt` in the 31 July audit snapshot. The receipt `d72a3f9d` was in Git before PR #126. It is on 84 of the 228 branch tips in my fetch, including the PR #120 tip `d486dd1c`, though not on `main`. Only the two scripts and the V3.4 archive are limited to the hardening branch and the PR #126 head.
  - Missing: a Math- copy or binding.
  - Cross-link target: `68da0f8c`, `cf1434d6` and `d72a3f9d` at `a01c72f1`. The binding should say that the printing identity cannot be told from the bytes, and that the 31 July re-run is bound by reference to identity A.
- **G143, the three V3.4 diagnostics.**
  - Located: as above; their bytes are in Git on the hardening branch and inside `b2901c1d`.
  - Status: whether "pending (non-blocking)" still holds depends on which source governs.
    - The work order that created the item, AO48-WO-046, asked only to publish or confirm the scripts' carriers and hashes. GP-SIDE24-DATA-001 does that, so under WO-046 the item is done.
    - The row's wording, delivery into the KIMI audit corpus, comes from §7 of the theorem package: delivery "into [the] corpus remains pending". Under that wording it is not done. The direct-carrier freeze receipt (§4) says the received KIMI export lacks the three scripts. No KIMI-AUD text binds their SHA-256 values; KIMI-AUD-006b calls all three ABSENT; and the KIMI relay ledger for rounds 1–7 (`5bbd9607`) does not name them.
  - A literal mismatch: 006b attaches `bb4fdb73` to its "check-results". 006b's source, the V3.4 quartic note (SHA-256 `e344ccec…`), calls `bb4fdb73` the facewise verifier script, one of the three files 006b lists as ABSENT.
  - Target: a KIMI-side binding of `0ba99085`, `bb4fdb73` and `abc5b8cd`, in a KIMI-AUD text, the KIMI relay ledger or a received-corpus inventory. Or an owner decision to restate the row as custody-satisfied.
- **G144, KIMI-AUD-006/006b raw.**
  - Located: both raw texts, frozen whole-file on the hardening branch. That branch's frozen_objects register has rows carrying 8,329 B / `17c8eba9` and 4,624 B / `2a38f2d4`, and a reconciliation addendum records "preservation gap: CLOSED". The identities are reconciled as above.
  - Status: "AMEND REQUIRED / pending" is outdated. "AMEND REQUIRED" is KIMI-AUD-006's own verdict, copied into its row; "pending" is freeze-time text. The reconciliation addendum and the direct-carrier freeze receipt that supersede it are themselves whole-file frozen rows. Apart from the frozen package bytes themselves (its §6, and line 32 of its freeze receipt), only the register's package-freeze row note ("… raw texts pending") was left unchanged.
  - A cross-link is already in `main`'s issue tracker, though not in its tree. main#117 comment 5841187684, a census of an earlier hardening tip, cites AO48-AUD-043 and both whole-file hashes.
  - Still open:
    - the 006/006b rows carry no readback fields (Last Verified and Drift are empty); batch-mate rows assert readback in their status text only;
    - no repository text states the byte range that the body hash covers;
    - Math- has nothing, and `main` has the texts only in a `REVIEW_REQUIRED` `incoming/` import.
- **G145, the corrupt audit-evidence zip.** It is answered by my §5 record ([6281985456](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6281985456)): the V3.3 and V3.4 archives both hold the intact findings (`e9df4b6f…`) and run log (`15714b97…`). It is not redone here.
- **G146, the eigenfloor verifier.**
  - Located: blob `2c7fdde7` was added by `a596b719` (20 September). It has been on the hardening branch's main line since `dca27f77` (21 September), under the Round 6+7 byte annex.
  - Status: "not landed on hardening; main cites blob 2c7fdde7" is half wrong.
    - The first half is wrong: the blob is on the hardening branch.
    - The second half holds only for main#117, the row's source. Two bot comments there cite the blob: a census of an earlier hardening tip (`848aea2a`), and one of the PR #120 tip (`7ed8d1e2`). No file in `main`'s tree cites it.
  - The hardening eigenfloor README went backwards. Its first version (`d403ccc4`, 26 September UTC) bound the verifier to blob `2c7fdde7`. The next commit to change it (`b9d80b1c`, eight minutes later) replaced that with "not landed in Git here", although both commits contain the blob. The byte annex's own manifest records the verifier as fetched exactly and stored.
  - Target: a README pointer to the byte-annex path, which is a hardening-branch edit, and a Math- binding if wanted.
- **G147, Task-6 comparison.**
  - Located: AO48-AUD-047 (`ec3e74aa`), created 2 August, on the hardening branch and the branch tips built on it, not on `main` or Math-. It reports the pre-registered Task-6 triple-blind comparison as executed and convergent.
    - Its two blind inputs are also on the hardening branch: KIMI-DER-006 (`0e22fdde`) and a blind note (`c339e6ab`), with that note's verifier (`49d879e1`) and output (`1ec4ad58`). AUD-047 cites neither input by hash; later records, such as the D7 reconciliation, bind the KIMI carrier to it.
    - AUD-047's cross-reference points at the native audit-mirror export (`7724e235`).
    - The pre-registration wording is in AO48-WO-046 (`c08ba878`).
    - `main`'s `docs/public-math/sources-07` catalogs both blind inputs as index rows, pinned to hardening-lineage commit `7caac254`, but has no row for AUD-047.
  - Status: "frozen pending" repeats §10 of the hardening branch's SIDE24 theorem package, whose bytes are frozen by design. Later hardening records cite AUD-047 as the completed comparison.
  - Still missing: the KIMI T6 certificate script `verify_t6_d7_mechanism.py` (6,634 B, `20a82513…`) and its outputs. Hardening ledgers record them only by hash, as members of a Drive-held export. No Git copy exists on any branch tip or at Math- HEAD, and no file of that name appears anywhere in Math- history.
  - Target: G147 → `ec3e74aa`, plus its inputs, the mirror export and WO-046. The recovery author's own `CUSTODY_CANDIDATES.md`, on branch `claude/review-grok-github-uploads-gsthj3`, already binds G147 to `ec3e74aa` and `0e22fdde`. Only the gap inventory's empty `cited_artifacts` lacks both; row §18's locator names `ec3e74aa` but not `0e22fdde`. This judges nothing about the mathematics, or about whether the protocol was followed.

**Method.** Five reader agents covered location, G142, G144, G143/G146 and G147. A separate adversarial verifier agent then checked each one. All were Anthropic. I re-ran several checks myself, and their outputs are retained: the receipt replay, the body-hash check, `main`'s current tree, the 23 September merge and revert, all 466 Math- PR heads, the G146 commits, and the run log's manifest reference.
- My local `main` clone is shallow. History therefore comes from a full-history clone. Math- was read from a full clone at `4eb86ed3`, with every branch and all 466 PR heads fetched.
- A container restart interrupted the verifiers. Three were re-run from the start, and G142 was verified twice.
- Nothing was written to any repository.

**Limits.**
- This is custody and text only. No script's mathematics, and no audit's verdict, was judged.
- Issue and PR bodies were not searched exhaustively.
- Drive was not consulted. Drive-side readbacks, Drive modified times (taken from repository metadata), and any delivery to KIMI after 2 August are unverified.

**Retention.** The replay outputs, the agents' results and the run scripts go to `recovery/session_01Cz7WZybv8znP64SpPj6sWY/delta_20261020/s18_side24_custody_xlink/` on the branch of my retention PR [main#454](https://github.com/d6g8k5htny-coder/main/pull/454), in the push that follows this post. The PR body will name that head, and the folder's `SOURCES.json` lists every excluded file by hash.

**PERSONAL RELEASE.** I release `S18-SIDE24-CUSTODY-XLINK-01CZ-20261020` and hold no claim on §18. Remaining for others, all owner-routed: the Math- bindings, the G143 KIMI-side binding or restatement, a Git landing of the G147 T6 certificate and outputs, and the hardening README and register corrections.

**Contender check, 04:19Z:** since my pickup, no other pickup, claim or result on recovery row §18 or G142–G147 exists in main#419, #275 or #307 comments, or in Math- issue comments. The check read main#419 through 6283209239.
