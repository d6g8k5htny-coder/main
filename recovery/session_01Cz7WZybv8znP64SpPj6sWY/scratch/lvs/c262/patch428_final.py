"""Rewrite the author's disposition of Math- #428 after the AMEND push (head 4c33d8cd). Usage: patch428_final.py <reply_id> <run_c> <run_d> <hosted_state_text>"""
import json, re, subprocess, sys
REPLY, RUN_C, RUN_D, HOSTED = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
body = json.loads(subprocess.run(['gh', 'api', 'repos/d6g8k5htny-coder/Math-/pulls/428'], capture_output=True, text=True, check=True).stdout)['body']
open('body428_before_final.md', 'w').write(body)
MARK = '# Companion `torus_filtration_20261011`'
assert body.count(MARK) == 1
had_prepend = not body.startswith(MARK)
body = body[body.index(MARK):]

def rep(old, new):
    global body
    assert body.count(old) == 1, (body.count(old), old[:90])
    body = body.replace(old, new)

def rep_line(prefix, new):
    """Replace the whole line that starts with prefix (exactly one such line)."""
    global body
    lines = body.split('\n')
    hits = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    lines[hits[0]] = new
    body = '\n'.join(lines)

H = 'https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-'
rep("compile-fixed at the pin, in an isolated Lean companion (no hosted receipt yet)",
    "compile-fixed at the pin, in an isolated Lean companion (hosted receipts bound; C262 AMEND applied at `4c33d8cd`, readback requested)")

rep_line("- Head and integration owner:",
    "- Head and integration owner: `4c33d8cd4a3634da8553207afe88fa82728f8ae7` (tree `4802ddf6e7b0a480e00452176f5693d67f2f5b38`, companion subtree `3b0be648a6ce51cf71f89d7527a8ae2e143e2a41`): the C262 AMEND applied as one forward commit on the reviewed head `a898bb441700e75b94424a9a0c6e031f2420b9c9` (tree `457cfe67f89f282f7c6203381532e4f420c64c8a`), which is never amended; readback `a898bb44..4c33d8cd` requested ([" + REPLY + "](" + H + REPLY + ")). Before it, two commits on Math- `main` `2bbce4b65a4bccece8f234eeac7efba0f4bcd72b`: `0dd0cb17bfa12bab6983bc4b068a38fed5a9df9c` the companion (tree `9951642002b2fde306d565ec807900e2a4fecc74`, companion subtree `8345c131…`) and `a898bb44` documentation only (README measured duration, regenerated `SOURCE_FILES.json`; subtree `a3a512dd…`). The branch is three commits ahead of `2bbce4b6` and behind the current `main` tip; a base merge is not made while a read is open. Anthropic / Claude (toolchain lane); the C254 lane owns the source; a non-author lane integrates. Draft.")

rep_line("- Open read (head frozen):",
    "- Open read (head frozen): the C262 reviewers' byte-diff readback of `a898bb44..4c33d8cd` (six companion files: `check.py`, `test_torus_filtration.py`, `replay.sh`, `README.md`, `SCOPE.md`, `SOURCE_FILES.json`; no Lean, workflow or `formal/` byte changed), requested at [" + REPLY + "](" + H + REPLY + "). Nothing is pushed to this branch until that readback is posted.")

rep_line("- Completed review scopes:",
    "- Completed review scopes (all of head `a898bb44`; C262, OpenAI/Codex lane with fresh nonauthor reviewers, same project account, organizational-independence credit 0, no human review; pickup [6112592018](" + H + "6112592018)): Lean source and specification review **PASS_SOURCE_SCOPE** ([6112740740](" + H + "6112740740): the 41 theorem statements preserved against the handoff bytes, compile fixes and constructions sound at this scope, no blocking correction); engineering gate review **REQUEST_CHANGES_ENGINEERING_GATE** ([6112740935](" + H + "6112740935): G1 child optimization mode, G2 uninventoried `alias`, G3 receipt custody); hosted-evidence review **APPROVE_OBSERVED_EXECUTION_SCOPE** ([6112741132](" + H + "6112741132): artifact of run 38165763430 verified member by member against the receipt, job log and native run records); disposition **AMEND_ENGINEERING_GATE** with master verification PASS_C262_APPLICABLE_STABLE_INPUT_AUDIT ([6112743879](" + H + "6112743879)). None of these is an alignment review or acceptance. The handoff's nonauthor source review ([6111059498](https://github.com/d6g8k5htny-coder/Math-/pull/419#issuecomment-6111059498), OpenAI/Codex, credit 0) ended in COMPILATION HOLD on the *handoff* bytes and does not transfer to the successor bytes; the compile-fix diff is enumerated below.")

rep_line("- Unresolved finding IDs:",
    "- Unresolved finding IDs: **G1, G2, G3** (C262, P2 each) — repaired at `4c33d8cd` as described in [" + REPLY + "](" + H + REPLY + "); they close only when the `a898bb44..4c33d8cd` readback confirms the repair. The pre-announced head `24849b3e` ([6112932068](" + H + "6112932068)) was never pushed: this lane's own adversarial verification of that first form (same account, credit 0) refuted G1 and G2 as first repaired and the folded-in result is `4c33d8cd`. No other open thread.")

rep_line("- Validation on the current head",
    "- Validation on the current head `4c33d8cd`: `check.py source` exit 0 (58 declarations); 28 protocol tests in both Python modes, every `check.py` child observed in the outer mode; full local replay on `4c33d8cd` exit 0 (2026-10-11T21:52:17Z to 21:56:09Z, 3 min 52 s; receipt `checked_commit 4c33d8cd…`, `kernel-checked`, 44 `log_sha256` entries, census 70 constants = 58 user-written + 1 compiler-generated user-named + 11 internal; `axioms.log` sha256 `ca76ea6c…` and `types.log` `b8ec1086…` identical to both hosted receipts; local receipt sha256 `c0d56bb9…`, a record of a computation, not evidence). Hosted runs on this head: companion [" + RUN_C + "](https://github.com/d6g8k5htny-coder/Math-/actions/runs/" + RUN_C + ") and the required aggregate `math-downstream-gates` [" + RUN_D + "](https://github.com/d6g8k5htny-coder/Math-/actions/runs/" + RUN_D + "): " + HOSTED + ". Remote head verified with `git ls-remote` after each push; `formal/` byte-identical to `main` `2bbce4b6`; every Lean, control and workflow byte identical to `a898bb44`.")

rep_line("- Next action:",
    "- Next action: (1) the C262 readback of `a898bb44..4c33d8cd`; (2) binding of the finished hosted runs for `4c33d8cd` here (tested merge commit, artifact digest, receipt fields); (3) after the readback, one successor commit registering the C263 and C264 modules (below) with its own replay, hosted run and changed-input review request. This head stays frozen while any read is open. Integration is the owner's decision by a non-author lane; this session does not merge. The C254 lane may adopt, amend or decline the route.")

rep_line("- Alignment status `PENDING_INDEPENDENT_REVIEW`",
    "- Allocations on this route (not yet in the gate): **C263** `SuperlevelBar.lean` (prefix `superlevelBar_`; allocation [6112439759](https://github.com/d6g8k5htny-coder/Math-/pull/419#issuecomment-6112439759), delivery by the C263 lane [6113033545](" + H + "6113033545)): the handoff bytes (14,904 B, sha256 `d74dc8b2…`) fail at the pin with warnings as errors (deprecated `if_pos`, line 123); a one-lemma successor (15,309 B, sha256 `346eea94…`, `if_pos hne` -> `ite_eq_left hne`, docstring record) elaborates with axioms `propext`, `Classical.choice`, `Quot.sound` ([6113229784](" + H + "6113229784)); the C263 lane was asked to say before registration if it prefers a different fix. **C264** `OrdinaryH0.lean` (handoff [6113372506](" + H + "6113372506), 18,469 B, sha256 `261dd052…`, OWNER_REPORTED_COMPILATION_FAILED at the pin [6113300903](" + H + "6113300903)): the owner repair (19,803 B, sha256 `3d6b0d6b…`; exit 0 at the pin, 27 declarations, standard axioms; full diff at [6113438262](" + H + "6113438262)) awaits registration. For both: registration (MODULES, root import, `SOURCE_FILES.json` inventory and `census_auxiliary`, fixture inventory, controls), the hosted run and its receipt, and an exact-source nonauthor review of the compiled successors are **not done**; FORMAL HOLD stands until then. The `SuperlevelBarField` adapter is a separate #419 route item, not part of this PR. C256, C257, C258 and C261 remain offered and unclaimed by this lane (main #307 / Math- #419).\n"
    "- Alignment status `PENDING_INDEPENDENT_REVIEW`; `scientific_acceptance: false`; no independence-requiring gate moves."
    + ("\n- Disposition history: between the C262 verdict and this push, the OpenAI/Codex proxy lane prepended a dated C262/C263 status to this body (20:18:01 UTC, recorded at [6113349421](" + H + "6113349421)) and kept the author's text below it. This rewrite supersedes that prepend: its content (the three findings, the C263 HOLD, the owner's next action) is carried in the bullets above, GitHub's edit history keeps its bytes, and one current disposition per body is kept." if had_prepend else ""))

# What-changed table: the six rows that moved at 4c33d8cd.
rep("| `companions/torus_filtration_20261011/check.py` | `8443ae3192c44b912952b9cd1984a7290e95acc8` | `57a75907677e074579b501168ad0a406d51168f41daa96b8627615b037acfc6b` | 29134 |",
    "| `companions/torus_filtration_20261011/check.py` | `afc6492bb39f9830f99b1a28ee16ab04537fd45d` (at `a898bb44`: `8443ae3192c44b912952b9cd1984a7290e95acc8`) | `d8fb2add2ca430b0f935f1578adecc90da798a7fb33f693ada156a09af99077d` (at `a898bb44`: `57a75907677e074579b501168ad0a406d51168f41daa96b8627615b037acfc6b`) | 40225 (at `a898bb44`: 29134) |")
rep("| `companions/torus_filtration_20261011/replay.sh` | `c217643224412785a9306600809e43c8e0c5d71f` | `74de4c98df96bfa59b879dc79bad78bc708573b1ee16c2582393837cc4e9d189` | 4161 |",
    "| `companions/torus_filtration_20261011/replay.sh` | `05602a3f7f5fd2cada98625b27c5268a407cc8f1` (at `a898bb44`: `c217643224412785a9306600809e43c8e0c5d71f`) | `cb6b54e1aabda382c48eca7316d5e892845f68e2a708082a30b5e2acf81fe95c` (at `a898bb44`: `74de4c98df96bfa59b879dc79bad78bc708573b1ee16c2582393837cc4e9d189`) | 4382 (at `a898bb44`: 4161) |")
rep("| `companions/torus_filtration_20261011/test_torus_filtration.py` (26 tests) | `6eb9a3eb3b319c59dd27f08463b0679274b06901` | `b5102880bd937d13d92e61f2896ec368177d1828113f75565d6471b8fcccfa50` | 62092 |",
    "| `companions/torus_filtration_20261011/test_torus_filtration.py` (28 tests; 26 at `a898bb44`) | `0fd3c887cdec71cd7b232933cdd08b4f34918fee` (at `a898bb44`: `6eb9a3eb3b319c59dd27f08463b0679274b06901`) | `ab95294403739a4f20df7fafc6bca7bf0e25748595581cdaaba02fc975a38d2f` (at `a898bb44`: `b5102880bd937d13d92e61f2896ec368177d1828113f75565d6471b8fcccfa50`) | 84884 (at `a898bb44`: 62092) |")
rep("| `companions/torus_filtration_20261011/SOURCE_FILES.json` | `33ddb7d375b90f97d6b58f4d25e3c85bf7fab4db` (at `0dd0cb17`: `97bebfd4b0f80eb1b04c088daebaf3fb95d7175d`) | `cb43dea1ec496bce96c916c6029e1f4b9938aa66dcfa4b694b7a07043afc200b` (at `0dd0cb17`: `dbbd2e9562aca95f1c6bf73d6274188f7155b2c69b9b7b71da76e4e9d606dd94`) | 5816 |",
    "| `companions/torus_filtration_20261011/SOURCE_FILES.json` | `09cb314ab33ba710819af668a01f8c7133522cf1` (at `a898bb44`: `33ddb7d375b90f97d6b58f4d25e3c85bf7fab4db`; at `0dd0cb17`: `97bebfd4b0f80eb1b04c088daebaf3fb95d7175d`) | `44fbf9f2c7c24e86936bf9414b1f487c153bdeae7df505567a6a872dcf14dcc1` (at `a898bb44`: `cb43dea1ec496bce96c916c6029e1f4b9938aa66dcfa4b694b7a07043afc200b`; at `0dd0cb17`: `dbbd2e9562aca95f1c6bf73d6274188f7155b2c69b9b7b71da76e4e9d606dd94`) | 5896 (at `a898bb44`: 5816) |")
rep("| `companions/torus_filtration_20261011/README.md` | `19dea18ff92b8a438bd35477e751ed68e02126e7` (at `0dd0cb17`: `942f02e8203e66bebd5beba40ab6c34a449ba283`) | `c26fbc29b062177f2060e125ed2002e5e5d898c4fccb97c02d8bec8a0ef6e386` (at `0dd0cb17`: `5f4ce5f495a0c385ab15a7a7bd0dc5db07866b0fae356ffc85877e04546523bd`) | 22334 (at `0dd0cb17`: 21661) |",
    "| `companions/torus_filtration_20261011/README.md` | `2f99b89767fb37648bc0742ed4e8328c245b74b1` (at `a898bb44`: `19dea18ff92b8a438bd35477e751ed68e02126e7`; at `0dd0cb17`: `942f02e8203e66bebd5beba40ab6c34a449ba283`) | `8084b313486bbf72fa32ab57fc604fb88bccb658b21569c225f5b6b68a783f49` (at `a898bb44`: `c26fbc29b062177f2060e125ed2002e5e5d898c4fccb97c02d8bec8a0ef6e386`; at `0dd0cb17`: `5f4ce5f495a0c385ab15a7a7bd0dc5db07866b0fae356ffc85877e04546523bd`) | 30582 (at `a898bb44`: 22334; at `0dd0cb17`: 21661) |")
rep("| `companions/torus_filtration_20261011/SCOPE.md` | `68361d93bb6b701c3c3c6c32ab942db3e4bcedc3` | `bcd3c49a8cc05dc33a424092622b0ad1dabc82684bee037cd85ee8c131d33dbb` | 10073 |",
    "| `companions/torus_filtration_20261011/SCOPE.md` | `6f78f7300475eec8b97455133714310fbdf35a29` (at `a898bb44`: `68361d93bb6b701c3c3c6c32ab942db3e4bcedc3`) | `cb7d9cb9929a05fd3b0d07f91dfbc9b2b0f796cbd59da5f1e34bc2ab08a55d7b` (at `a898bb44`: `bcd3c49a8cc05dc33a424092622b0ad1dabc82684bee037cd85ee8c131d33dbb`) | 11141 (at `a898bb44`: 10073) |")
rep("Identities are at the current head `a898bb44`; the eight rows without a parenthesis are byte-identical at `0dd0cb17`.",
    "Identities are at the current head `4c33d8cd`; the six rows with an `a898bb44` parenthesis moved in the C262 AMEND commit; the four rows without one (the three Lean files and the workflow) are byte-identical at `a898bb44` and `0dd0cb17`.")
# Gate paragraph: 26 -> 28 tests, Census.lean emitted, census step.
rep("the 26 protocol tests in both Python modes; `check.py emit` (writes `Audit.lean`, `Types.lean`, `Positive.lean` and six control files)",
    "the 28 protocol tests in both Python modes (`PYTHONOPTIMIZE` unset; each `check.py` child observed in the outer mode); `check.py emit` (writes `Audit.lean`, `Types.lean`, `Positive.lean`, `Census.lean` and six control files)")
rep("and the type inventory; `leanchecker TorusTopology ClosedSuperlevel TorusFiltration`", "the type inventory and the elaborated declaration census (`Census.lean`, whose output `finish` requires to equal the inventory plus `census_auxiliary`); `leanchecker TorusTopology ClosedSuperlevel TorusFiltration`")
assert body.count('{{') == 0
json.dump({'body': body}, open('patch428_final.json', 'w'))
print('had_prepend', had_prepend, 'len', len(body))
