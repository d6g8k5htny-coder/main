import sys
S='<SCRATCH>'
t=open(f'{S}/lvs/c26x_fix/body428_before.md',encoding='utf-8').read()
P='https://github.com/d6g8k5htny-coder/Math-/pull/'
R='https://github.com/d6g8k5htny-coder/Math-/actions/runs/'
def rep(a,b):
    global t
    n=t.count(a)
    assert n==1, (n,a[:90])
    t=t.replace(a,b)

# heading
rep("# Companion `torus_filtration_20261011`: the C254 torus-quotient and closed-superlevel modules, compile-fixed at the pin, in an isolated Lean companion (hosted receipts bound; C262 AMEND applied at `4c33d8cd`, readback requested)",
    "# Companion `torus_filtration_20261011`: the C254 torus-quotient and closed-superlevel modules and the C263/C264 successors, compiled at the pin, in an isolated Lean companion (C263/C264 registered at `940dffc9` under FORMAL HOLD; hosted receipt bound; readback and exact-source review requested)")

# source author sentence
rep("**Source author: the OpenAI/Codex lane (C254, claim `8a5c7a24-784b-4aaa-b951-cd9f37b0dc64`).**",
    "**Source author: the OpenAI/Codex lane (C254, claim `8a5c7a24-784b-4aaa-b951-cd9f37b0dc64`; C263 and C264 handoffs, attributed per module in the README's registration table).**")
rep("Compile fixes, companion gate, workflow and this PR: Anthropic / Claude",
    "Compile fixes (C254, C263), the C264 repair, companion gate, workflow and this PR: Anthropic / Claude")

# head bullet
rep("- Head and integration owner: `4c33d8cd4a3634da8553207afe88fa82728f8ae7` (tree `4802ddf6e7b0a480e00452176f5693d67f2f5b38`, companion subtree `3b0be648a6ce51cf71f89d7527a8ae2e143e2a41`): the C262 AMEND applied as one forward commit on the reviewed head `a898bb441700e75b94424a9a0c6e031f2420b9c9` (tree `457cfe67f89f282f7c6203381532e4f420c64c8a`), which is never amended; readback `a898bb44..4c33d8cd` requested ([6114262101]("+P+"428#issuecomment-6114262101)).",
    "- Head and integration owner: `940dffc9bac927250c03a8194aaa75552cdda060` (tree `1388d3a98f60e3fd8a55d1948d86a587afb844ee`, companion subtree `9e9b99bb506dfb145b2695412df199c974582063`): the C263/C264 registration as one forward commit on `4c33d8cd4a3634da8553207afe88fa82728f8ae7` (tree `4802ddf6e7b0a480e00452176f5693d67f2f5b38`, companion subtree `3b0be648a6ce51cf71f89d7527a8ae2e143e2a41`), the head closed by the C262 readback [5485368068]("+P+"428#pullrequestreview-5485368068), which is never amended; readback `4c33d8cd..940dffc9` requested ([6115569460]("+P+"428#issuecomment-6115569460)). `4c33d8cd` is the C262 AMEND as one forward commit on the reviewed head `a898bb441700e75b94424a9a0c6e031f2420b9c9` (tree `457cfe67f89f282f7c6203381532e4f420c64c8a`).")
rep("The branch is three commits ahead of `2bbce4b6` and behind the current `main` tip;",
    "The branch is four commits ahead of `2bbce4b6` and behind the current `main` tip;")

# hosted receipt bullet: prepend current-head binding
rep("- Hosted receipt: **bound** for the companion head `0dd0cb17`.",
    "- Hosted receipt on the current head `940dffc9`: **bound**. Run [38187870416]("+R+"38187870416) (`torus-filtration.yml`, attempt 1, job [114614043183]("+R+"38187870416/job/114614043183), success, 2026-10-12T00:24:20Z to 00:28:39Z) on PR head `940dffc9`: the actually tested checkout is the synthetic merge `e3a42ea1d62e46083c1ea1b3dd18c1762740d1bb` of that head into the `main` tip `79cb314d` (parents `79cb314d`, `940dffc9`; tree `793bc97b…`, companion subtree `9e9b99bb…` = the head's, `formal` tree `38b6d1a7…` = `main`'s; fetched and checked here), recorded as `checked_commit`. GitHub later recomputed the merge ref as `a66480a9…` with the same parents and the same tree. Receipt (read from the job log, where `check.py finish` prints it): `formalization_status: kernel-checked`, `theorem_count: 81`, definitions 3 abbrev / 22 def / 3 instance, `leanchecker TorusTopology ClosedSuperlevel SuperlevelBar OrdinaryH0 TorusFiltration` (mode `modules`), census 133 = 109 user-written + 1 compiler-generated user-named + 23 internal, all eight controls refused for their intended reasons (five `REJECTED_BY_LEAN` including the new `RejectEssentialDeathBound` and `RejectReducedPointClass`, three `REJECTED_BY_AXIOM_AUDIT`), 54 `log_sha256` entries, with `axioms.log` `4056ba24…`, `types.log` `3549f2b1…` and `census.log` `b044e298…` equal byte for byte to this lane's local replay of `940dffc9`; Lean 4.34.1 (`5045d005…`), Mathlib `d13f23b7`; 28 tests passed with outer optimize 0 and 1; every source entry's blob equal to the head's; `run_id 38187870416`, `run_attempt 1`; `alignment_status: PENDING_INDEPENDENT_REVIEW`, `scientific_acceptance: false`. Artifact `torus-filtration-38187870416-1` id 11719939176, 55 files, 36,924 B, GitHub digest sha256 `09e4ff6ec738a141d3630ef98cd8da649833b6d73bc9a537e90625f8d84229b5` (equal to the upload step's log line; this lane cannot download artifact bytes, so the digest is GitHub's, not re-hashed here), expires 2026-11-10. The required aggregate [38187870597]("+R+"38187870597) (`downstream-replay` 114614043741, `formal / formal-evidence` 114614043936, `math-downstream-gates` 114614734640, all success) printed the same `checked_commit e3a42ea1…`; artifacts `downstream-gate-38187870597` id 11720323046 (digest `7230fc73…`) and `formal-evidence-38187870597-1` id 11720397770 (digest `7f1950c5…`). This receipt records a computation on the registered bytes. It is not a review of the C263/C264 successors, and FORMAL HOLD stands.\n- Hosted receipts on earlier heads (kept for the record): **bound** for the companion head `0dd0cb17`.")

# local replay bullet
rep("- Local replay on this head (record of a computation on this lane's worktree, not a receipt):",
    "- Local replay on the first companion head `0dd0cb17` (record of a computation on this lane's worktree, not a receipt):")

# open read bullet
rep("- Open read: none. The C262 readback of `a898bb44..4c33d8cd`",
    "- Open read: the changed-input readback of `4c33d8cd..940dffc9` requested in [6115569460]("+P+"428#issuecomment-6115569460); this head stays frozen while it is open. Separately requested there: an exact-source nonauthor review of the two compiled successors, which is FORMAL HOLD's own lifting condition. Closed read: the C262 readback of `a898bb44..4c33d8cd`")

# unresolved bullet
rep("No other open thread.\n- Validation on the current head `4c33d8cd`:",
    "No other open thread. Two earlier unpushed forms of the registration commit were refuted by this lane's own adversarial verification (same account, credit 0) and folded into `940dffc9`; no finding is open on it.\n- Validation on the current head `940dffc9` (author-side, local): `check.py source` exit 0 (109 declarations, 81 theorems); 28 protocol tests (671 subtests) in both Python modes; full local `replay.sh` with `LEAN_NUM_THREADS` unset, exercising the new default of 2: exit 0, 2026-10-12T00:16:50Z to 00:23:46Z, receipt `checked_commit 940dffc9…`, `kernel-checked`, 54 `log_sha256` entries, census 133 = 109 + 1 + 23, all eight controls refused for their intended reasons; local receipt sha256 `f336ddec86bc19fa9acd8a11fbf9a78cff627c205d2edfd86557a2bc787a473e`, a record of a computation, not evidence. Remote head verified with `git ls-remote` after the push. `formal/`, the workflow, `TorusTopology.lean` and `ClosedSuperlevel.lean` are byte-identical to `4c33d8cd`.\n- Validation on the previous head `4c33d8cd`:")

# next action
rep("- Next action: (1) one successor commit registering the C263 and C264 modules (prepared locally on a wip branch, under this lane's own adversarial verification; pushed only after it passes), followed by its own `4c33d8cd..<new>` changed-input readback request and the binding of its hosted receipt here; (2) the C258 `ComponentKernel.lean` slice when its author delivers exact bytes (Math- 419/6114716011); (3) integration is the owner's decision by a non-author lane; this session does not merge.",
    "- Next action: (1) answer the readback of `4c33d8cd..940dffc9` and any finding it raises; (2) the exact-source nonauthor review of the two compiled successors, by a lane other than this one; (3) the C258 `ComponentKernel.lean` slice when its author delivers exact bytes (Math- 419/6114716011), as a further forward commit after the open read closes; (4) integration is the owner's decision by a non-author lane; this session does not merge.")

# allocations bullet
rep("- Allocations on this route (not yet in the gate): **C263**",
    "- C263 and C264 on this route (registered at `940dffc9`, FORMAL HOLD): **C263**")
rep("the C263 lane was asked to say before registration if it prefers a different fix.",
    "the C263 lane was asked to say before registration if it prefers a different fix; no preference was posted, so that fix stands.")
rep("awaits registration. For both: registration (MODULES, root import, `SOURCE_FILES.json` inventory and `census_auxiliary`, fixture inventory, controls), the hosted run and its receipt, and an exact-source nonauthor review of the compiled successors are **not done**; FORMAL HOLD stands until then.",
    "is registered. For both: registration is done at `940dffc9` (MODULES, root import, `SOURCE_FILES.json` inventory and successor record, fixture inventory, one new false control each; `census_auxiliary` needed no new entry) and the hosted run on that head is bound above. An exact-source nonauthor review of the compiled successors is **not done**, and FORMAL HOLD stands until it exists. The concrete ordinary-H0 controls of the C264 owner task (item 4 of [6113277485]("+P+"428#issuecomment-6113277485)) are not provided; that obligation stays open.")

open(f'{S}/lvs/body428/stage1.md','w',encoding='utf-8').write(t)
print('stage1 ok', len(t))
