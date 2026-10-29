# CBZ-0 decision ledger

Append-only. Historical source outcomes remain in their original records.

| UTC | Phase | Layer | Action and rationale | Reversible | Evidence |
|---|---|---|---|---|---|
| 2026-10-29 16:48 | 1 | B | Applied the owner's new CBZ-0 protocol; superseded earlier owner procedural freezes, family-review gates and approval/custody ceremonies. Mathematical rigor and platform permissions remain binding. | Yes | Owner's current MASTER DIRECTIVE; [board notice](https://github.com/d6g8k5htny-coder/main/issues/619#issuecomment-6502880391) |
| 2026-10-29 16:50 | 1 | C | Preserved old local research worktrees and cloned current sources separately under `/workspace/cbz`; stale local Math main is not treated as live truth. | Yes | Math live source `8151f592918a845bf19eb8eafaf8444540a32734`; main `4ef1867175dbe1072a54a02e95f80c4e2c70a4ff` |
| 2026-10-29 16:50 | 1 | B | Assigned independent read-only repository/ref, formal/auditor, foundation/pin and regression discovery. Root retains ownership of canonical shared files. | Yes | Governance B-003; discovery directories |
| 2026-10-29 16:53 | 1 | C | Created recoverable `pre-cbz-discovery-1` tags before consolidation; successfully pushed Math tag. No branch or source was deleted. | Yes | Math tag at `8151f592918a845bf19eb8eafaf8444540a32734` |
| 2026-10-29 16:54 | 1 | C | Reused the existing compiled-environment formal auditor as the first audit candidate. Building a new lexical replacement would lose verified coverage. | Yes | Math `formal/gate.py`, `gate_helper.py`, PR #312 |
| 2026-10-29 16:55 | 1 | C | Resolved agent requests for missing gate definitions autonomously: supplied the owner's gate matrix and continued bounded discovery. No question was referred to the owner. | Yes | Parallel discovery handoffs; Governance B-003/B-005 |
| 2026-10-29 16:56 | 1 | C | Began installing the repository-pinned Lean toolchain for fresh build/kernel execution. Historical hosted CI will remain historical evidence. | Yes | Math `formal/lean-toolchain`, `lake-manifest.json` |
| 2026-10-29 16:54 | 1 | C | Timestamp correction: the entries labelled 16:55 and 16:56 above used projected times by mistake; both actions had actually completed by the observed clock reading 16:54:09 UTC. The actions and evidence are unchanged. | Yes | UTC clock observation 2026-10-29 16:54:09; append-only correction |
| 2026-10-29 16:54 | 1 | C | Direct fast-forward push to main was rejected by repository rule GH013 (PR required and three current status checks required). Use a PR with actual automated checks and autonomous integration; no owner input is needed. Do not bypass the rejection. | Yes | Git push exit 1, remote rule readback |

| 2026-10-29T21:20:29.131Z | 2 | B | Replaced the public-intake checker's500-file pre-enumeration limit with the documented3000-file API completeness boundary and strengthened exact page cardinalities. Retained50-file public-package admission, all existing checks and85 prior test bodies.97 controls/mode and captured965-file replay passed; independent review retains the pre-existing A-to-B-to-A endpoint race limit. No workflow/context/protection change or mathematical status effect. | Yes | `tools/public_intake_check.py` SHA25627afbbe278183e30ef647211a884a0d0fb2c6cfa674e4f181cca96add350e57e; `tests/test_public_intake.py`; ordinary PR evidence. |
