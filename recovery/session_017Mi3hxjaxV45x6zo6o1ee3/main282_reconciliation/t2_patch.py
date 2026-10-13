"""Build the T2 body for main#282: new current-disposition header on top, the 2026-10-07 18:35 header
retitled as superseded history, everything else byte-identical.

Usage: python3 -I t2_patch.py <body_full.md> <out body_new.md> <readback_comment_id> <utc HH:MM> [remaining_ids_csv]
Read-only: PATCHing is done separately with gh api.
"""
import sys
from pathlib import Path

src, dst, readback_id, hhmm = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
remaining = sys.argv[5] if len(sys.argv) > 5 else ""
body = Path(src).read_text()
OLD_TITLE = "## Current disposition — 2026-10-07, 18:35 UTC"
assert body.startswith(OLD_TITLE), body[:80]
body = body.replace(OLD_TITLE, "## Historical disposition — 2026-10-07, 18:35 UTC (superseded by the header above)", 1)
remaining_txt = (
    f"remaining unresolved native finding IDs after T1: {remaining}" if remaining
    else "no native review thread remains unresolved after T1 (17 resolved by ID with linked evidence; the historical eight untouched)"
)
header = f"""## Current disposition — 2026-10-12, {hhmm} UTC

Successor custodian pass (C202-282-T1/T2) by Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3`, delegated for Dylan Roy, successor to the stopped author session `01UzNu9j…` (its own closeout UNKNOWN); activation on main#307 as linked in the readback. Metadata reconciliation only: no source byte, ref, check, protection or setting changed. Scientific effect **NONE**; organizational-independence credit 0. Same-provider as the original author; not a nonauthor review.

- **Current candidate:** open, non-draft, unmerged at head `33c40f77132a27cdb9e6507387d1b6c267b2ef6b` (sole parent `6ac2b53d`; 2026-10-08 02:39Z) over PR base metadata `d4d5271e332387eab25729ccb4bfa565e1ffba07`; default `main` separately read at `9d51627ef6d16018d651f71da7b5188dd353b806` (24 ahead, 261 behind). The two modified docs carry on `main` the same blobs as at the PR base (`docs/PUBLIC_SHOP_SETUP.md` `4c9707a6`, `docs/FORMAL_REQUIRED_CHECKS.md` `06765589`); `governance/OP-ACCESS-20261007.md` is absent from `main`.
- **Source identities at the head:** `governance/OP-ACCESS-20261007.md` `00aa5ba05788796b78803a0444b2a0d9b6434059` (19,464 B, SHA-256 `f7675b68abd802e372033789e2ad9faee8597cba0dbe80663eea2ff0abfc27b4`); `docs/PUBLIC_SHOP_SETUP.md` `6eb9638b26782b79c3fad0f0683c5eea28d597f0`; `docs/FORMAL_REQUIRED_CHECKS.md` `1cbe8611e8bbe05771d0a69c70aedabd2f7cd2d4`. Identical to the identities in C202's map.
- **Review disposition:** the 17 native threads that were unresolved are reconciled one by one in the readback [282/{readback_id}](https://github.com/d6g8k5htny-coder/main/pull/282#issuecomment-{readback_id}) against C202's map [282/6061731184](https://github.com/d6g8k5htny-coder/main/pull/282#issuecomment-6061731184), the scoped successor reviews it names (`5445669456`, `5445730858`, `5445901007`, `5446550667`, `5446608316`, `5448436937`, `5450232763`, `5450428614`, `5454563811`) and the preserved passages re-read at `33c40f77`; {remaining_txt}. Earlier reviews stay historical evidence; none is reinterpreted as blanket certification.
- **Verification:** seven hosted check runs at `33c40f77` completed `success` on 2026-10-08 (verify, formal / formal-evidence, landing-checks, public-shop, browser-smoke, navigation, public-intake); not re-run and not rebound to any later base. Draft [main#347](https://github.com/d6g8k5htny-coder/main/pull/347) also edits `docs/FORMAL_REQUIRED_CHECKS.md`, in a different region; whichever lands second needs an ordinary base refresh.
- **Next owner / action:** C202-282-T3, a distinct eligible integrator (not this session, not the original author, not C202's reviewer): fresh pickup after live contention reconciliation, base refresh by merge commit under branch custody, current checks, protections, then merge. `AGENTS.md` pointer stays a separate task after landing. No readiness change or merge is performed by this pass.

The 18:35 UTC header below and the complete earlier description are retained unchanged as historical snapshots; their candidate/status language does not override this header.

---

"""
Path(dst).write_text(header + body)
print("wrote", dst, len((header + body).encode()), "bytes")
