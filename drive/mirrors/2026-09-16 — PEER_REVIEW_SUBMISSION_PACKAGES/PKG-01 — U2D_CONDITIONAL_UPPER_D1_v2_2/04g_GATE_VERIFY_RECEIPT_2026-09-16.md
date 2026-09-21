# Gate verify receipt — D1 v2.2 — 2026-09-16

Verifier: agent 5 (coding)
Extract: `/workspace/drive_peer_review_triage/extract_sep15/` from zip SHA-256 `a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b`

## Result
- `d1_falsify_v3.py` both modes (`python3` and `python3 -O`): **PASS** exit 0
- PASS digest: `d800849ed5966b4d583a10cf72525e89350d9a53780b45a807a21ccbadf91085` (matches receipts)
- Transcripts byte-identical to `v3_t_normal.txt` / `v3_t_opt.txt`

## Drive PKG-01 notes
- Gate files 04d / 04e / 04c / assembly match
- `v3_t_opt` was missing on Drive; uploaded as `04f_v3_t_opt.txt` after this receipt
- Non-gate size drift on START_HERE / MANIFEST from parallel packaging — not a gate hash mismatch

## Not claimed
No theorem closure. Five open validity premises remain (see `12_OPEN_PREMISES_WORKLIST.md`).
