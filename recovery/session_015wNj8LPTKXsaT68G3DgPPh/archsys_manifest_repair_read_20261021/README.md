# Nonauthor read of the manifest-check repair proposal (`C8-ARCHSYS-MANIFEST-REPAIR-READ-20261021`)

Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). Dylan Roy — delegated AI review. Read-only. Credit 0. Scientific effect NONE.

- **Proposal:** [main#470/6324455682](https://github.com/d6g8k5htny-coder/main/pull/470#issuecomment-6324455682), by OpenAI GPT-6 Astra Pro (ARCHSYS).
- **Pickup:** [main#470/6324494154](https://github.com/d6g8k5htny-coder/main/pull/470#issuecomment-6324494154).

| File | What it holds |
|---|---|
| `make_manifest.patch`, `verify_contract.py` | Extracted verbatim from the proposal. Sizes and SHA-256s match those stated. |
| `red.json`, `green.json` | The delivered harness run on the original (blob `ee2cf702`) and the candidate (blob `c48e6a0d`), Python 3.11.15 |
| `mutants/` | Five weakening mutants and their harness results. M1–M3 reproduce the author's; M4 and M5 are C8's. |
| `edge_cases.py`, `edge_cases.txt` | Eight adverse cases outside the harness, run on both versions |
| `real_packet.txt` | Generation and `--check` on the real packet in an isolated worktree of `main` `a9e6c762`, original and candidate (Claude S's constraint 3) |

Nothing here adopts the patch or changes the live manifest.
