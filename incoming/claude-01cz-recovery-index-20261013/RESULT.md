# Recovery index: session 01Cz scratch work (2026-10-13)

**Scientific effect: NONE.** **Review status: REVIEW_REQUIRED.**
Byte custody and indexing only. This packet does not accept any theorem, change any claim,
Prop or packet label, or discharge any obligation.

## What this is

The owner asked every Claude session to preserve its still-local project work so that no
derivation, probe or tool has to be repeated after an interruption (main #307, comments
6136156988, 6136210973; implementation decision 6136231457: register each checkpoint once
through the existing delivery procedure). The bulk of this session's material is far above
the intake limits of one packet (50 files, 2 MiB), so the payloads live on a retained reading branch
and this packet is the durable index that points at them by commit and digest. Retention is explicit: the payload
branch and commit are retained while any required work relies on them, until a verified durable replacement contains
the actual bytes and this index is updated; an index alone is not a replacement backup; no deletion is requested
(main #307, 6136476552).

## Pointers

- Reading copy: `d6g8k5htny-coder/main`, branch `claude/recovery-01cz-20261013`, commit
  `ee7491ba00b8df361b459e8f0dfb6fd38dfa6e4e`, tree `501a3519`, directory `recovery/session_01Cz7WZybv8znP64SpPj6sWY/` (moved unchanged from
  `sandbox/recovery_01cz_20261013/` at `037436b454003593af8039907a68490331300f9e`; same payload bytes and `MANIFEST.json`, README reworded):
  4,590 payload files, 85,671,616 bytes under `scratch/<original relative path>`,
  with `MANIFEST.json` (per file: area, kind, bytes, SHA-256 of the published copy and of the
  exact as-run bytes, scrubbed flag), `EXCLUDED.json` (185 entries) and `README.md`.
  All three are pinned as sources of this packet.
- Handoff reading copy (renderer of main #358 and the round-5/6 binders, 26 files): branch
  `claude/handoff-lean-register-tools-20261013`, commit `b88efe53a862ef56fcb55457e568601a8c47d5ea`; its
  `MANIFEST.json` and `README.md` are pinned as sources too.
- `RECOVERY_INDEX.json` (this packet) lists the areas, the landed objects each area fed, the
  remaining obligations, blockers and the next reproducible step.

| area | files | bytes | scrubbed |
|---|---|---|---|
| `renderer_versions_1_9` | 147 | 7,422,313 | 33 |
| `binders_rounds_1_9` | 16 | 318,358 | 10 |
| `binders_rounds_10_14_with_logs` | 350 | 18,369,854 | 125 |
| `round4_triage_probes` | 268 | 2,229,042 | 44 |
| `proof_packet_rounds_5_6_verification` | 342 | 12,943,211 | 12 |
| `museum_review_probes` | 1152 | 14,757,238 | 107 |
| `torus_companion_c254_c26x` | 2024 | 23,905,354 | 216 |
| `lvs_misc_notes_and_fixes` | 182 | 2,372,381 | 21 |
| `frontier_round_and_handoff_tooling` | 52 | 1,785,354 | 1 |
| `lvs_top_level_notes` | 57 | 1,568,511 | 14 |

## Labels and limits

Everything on the reading branches is scratch: candidates, probes, mutants, logs, drafts and
failed attempts. None of it is a receipt, a review verdict or evidence about a mathematical
claim. The only edits to the published copies replace the session's private scratch root, its
session UUID and the owner's relay address by placeholders; both digests are in the manifest.
Repository and tree copies, Lean build caches and build products were not copied (reproducible
from the commits the neighbouring notes name and from the pinned toolchain). Organizational
independence credit of this packet: 0 (Anthropic / Claude, the same session that produced the
material). Retention: the reading branches are recovery inputs and are retained while any required work relies on them,
until a verified durable replacement contains the actual bytes and this index is updated; no deletion is requested.
