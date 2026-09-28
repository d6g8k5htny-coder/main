# CL_ANTHROPIC_BUNDLE v3 native copies — recovered from Drive, 2026-09-28

**Scientific effect: NONE.** **Review status: REVIEW_REQUIRED.**
Byte custody only. This packet accepts no theorem and changes no `STATUS.md`,
`Math-/PROOF_INDEX.md`, `GRAPH.json` or landing-claim row. It does not recover
`CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip`, the carrier the Math- downstream gate records
as `BLOCKED_ABSENT` for SYM-Fw-jet, and it does not discharge SYM-Fw-jet.

## What this is

On 2026-09-15 Claude wrote the mirror manifest `CL-MIRROR-001_MANIFEST.sha256.txt`. It
lists 46 frozen Kimi/K3 carriers, each with SHA-256, size and original path, that had
never been mirrored to Drive. The manifest says the byte-native copies went into the
chat-delivered `CL_ANTHROPIC_BUNDLE_2026-09-16_v3.zip`, folder
`06_UNMIRRORED_FROZEN_CARRIERS_BYTE_NATIVES/`. It also says they are "present
byte-for-byte inside the operator's own `09152026OKComputer_Project_Gap_Closure.zip`".

The bundle itself has not been found, but that zip is on Drive:

- Drive id `1vSI-evINWskhXVyiZ0slt-rLT74sPpXH`;
- 30,148,285 bytes, SHA-256
  `a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b`, which matches the
  manifest's own `a2136bc0…` reference.

It was downloaded through its public link on 2026-09-28. **All 46 listed files were
found in it, and every one matches its manifest SHA-256 and size.**

This packet publishes 45 of them. The 46th, `D3_percolation/PERC_DECAY.md`, was already
mirrored and is in `incoming/claude-audit-package-recovery-20260928/` (main #199). As of
2026-09-28, none of the 45 was on a current branch of the ten account repositories.

## Layout

Paths follow the manifest's `original_path` under `K3_SIDE24_LB/`, with three changes
that alter no bytes:

- `[agent-tree]/19fcef2e-c1c2-8c6c-8000-0f5a243156d9/` becomes `agent-tree-19fcef2e/`,
  because brackets are not intake-safe;
- `.py` and `.sha256` files get an added `.txt` suffix;
- the uploaded titles the manifest gives (for example `W8_PHASE2_STATUS.md`) are not
  used as filenames.

`SOURCE_MAP.json` records, for every file, its manifest `original_path`, uploaded title,
member name inside the zip, size and SHA-256.

Folders: `UPPER2D/` (B4LOC damline, D3 percolation, H3 band floor and ceiling, D1 v3
gate and receipts, H5 closure tightening, BRANCH_dir), `LPW_CONSTANT/` (v3 and v4),
`RETURN_06/` (executive state, theorem table, obligation ledger, dependency DAG,
adversarial report, addenda, source and tree manifests), `W3_numerics/`, `W8_lambda/`,
and `agent-tree-19fcef2e/work/` (W8 phase-2 scope and status).

## What was searched and not found

In all three gap-closure zips on Drive (`09152026…` above; `SepOKComputer…`
`16IVqTiZflqF4-IM9Q9rSIOernzdZwm0D`; `…(4).zip` `1Ykd9FO5oRUA5KoqOcDw64hwsQwXbs5ku`), each
hash-verified against its recorded inventory digest, the search covered 3,530 entries,
including nested zips, by filename and content. It found:

- no `rnu_env.py`, `allcell_fdz_enclosures.json` or `CL_ANTHROPIC_BUNDLE_*` file;
- no 24-jet interval table.

`RETURN_06/OBLIGATION_LEDGER.md` defines OBL-H5-JETMOD, the certified interval bounds
for the full 24-jet set over r-bands. That is the obligation, not its certificate.

## Screening

- Intake rules checked locally: safe paths, allowed suffixes, sizes, UTF-8, strict JSON.
- Privacy per `governance/OP-PRIVACY-20260927.md`: no e-mail addresses, phone numbers,
  access-key links, local user paths, credentials or URLs.
- The material is SIDE24 research (Kimi/K3 carriers), within project scope.

## Provenance

Recovered by Anthropic Claude (Claude Code) on 2026-09-28 at Dylan Roy's explicit
instruction. The carriers' own authors are as stated inside them, mostly Kimi/K3
sessions; this packet changes no authorship. Same-account custody: no independence
credit.
