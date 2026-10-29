# Custody note — required formal checks evidence, 2026-09-27

**Scientific effect: NONE.** This folder preserves the deployment evidence for
`OP-FORMAL-ENFORCEMENT-20260927-v1.0` (main #188, Math- #96). It was produced
by OpenAI / ChatGPT, and Dylan passed it on for preservation, because the original
GitHub Actions artifacts expire.

- Delivered bundle: `Required_Formal_Checks_Evidence_20260927.zip`, 659,141 bytes,
  SHA-256 `24926e959b806a5103fed9cae2336ae54b2178b70307dea1f14379c3a9346868`.
  Its 13 members are stored here unchanged. `sha256sum -c SHA256SUMS` in this
  folder verifies the 12 files it lists.
- The three original workflow artifacts match the report's identities:
  `Main188_landed_formal_36358139870.zip` (`4fc0ac1e…16aa7`),
  `Math96_landed_formal_36359933550.zip` (`bc034213…0f4a`) and
  `Math96_landed_downstream_36359933550.zip` (`43bd4650…3db2`).
- Cross-checked against GitHub on 2026-09-28: main push run 36358139870
  (`3592abb`) and Math- push run 36359933550 (`22e79e8`) both concluded
  `success`. The later heads, main `8886d84` and Math- `48cfe40`, were also
  green with the formal workflow included.

What this establishes and what it does not is stated in [README.md](README.md).
In short: code-level enforcement of fresh formal evidence through the existing
required checks, and no change to ruleset settings, statement alignment or any
scientific status.

Recorded by Anthropic / Claude (Claude Code).
