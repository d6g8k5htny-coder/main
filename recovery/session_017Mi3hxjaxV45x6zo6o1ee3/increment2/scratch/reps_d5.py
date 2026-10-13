[
("(v1.2). **Scientific effect NONE.**", "(v1.3). **Scientific effect NONE.**"),
("""The C8 band's share of the birth mass by dimension:

| `d` | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| band share | `48.3 %` | `34.6 %` | `15.4 %` | `4.37 %` |
""", """**How the band's share moves with dimension.**

| `d` | height-window share `F^(d)_{[0,1]}` (every `K`) | C8 band's share of `c_{d,ref}`, `F^(d)_{[0,1]} G_{[1/2,2]}` |
|---|---|---|
| 2 | `48.3 %` | `3.25 %` |
| 3 | `34.6 %` | `2.33 %` |
| 4 | `15.4 %` | `1.04 %` |
| 5 | **`4.37 %`** | **`0.295 %`** |

The first column is the share of birth heights in `B = [0, 1]`. By the factorization `c_{B,K} = c_{d,ref} F_B G_K` it does not depend on the gap window. The second column is the C8 band's share of the whole coefficient, with the dimension-free gap factor `G_{[1/2,2]} = 6.74 %`. Both are ratios of certified values from the separate packets, not a theorem about birth mass (v1.3).
"""),
("- **Head / owner:** `b22ff4b` (v1.2) on", "- **Head / owner:** `0d6bc80` (v1.3) on"),
("""personal owner reading pending; zero organizational-independence credit.
- **Revision v1.2 (`b22ff4b`).**""", """personal owner reading pending; zero organizational-independence credit.

  **xAI / Grok partial Slice B (Harper, comment 5932371862, head `b22ff4b`; script bytes unchanged at `0d6bc80`):** NOT READY. Floor check only: `(8 − 2√13)/3 = 0.262965…`, so `1/4` and `0.262` hold and `0.263` and `3/10` fail, matching `LAMBDA_FLOOR`. It does not re-derive the Schur complement, and the Owen-type cancellation and `F_4` were not re-expanded. Unanimous READY is not present. Acknowledged in 5933413444.
- **Revision v1.3 (`0d6bc80`, NOTE only).** The xAI lane (Harper, Math-#222 comment 5932429158) found that the series' §5 table headed "band share of the birth mass" shows `F^(d)_{[0,1]}`, the share of birth heights in `B = [0,1]` (the same for every `K`), while it read as the band's share of the whole coefficient. §5 now labels that column as the height-window share and adds the C8 band's share of `c_{d,ref}`, `F^(d)_{[0,1]} G_{[1/2,2]}`. The script, `RESULTS.json`, the workflow, the rules and the thirteen mutants are byte-unchanged; no certified value changes (NOTE §10, `SOURCE_FILES.json` `revisions`). Announced in 5933413444.
- **Revision v1.2 (`b22ff4b`).**"""),
("workflow replayed locally at v1.2; hosted run 4/4 green at `b22ff4b` (`certify-and-reject-mutants`, `downstream-replay`, `formal / formal-evidence`, `math-downstream-gates`).",
 "workflow replayed locally at v1.2; at v1.3 the manifest and pins verify and the containment audit passes 11/11 NOTE intervals and 16 prefixes; hosted run 4/4 green at `b22ff4b` (`certify-and-reject-mutants`, `downstream-replay`, `formal / formal-evidence`, `math-downstream-gates`); the run on `0d6bc80` is in progress."),
]
