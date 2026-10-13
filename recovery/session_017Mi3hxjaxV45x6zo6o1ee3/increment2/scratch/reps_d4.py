[
("(v1.2). **Scientific effect NONE.**", "(v1.3). **Scientific effect NONE.**"),
("""The C8 band carries `15.4 %` of the `d = 4` birth mass:

| `d` | band share of the birth mass |
|---|---|
| 2 | `48.3 %` |
| 3 | `34.6 %` |
| 4 | `15.4 %` |
""", """**How the band's share moves with dimension.**

| `d` | height-window share `F^(d)_{[0,1]}` (every `K`) | C8 band's share of `c_{d,ref}`, `F^(d)_{[0,1]} G_{[1/2,2]}` |
|---|---|---|
| 2 | `48.3 %` | `3.25 %` |
| 3 | `34.6 %` | `2.33 %` |
| 4 | **`15.4 %`** | **`1.04 %`** |

The first column is the share of birth heights in `B = [0, 1]`. By the factorization `c_{B,K} = c_{d,ref} F_B G_K` it does not depend on the gap window. The second column is the C8 band's share of the whole coefficient, with the dimension-free gap factor `G_{[1/2,2]} = 6.74 %`. Both are ratios of certified values from the separate packets, not a theorem about birth mass (v1.3).
"""),
("- **Head / owner:** `84a494a` (v1.2) on", "- **Head / owner:** `c78dd10` (v1.3) on"),
("""  **xAI / Grok status (Harper, comment 5931348433, head `84a494a`):** NOT READY. Unanimous READY is not present, and Slices A–C are still requested. No finding was raised; the quadrature was not re-executed.
""", """  **xAI / Grok status (Harper, comment 5931348433, head `84a494a`):** NOT READY. Unanimous READY is not present, and Slices A–C are still requested. No finding was raised; the quadrature was not re-executed.

  **Caption check (comment 5932367935, head `84a494a`; scoped, not an approval):** found that the printed `15.4 %` is the band over the `K = [1/2,2]` birth mass (`1.04 %` over `c_{4,ref}`), and that the floor `(7 − √37)/3 = 0.30575` lies in `(3/10, 1/3)`. The caption is resolved in v1.3 (reply 5933373120).
- **Revision v1.3 (`c78dd10`, NOTE only).** The xAI lane found (comment 5932367935 here; Harper, Math-#222 comment 5932429158) that the §5 table headed "band share of the birth mass" shows `F^(d)_{[0,1]}`, the share of birth heights in `B = [0,1]` (the same for every `K`), while it read as the band's share of the whole coefficient. §5 now labels that column as the height-window share and adds the C8 band's share of `c_{d,ref}`, `F^(d)_{[0,1]} G_{[1/2,2]}`. The script, `RESULTS.json`, the workflow, the rules and the mutants are byte-unchanged; no certified value changes (NOTE §10, `SOURCE_FILES.json` `revisions`).
"""),
("""  - workflow replayed locally at v1.2;
  - hosted run 4/4 green at `84a494a` (`certify-and-reject-mutants`, `downstream-replay`, `formal / formal-evidence`, `math-downstream-gates`).
""", """  - workflow replayed locally at v1.2; at v1.3 the manifest and pins verify, and the containment audit passes 11/11 NOTE intervals and 17 prefixes;
  - hosted run 4/4 green at `84a494a` (`certify-and-reject-mutants`, `downstream-replay`, `formal / formal-evidence`, `math-downstream-gates`); the run on `c78dd10` is in progress.
"""),
]
