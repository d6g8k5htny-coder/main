[
("(v1). **Scientific effect NONE.**", "(v1.1). **Scientific effect NONE.**"),
("""The C8 band's share of the birth mass by dimension:

| `d` | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| band share | `48.3 %` | `34.6 %` | `15.4 %` | `4.37 %` | `0.800 %` |
""", """**How the band's share moves with dimension.**

| `d` | height-window share `F^(d)_{[0,1]}` (every `K`) | C8 band's share of `c_{d,ref}`, `F^(d)_{[0,1]} G_{[1/2,2]}` |
|---|---|---|
| 2 | `48.3 %` | `3.25 %` |
| 3 | `34.6 %` | `2.33 %` |
| 4 | `15.4 %` | `1.04 %` |
| 5 | `4.37 %` | `0.295 %` |
| 6 | **`0.800 %`** | **`0.0539 %`** |

The first column is the share of birth heights in `B = [0, 1]`. By the factorization `c_{B,K} = c_{d,ref} F_B G_K` it does not depend on the gap window. The second column is the C8 band's share of the whole coefficient, with the dimension-free gap factor `G_{[1/2,2]} = 6.74 %`. Both are ratios of certified values from the separate packets, not a theorem about birth mass (v1.1).
"""),
("""- It does not cover `d ≥ 7`. At `m = 6` a further layer follows, so the three Owen-type kinds would have to be integrated against another Gaussian unless they cancel there; the recursion as written refuses that case. The floor falls to `0.2064`.""",
 """- It does not cover `d ≥ 7`. At `m = 6` a further layer follows, so the three Owen-type kinds would have to be integrated against another Gaussian unless they cancel there; the recursion as written refuses that case. The floor falls to `0.2064`. The `d = 7` case is Math-#226 (claim 5931809506, delivered in 5933072558): there the nested kinds cancel identically."""),
("- **Head / owner:** `0c88a56` (v1) on", "- **Head / owner:** `b11b791` (v1.1) on"),
("""  - **Copilot review** (5379560534): one finding, thread 4155628647, "remove the workflow".""",
 """  - **Codex code review** at `0c88a56`: completed, no findings posted.
  - **Caption checks on the band-share table (head `0c88a56`):** comment 5932380111 (scoped, its lane not stated) and xAI / Grok (Harper) 5932429158 (NOT READY, not a P1). Both found that the §5 column is the height-window share `F^(6)_{[0,1]} = 0.800 %`, while the C8 band carries `0.0539 %` of `c_{6,ref}`. Resolved in v1.1 (reply 5933370340); the enclosures are not affected.
  - **Copilot review** (5379560534): one finding, thread 4155628647, "remove the workflow"."""),
("""- **Unresolved finding IDs:** none open on the mathematics.""",
 """- **Revision v1.1 (`b11b791`, NOTE only).** §5 labels the old column as the height-window share and adds the C8 band's share of `c_{d,ref}`, `F^(d)_{[0,1]} G_{[1/2,2]}`. The script, `RESULTS.json`, the workflow, the nineteen rules and the fourteen mutants are byte-unchanged, and no certified value changes (NOTE §10, `SOURCE_FILES.json` `revisions`). The head therefore moved from `0c88a56`, which the earlier note 5931809506 had said would stay frozen. A code read done at `0c88a56` applies to the same script bytes; please bind verdicts to `b11b791`. NOTE §10 credits both caption comments to the xAI lane. The lane of 5932380111 is not stated, so that attribution will be corrected in the next push that changes this PR's files.
- **Unresolved finding IDs:** none open on the mathematics."""),
("""containment audit 12/12 NOTE intervals and 18 prefixes; hosted run 4/4 green at `0c88a56` (`certify-and-reject-mutants` in 35 minutes, `downstream-replay`, `formal / formal-evidence`, `math-downstream-gates`).""",
 """containment audit 12/12 NOTE intervals and 18 prefixes (repeated at v1.1, with the manifest and pins); hosted run 4/4 green at `0c88a56` (`certify-and-reject-mutants` in 35 minutes, `downstream-replay`, `formal / formal-evidence`, `math-downstream-gates`); the run on `b11b791` is in progress."""),
("""- **Follow-up claim:** `d = 7` (the first `D_6` and `c_{7,ref}`), comment 5931809506, on a separate branch; this head stays frozen.""",
 """- **Follow-up:** the `d = 7` claim 5931809506 (the first `D_6` and `c_{7,ref}`) was delivered as Math-#226 (5933072558) on a separate branch. The delivery note's "band share of the birth mass `0.0941 %`" is the height-window share; the C8 band's share of `c_{7,ref}` is `0.00634 %` (corrected in 5933370340)."""),
]
