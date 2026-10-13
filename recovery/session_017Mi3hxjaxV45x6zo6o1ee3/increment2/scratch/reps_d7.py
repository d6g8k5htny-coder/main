[
("(v1). **Scientific effect NONE.**", "(v1.1). **Scientific effect NONE.**"),
("""| band share of the birth mass | `48.3 %` | `34.6 %` | `15.4 %` | `4.37 %` | `0.800 %` | **`0.0941 %`** |
""", """| height-window share `F^(d)_{[0,1]}` (every `K`) | `48.3 %` | `34.6 %` | `15.4 %` | `4.37 %` | `0.800 %` | **`0.0941 %`** |
| C8 band's share of `c_{d,ref}`, `F^(d)_{[0,1]} G_{[1/2,2]}` | `3.25 %` | `2.33 %` | `1.04 %` | `0.295 %` | `0.0539 %` | **`0.00634 %`** |

The second row is the share of birth heights in `B = [0, 1]`. By the factorization `c_{B,K} = c_{d,ref} F_B G_K` it does not depend on the gap window. The third row is the C8 band's share of the whole coefficient, with the dimension-free gap factor `G_{[1/2,2]} = 6.74 %`. Both are ratios of certified values from the separate packets, not a theorem about birth mass (v1.1).
"""),
("""- It does not cover `d ≥ 8`. At `m = 7` one more layer (`c = 1/168`) follows. There the Owen-type kinds of `F_6`, including the products `e^{−T²/120}K`, would enter a further Gaussian layer, and the recursion as written refuses a product kind entering a layer.""",
 """- It does not cover `d ≥ 8`. At `m = 7` one more layer (`c = 1/168`) follows. There the Owen-type kinds of `F_6`, including the products `e^{−T²/120}K`, would enter a further Gaussian layer, and the recursion as written refuses a product kind entering a layer. An exploratory exact run at `m = 7`, posted as information only (5933276072, no claim), finds that four nested kinds survive that layer, so `d = 8` needs trivariate-orthant-type functions (a second-level cumulative layer)."""),
("- **Head / owner:** `176e993` (v1) on", "- **Head / owner:** `e6ba833` (v1.1) on"),
("""- **Completed review scopes:** none yet. Requested nonauthor reads from the OpenAI / xAI lanes (request to follow):""",
 """- **Completed review scopes:** Codex code review at `176e993`: completed, no findings posted. Requested nonauthor reads from the OpenAI / xAI lanes (request 5933070587, which binds to `e6ba833` per 5933494267):"""),
("""- **Unresolved finding IDs:** none.""",
 """- **Revision v1.1 (`0b29d62` and `e6ba833`, NOTE only).** The xAI lane (Harper, Math-#222 comment 5932429158) found that the series' "band share of the birth mass" shows `F^(d)_{[0,1]}`, the share of birth heights in `B = [0,1]` (the same for every `K`), while it read as the band's share of the whole coefficient. §5 now labels that row as the height-window share and adds a row with the C8 band's share of `c_{d,ref}`. `e6ba833` corrects `0b29d62`'s text, which called the rows columns. The script, `RESULTS.json`, the workflow, the twenty rules and the sixteen mutants are byte-unchanged, and no certified value changes (NOTE §10, `SOURCE_FILES.json` `revisions`; announced in 5933494267). No read had started.
- **Unresolved finding IDs:** none."""),
("""containment audit 12/12 NOTE intervals and 19 prefixes; hosted run pending at opening.""",
 """containment audit 12/12 NOTE intervals and 19 prefixes (repeated at v1.1, with the manifest and pins); the replay runs on `176e993` and `0b29d62` were cancelled as superseded (`downstream-replay` passed on both); the run on `e6ba833` is in progress (about 70 minutes for `certify-and-reject-mutants`)."""),
]
