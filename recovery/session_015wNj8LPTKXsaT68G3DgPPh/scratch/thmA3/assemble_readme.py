D='/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/thmA3'
P='/home/user/wt_d3/frontiers/c8_theorem_a_constants_d3_20261001'
s=open(D+'/README_draft.md').read()
table='''| `r_*` | `C3(r_*)` | `C3 / c_G^(3)(0,2)` | `C3 r_*^3` | `z3(r_*)` |
|---|---|---|---|---|
| `1/4096` | `22754801.7964` | `1.0148` | `3.4e-4` | `6.0210` |
| `1/2048` | `22824067.0244` | `1.0179` | `2.7e-3` | `5.9980` |
| `1/1024` | `22974134.2920` | `1.0246` | `0.0214` | `5.9517` |
| `1/512` | `23332922.4752` | `1.0406` | `0.174` | `5.8590` |

(`1/256`, `1/128` and `1/64` are also certified, with `C3 r_*^3 > 1`; every cap-route constant has `C r^3 > 1` at
`r = 1/256` in `d = 3`.)'''
s=s.replace('@@TABLE@@', table)
s=s.replace('@@PAIR@@', '`(C, r_*) = (23332922.4752, 1/512)` (the bound at `r = r_*` is below `0.174`) or `(22974134.2920, 1/1024)`')
s=s.replace('@@NEAROPT@@', 'The constants are within `1.5%`–`4.1%` of the smallest possible `d = 3` cap-route constant for `r_* <= 1/512`.')
s=s.replace('@@Z4096@@', '6.0210')
assert '@@' not in s
open(P+'/README.md','w').write(s)
print(len(s.encode()))
