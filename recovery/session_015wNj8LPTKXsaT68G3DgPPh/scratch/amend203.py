import pathlib, sys
root = pathlib.Path('.')
def sub(path, pairs):
    p = root / path
    t = p.read_text()
    for old, new in pairs:
        n = t.count(old)
        if n != 1:
            sys.exit(f'{path}: expected one occurrence, found {n}: {old[:80]!r}')
        t = t.replace(old, new)
    p.write_text(t)

sub('NOTE.md', [
("# The cap-failure probability of Theorem A has an exact leading coefficient in the plane",
 "# The cap-failure probability in the proof of Theorem A has an exact leading coefficient in the plane"),
("`C` nor `r_*`; it determines the exact size of the quantity that `C r^3` bounds, hence a lower bound for every admissible `C`.",
 "`C` nor `r_*`; it determines the exact size of the cap-failure probability `Q^W(G_r^c)` that the proof of Theorem A bounds\n"
 "([LP] (7.8)), hence a lower bound for every cap-route constant `C_cap`; it gives no lower bound for the constant of Theorem A\n"
 "itself, which bounds the smaller probability `1 - p_r` (v1.2; section 5, item 1)."),
("(on the typed support `-f_yy(M) = lambda_min(-A_M)`). Theorem A of [LP] is `Q^W(G_r^c) <= C r^3` for `0 < r <= r_*`.",
 "(on the typed support `-f_yy(M) = lambda_min(-A_M)`). Theorem A of [LP] is `0 <= 1 - p_r <= C r^3` for `0 < r <= r_*`\n"
 "([LP] (1.1); `p_r` is the `Q^W` probability that the global ordinary elder death partner of `M` is `S`). Its proof establishes\n"
 "the stronger cap bound [LP] (7.8), `Q^W(G_r^c) <= C_3 r^3 + C_4 r^4 <= C r^3`, and deduces (1.1) from `1 - p_r <= Q^W(G_r^c)`\n"
 "([LP] section 8: on `G_r` the global elder partner is `S`). A **cap-route constant** is any `C_cap` with\n"
 "`Q^W(G_r^c) <= C_cap r^3` for `0 < r <= r_*`; every cap-route constant is a constant for [LP] (1.1), not conversely."),
("**Corollary (what this says about `C` and `r_*`).** Every constant `C` for which [LP] Theorem A holds at `(b, k)` (for a\n"
 "given kernel and frame) satisfies `C >= c_G(b, k)` for that kernel and frame. The certified numbers are the reference-kernel",
 "**Corollary (what this says about `C_cap` and `r_*`).** Every cap-route constant `C_cap` at `(b, k)` (for a given kernel and\n"
 "frame) satisfies `C_cap >= c_G(b, k)` for that kernel and frame. This is not a lower bound for every constant `C` of [LP] (1.1):\n"
 "the only relation between the two events is `1 - p_r <= Q^W(G_r^c)`, and the smaller probability has the smaller coefficient\n"
 "`alpha_1 + alpha_2` of [S] (S2) (Remark (d); section 5, items 1 and 4). The certified numbers are the reference-kernel"),
("reference-kernel statement (equivalently `L -> infinity`): every band containing `(0, 2)` needs `C > 5.3 x 10^6`, and the\n"
 "cap route cannot give a nontrivial bound (`C r^3 < 1`) above `r = c_G^{-1/3}`, which is `5.7 x 10^-3` at `(0, 2)`,",
 "reference-kernel statement (equivalently `L -> infinity`): every band containing `(0, 2)` needs `C_cap > 5.3 x 10^6`, and the\n"
 "cap route cannot give a nontrivial bound (`C_cap r^3 < 1`) above `r = c_G^{-1/3}`, which is `5.7 x 10^-3` at `(0, 2)`,"),
("of `c_G^{ref}`, up to that unenclosed deviation. The recorded constant of [CAP] (2)–(3), `C3_new + C4_new / 20 < 2.4 x 10^23` at `(6/5, 1/6)`, exceeds the sharp\n"
 "coefficient there by a factor above `6 x 10^18`.",
 "of `c_G^{ref}`, up to that unenclosed deviation. The recorded cap-route constant of [CAP] (2)–(3), `C3_new + C4_new / 20 < 2.4 x 10^23`\n"
 "at `(6/5, 1/6)`, exceeds the sharp cap coefficient there by a factor above `6 x 10^18`."),
("whose `du`-integral is at most `C N_r^{10}`, with `sup_r E N_r^{10} < infinity` by 2.1. Dominated convergence (in `u` and on the\n"
 "probability space) gives (2.3), where the limit field is the contact field given `f_yy(0) = 0` and `E` is over the law of 2.2.",
 "whose `du`-integral is at most `C N_r^{10}`, with `sup_r E N_r^{10} < infinity` by 2.1. Explicitly, taking the expectation\n"
 "first: at fixed `u` the expected integrand is at most `C (1 + u)^2 min(E N_r^4, (C/u)^4 E N_r^{12})` (Markov on\n"
 "`{N_r^2 >= u/C}`), which is bounded uniformly in `r <= 1` by the integrable function `C' (1 + u)^2 min(1, u^-4)`\n"
 "(`sup_r E N_r^{12} < infinity` by 2.1); and at fixed `u` the integrand converges almost surely under the coupling and is\n"
 "uniformly integrable (`sup_r E N_r^8 < infinity`). Vitali's theorem at fixed `u` and dominated convergence in `u` give (2.3),\n"
 "where the limit field is the contact field given `f_yy(0) = 0` and `E` is over the law of 2.2."),
("([LP] section 8), so `c_G >= alpha_1 + alpha_2` necessarily; the size of the gap is in section 5.",
 "([LP] section 8), so `c_G >= alpha_1 + alpha_2` necessarily; the size of the gap is in section 5. Hence `c_G` bounds\n"
 "cap-route constants only: a constant for [LP] (1.1) itself is bounded below by `alpha_1 + alpha_2` where (S2) holds, not by `c_G`."),
("## 5. Consequences for `C`, `r_*`, and the size of the cap criterion's slack",
 "## 5. Consequences for `C_cap`, `r_*`, and the size of the cap criterion's slack"),
("1. **Lower bound for every admissible `C`.** If `Q^W(G_r^c) <= C r^3` for all small `r` at `(b, k)`, then `C >= c_G(b, k)`\n"
 "   (for the kernel and frame in question). On the band of Math-#195 (`B = [0, 1]`, `K = [1/2, 2]`) the grid maximum of\n"
 "   `c_G^{ref}` is at `(0, 2)`: `C > 5.32 x 10^6` for the reference kernel.",
 "1. **Lower bound for every cap-route constant `C_cap`.** If `Q^W(G_r^c) <= C_cap r^3` for all small `r` at `(b, k)`, then\n"
 "   `C_cap >= c_G(b, k)` (for the kernel and frame in question). This does not bound a constant `C` of [LP] (1.1) itself: the\n"
 "   only relation is `1 - p_r <= Q^W(G_r^c)`, and a coefficient of the larger probability does not bound the constants of the\n"
 "   smaller one (logical countermodel of review 5372981851 on Math-#203: one uniform `U`, `F_r = {U <= r^3}` inside\n"
 "   `G_r^c = {U <= 2 r^3}`; the cap coefficient is 2 while `C = 1` is admissible for `F_r`). On the band of Math-#195\n"
 "   (`B = [0, 1]`, `K = [1/2, 2]`) the grid maximum of `c_G^{ref}` is at `(0, 2)`: `C_cap > 5.32 x 10^6` for the reference kernel."),
("2. **Where the cap route can be nontrivial.** `C r^3 < 1` forces `r < c_G(b,k)^{-1/3}`: `5.7 x 10^-3` at `(0, 2)`,",
 "2. **Where the cap route can be nontrivial.** `C_cap r^3 < 1` forces `r < c_G(b,k)^{-1/3}`: `5.7 x 10^-3` at `(0, 2)`,"),
("   for which Theorem A's bound is informative on the band is below `6 x 10^-3` at the band's corner, whatever the proof.",
 "   for which the cap bound [LP] (7.8) is informative on the band is below `6 x 10^-3` at the band's corner, whatever the proof\n"
 "   of (7.8). A proof of [LP] (1.1) that does not pass through `Q^W(G_r^c)` is not constrained by this item."),
("   a factor above `6 x 10^18` between the recorded and the sharp coefficient.",
 "   a factor above `6 x 10^18` between the recorded cap-route constant and the sharp cap coefficient."),
("bound `C` and no `r_*` are supplied (C8 stays OPEN on both).",
 "bound `C` or `C_cap` and no `r_*` are supplied (C8 stays OPEN on both), and no lower bound for the constant of [LP] (1.1)\n"
 "itself (section 5, item 1)."),
])

sub('README.md', [
("# The cap-failure coefficient of Theorem A in the plane (C8: what any `C` must exceed)",
 "# The cap-failure coefficient in the plane (C8: what any cap-route constant `C_cap` must exceed)"),
("[LP] section 7 / [CAP] (1), Theorem G (`NOTE.md`) gives the exact leading behaviour of the quantity that Theorem A bounds:",
 "[LP] section 7 / [CAP] (1), Theorem G (`NOTE.md`) gives the exact leading behaviour of the cap-failure probability that the\n"
 "proof of Theorem A bounds ([LP] (7.8)):"),
("the finite-torus jet-law deviation, of order `L^6 e^{-L^2/2}`, is not enclosed). Reference-kernel consequences: every\n"
 "admissible `C` at `(b, k)` is at least `c_G(b, k)`; on the Math-#195 band `C > 5.3 x 10^6`; the cap route is informative only",
 "the finite-torus jet-law deviation, of order `L^6 e^{-L^2/2}`, is not enclosed). Reference-kernel consequences: every\n"
 "cap-route constant `C_cap` (any constant with `Q^W(G_r^c) <= C_cap r^3`) at `(b, k)` is at least `c_G(b, k)`; on the\n"
 "Math-#195 band `C_cap > 5.3 x 10^6` (not a lower bound for the constant of Theorem A's `1 - p_r <= C r^3`, which bounds a\n"
 "smaller probability); the cap route is informative only"),
("below `r = c_G^{-1/3}` (`6 x 10^-3` at the band's corner `(0, 2)`); the recorded constant `2.4 x 10^23` of [CAP] at",
 "below `r = c_G^{-1/3}` (`6 x 10^-3` at the band's corner `(0, 2)`); the recorded cap-route constant `2.4 x 10^23` of [CAP] at"),
])

sub('SOURCE_MAP.json', [
("Supplies neither C nor r_*: C8 stays OPEN; it shows every admissible C on the Math-#195 band exceeds 5.3e6 and locates the cap route's informative range below r = c_G^(-1/3).",
 "Supplies neither C nor r_*: C8 stays OPEN; it shows every cap-route constant C_cap (Q^W(G_r^c) <= C_cap r^3, the bound [LP] (7.8)) on the Math-#195 band exceeds 5.3e6 and locates the cap route's informative range below r = c_G^(-1/3); it gives no lower bound for the constant of [LP] (1.1), which bounds the smaller probability 1 - p_r (v1.2, review 5372981851)."),
("none; this packet supplies a lower bound for every admissible C and no upper bound",
 "none; this packet supplies a lower bound for every cap-route constant C_cap (not for the constant of (1.1)) and no upper bound"),
])
print('ok')
