# Hessian jet at the pins, Bargmann–Fock, every r>0

**Object:** GROK-HEAVY-HESSIAN-JET-20260926-v1
**Scientific effect:** NONE. Not A3. Not D5. Not File-3 (ND).

## Already closed last round

`g = f_{ss}+f` is orthogonal to the six pair pins on the fold axis. Hence `f_{ss}(M)\sim N(-b,2)` exactly.

Algebra: `C(d)=C_1(d_t)C_2(d_s)` and `(\partial_{ss}+1)C_2|_{d_s=0}=0`. Any `t`-derivative of that product still vanishes on the axis.

## Joint law given six pins (numerical + pattern)

At `M`, `b=0`, `k=1`:

| r | E[f_{tt}]/r | Var(f_{tt}) | Var(f_{ts}) | Var(f_{ss}) |
|---|---|---|---|---|
| 0.40 | -5.841 | 4.13e-3 | 0.07787 | 2 |
| 0.20 | -5.960 | 2.65e-4 | 0.01987 | 2 |
| 0.10 | -5.990 | 1.70e-5 | 0.00499 | 2 |
| 0.05 | -5.998 | 1.00e-6 | 0.00125 | 2 |

Patterns:
- `E[f_{tt}(M)]/r \to -6k` with quartic error (parent 5.2).
- `Var(f_{ss})=2` exact; `f_{ss}` uncorrelated with `(f_{tt},f_{ts})`.
- `Var(f_{ts}) \sim r^2/2`.
- `Cov(f_{ss}(M),f_{ss}(S)\mid\mathrm{pins}) = 2\,e^{-r^2/2}`. Correlation `e^{-r^2/2}\to 1`.

So in the contact limit there is **one** transverse curvature, not two.

`|\det H_M|/r = |\alpha A - r \beta^2|` with `A\sim N(-b,2)` exact, `\alpha\to-6k`, `\beta=O_p(r)`.

## File 5 (5.4), exact

`\ell=\kappa r^3/6` gives

    r\,dr = (1/3)(6/\kappa)^{2/3} u^{-1/3}\,du.

Hence `(1/3)6^{2/3}=2\cdot 6^{-1/3}\approx 1.100642`. This is the change-of-variable that produces `\nu(\ell)\sim C_*\ell^{-1/3}` **given** a pair measure `r\,dr` and `q\to 1`. Still PROVEN-MODULO (ND) because `q\to 1` is Theorem A.

## D5 recon minor

`q^4\bigl((p-1/2)^2+q^2/4\bigr)` vanishes to order 4 on the axis `q=0`. Polynomial identity, not `(D5-\mu)`.
