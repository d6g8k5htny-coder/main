import sys
from pathlib import Path
P = Path(sys.argv[1])
s = P.read_text()
E = []
def r(old, new):
    E.append((old, new))

# A1 paraphrase rewrites
r("| 0b | It presents the links as genuine connections and disclaims that they show the project proves RH. | C |",
  "| 0b | It describes the links as real while disclaiming any implication that the project bears on RH. | C |")
r("| 3.4 | Zeros move with t; real zeros approach and collide, and with time reversed they leave the axis. | I (direction) |",
  "| 3.4 | As t varies the zeros move; it describes real zeros meeting, and leaving the real axis when the flow runs the other way. | I (direction) |")
r("| 3.5 | (Introduction) The zeros of Riemann's xi function collide under a heat flow. | I |",
  "| 3.5 | (Introduction) Under a heat flow, zeros of ξ merge. | I |")
r("| 4.2 | This is the same Gaussian-lattice construction as Jacobi's theta function. | CC |",
  "| 4.2 | The project's normalizer is the lattice sum that defines Jacobi's theta function. | CC |")
r("| 6.1 | The proposed route ends at proving that the zeta heat flow has no positive-time transition, then Λ = 0, then RH. | CC → I | The logic is valid but circular: the absence of a positive-time transition ⇔ Λ ≤ 0 ⇔ RH. The step is equivalent to RH, not merely related to it. |",
  "| 6.1 | Its route would show that H_t has no real-zero merger for t > 0, deduce Λ = 0, and conclude RH. | CC → I | Circular: Λ ≤ 0 (⇔ RH) already implies no real-zero merger for t > 0, since all zeros of H_t are simple for t > Λ [CSV 1994; snippet]; the converse step needs 6.2. Granting 6.2, the step is equivalent to RH, not merely related to it. |")
r("| 6.4 | The fundamental obstacle: the project's laws are random while ζ is deterministic, and a local fold says nothing global. | C |",
  "| 6.4 | The main obstacle: the project's laws concern random fields while ζ is deterministic, and a local fold carries no global information. | C |")

# A16 (i) in CLAIMS 3.4 evidence
r("is truncation-limited at about 1.5·10⁻⁴ and rules out the opposite sign.",
  "is truncation-limited at about 10⁻⁴ and rules out the opposite sign.")

# A11: CLAIMS 3.13
r("F(x₋) − F(x₊) = (4/3)A(2τ)^{3/2} = A r³/6 with r = 2√(2τ); so",
  "F(x₋) − F(x₊) = (4/3)A(2τ)^{3/2} = A r³/6 with r = 2√(2τ) (bar pairing for small τ: README Proposition 1(c)); so")

# A2 (d): CLAIMS 5.4
r("zeta short bars are decisive against −1/3 and consistent with ≈ 0 given the finite-cutoff bias seen in controls. |",
  "zeta short bars are decisive against −1/3 and consistent with ≈ 0 given the finite-cutoff bias seen in controls. A pointwise ℓ ≈ \\|A\\|r³/6 is not enough for a density transfer; (D1) also needs a monotone C¹ lifetime map and control of the joint law (hypotheses 1–4). |")

# A3: CLAIMS 5.6
r("with a mild e^{0.01t} weight 112/113 and 104/106. These windows (~110 bars) cannot estimate an exponent. |",
  "with a mild e^{0.01t} weight 112/113 and 104/106. These windows (~110 bars) cannot estimate an exponent. Prior work on the same antiderivative (up to scaling): Lagarias–Montague, arXiv:1106.4348, on the zeros of the integral of ξ [snippet; statement to be checked]. Its zeros are not the critical points of F used here, so the barcode question is distinct. |")

# A1 consistency: errors list item 5
r("5. Its decisive step is equivalent to RH (6.1).",
  "5. Its decisive step is implied by RH and, granting 6.2, equivalent to it (6.1).")

# A3: citation table row
r("| Cerf 1970; Cohen-Steiner–Edelsbrunner–Morozov 2006 | birth–death, vineyards | memory |",
  "| Lagarias–Montague (arXiv:1106.4348) | prior work on the integral of ξ; its zeros are not the critical points used here | snippet; statement to be checked |\n| Cerf 1970; Cohen-Steiner–Edelsbrunner–Morozov 2006 | birth–death, vineyards | memory |")

bad = []
for old, new in E:
    c = s.count(old)
    if c != 1:
        bad.append((c, old[:90])); continue
    s = s.replace(old, new)
if bad:
    for c, o in bad: print("COUNT", c, "::", o)
    sys.exit(1)
P.write_text(s)
print("applied", len(E), "edits")
