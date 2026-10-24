import sys
from pathlib import Path

P = Path(sys.argv[1])
t = P.read_text(encoding="utf-8")


def rep(old, new, count=1):
    global t
    n = t.count(old)
    if n != count:
        raise SystemExit("expected %d occurrence(s), found %d: %r" % (count, n, old[:80]))
    t = t.replace(old, new)


# citation-tag legend: second-hand tag
rep("[memory]: recalled from memory, unchecked. [num]: re-checked numerically in this packet\n"
    "(mpmath 1.4.1, high precision, not interval-certified); the script is named.",
    "[memory]: recalled from memory, unchecked. [via main#307 comment 6116121839]: taken second-hand\n"
    "from that comment, not checked against the source. [num]: re-checked numerically in this\n"
    "packet (mpmath 1.4.1, high precision, not interval-certified); the script is named.")

# 2.3
rep("| 2.3 | Combining the cubic gap with the probability of configurations gives the project's exponent. |",
    "| 2.3 | Feeding the cubic gap through the pair-intensity law yields the project's exponent. |")
# 3.2
rep("| 3.2 | At t = 0, H_t is the completed zeta function in coordinates where RH means all zeros are real. |",
    "| 3.2 | At t = 0, H_t is a rescaled ξ, and in this variable RH says that H_0 has only real zeros. |")
# 3.14
rep("| 3.14 | This follows analytically near a nondegenerate double zero. |",
    "| 3.14 | The local fold picture of 3.11–3.13 holds whenever the double zero is nondegenerate. |")
# 4.5
rep("| 4.5 | Higher-dimensional lattice sums lead to Epstein zeta functions. |",
    "| 4.5 | Theta series over ℤ^d with d ≥ 2 give Epstein zeta functions through the same Mellin transform. |")
# 4.6
rep("| 4.6 | A functional equation alone does not locate zeros. |",
    "| 4.6 | A functional equation by itself does not say where the zeros lie. |")
# 4.8
rep("| 4.8 | The theta link is an analytic link, not a proof link. |",
    "| 4.8 | The theta connection is a correct analytic identity; it supplies no step toward any proof about the zeros. |")
# 5.3
rep("| 5.3 | GUE statistics for zeta zeros remain conjectural. |",
    "| 5.3 | That zeta zeros obey GUE statistics is unproved. |")
# 5.4 credit for the density-transfer caveat
rep("A pointwise ℓ ≈ \\|A\\|r³/6 is not enough for a density transfer; (D1) also needs a monotone C¹ lifetime map and control of the joint law (hypotheses 1–4). |",
    "A pointwise ℓ ≈ \\|A\\|r³/6 is not enough for a density transfer; (D1) also needs a monotone C¹ lifetime map and control of the joint law (hypotheses 1–4). This caveat is taken from main#307 comment 6116121839 (README §2). |")
# 5.6 Lagarias-Montague provenance
rep("Prior work on the same antiderivative (up to scaling): Lagarias–Montague, arXiv:1106.4348, on the zeros of the integral of ξ [snippet; statement to be checked].",
    "Prior work on the same antiderivative (up to scaling): Lagarias–Montague, arXiv:1106.4348, on the zeros of the integral of ξ [via main#307 comment 6116121839; statement to be checked].")
# 6.1
rep("| CC → I | Circular: Λ ≤ 0 (⇔ RH) already implies",
    "| CC → I | Not a reduction: Λ ≤ 0 (⇔ RH) already implies")
# 6.3
rep("| 6.3 | Real-axis persistence may miss zeros off the line. |",
    "| 6.3 | A persistence analysis confined to the real axis might overlook zeros off the line. |")
# 7(a)
rep("| 7(a) | Proposed study: persistence of F_t on finite intervals, tied to zero events. |",
    "| 7(a) | Proposed study: barcodes of the antiderivative F_t over bounded windows, and how their changes match zero events of H_t. |")
# 7(b)
rep("| 7(b) | Proposed study: invariants or global theta constraints on H_t. |",
    "| 7(b) | Proposed study: what the heat flow preserves, and whether the theta representation imposes a global restriction on zero positions. |")
rep("The Rodgers–Tao energy argument is a technique for the lower bound, not a global constraint ruling out configurations for t > 0. |",
    "The Rodgers–Tao energy argument is a technique for the lower bound; it does not forbid any particular arrangement of zeros for t > 0. |")
# 7-table
rep("| 7-table | Summary table: lattice ↔ theta ↔ ζ exact; folds ↔ de Bruijn–Newman exact locally; persistence ↔ RH not established; the exponent does not explain the critical line. |",
    "| 7-table | A closing summary grades each link: theta link exact and classical, fold link exact only locally, no established persistence route to RH, and no role for the exponent in where the zeros lie. |")
# errors list item 9
rep("9. The global constraints it asks for already exist (7(b)).",
    "9. Global restrictions of the kind it proposes looking for already exist (7(b)).")
# citation table
rep("| Lagarias–Montague (arXiv:1106.4348) | prior work on the integral of ξ; its zeros are not the critical points used here | snippet; statement to be checked |",
    "| Lagarias–Montague (arXiv:1106.4348) | prior work on the integral of ξ; its zeros are not the critical points used here | via main#307 comment 6116121839 (second-hand); statement to be checked |")
rep("A nonauthor reviewer with working web access should confirm every snippet and memory item\n(README §9, slice B).",
    "A nonauthor reviewer with working web access should confirm every snippet and memory item,\nand the second-hand Lagarias–Montague item (README §9, slice B).")

P.write_text(t, encoding="utf-8")
print("ok")
