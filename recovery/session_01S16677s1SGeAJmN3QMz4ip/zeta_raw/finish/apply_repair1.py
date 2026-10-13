"""Throwaway: apply repair item 1 (ZF-AD-02/03 wording) to the local PR body copies and the
successor commit-message template. Asserts the old lines before replacing. Not part of the packet.

Usage: python3 -I apply_repair1.py FINISH_DIR
"""
import sys
from pathlib import Path

D = Path(sys.argv[1])

OLD_12 = ("  - Near a nondegenerate double zero, r² = 8τ and the antiderivative's lobe area is "
          "|A|r³/6, the project's k = F‴/12 fold law.")
NEW_12 = ("  - Near a nondegenerate double zero, r² = 8τ(1 + O(τ)) and the antiderivative's lobe "
          "area is (|A|/6) r³ (1 + O(τ)), the project's k = F‴/12 fold law. For small τ, on a "
          "fixed surrounding interval, it is a finite superlevel bar.")
OLD_16_18 = [
    "  - With the same cubic fold, r² pair repulsion (CUE; conjecturally ζ) moves the short-bar "
    "density exponent from −1/3 to about 0, as the project's transfer rule predicts.",
    "  - ζ on t ∈ [10⁵, 10⁶] gives −0.020 [−0.041, +0.001] (sampling-only), measured on "
    "antiderivatives of Hardy's Z over 1.6M zeros.",
    "  - That is decisive against −1/3 at these heights, and consistent with about 0 given the "
    "control bias. On the controls, the same estimator lands −0.017 to −0.065 from the known "
    "values.",
]
NEW_16_18 = [
    "  - Under the hypotheses of the project's transfer theorem (D1), the same cubic fold with r² "
    "pair repulsion (CUE; conjecturally ζ) gives short-bar density exponent 0 instead of −1/3. "
    "Both values are theoretical expectations under stated conditions, not calibrated answers "
    "for the finite controls (the fixed-N CUE bar law is the unproved T-CUE target). Pair "
    "correlation is neither a nearest-neighbour law nor a persistence-pair law, so reaching ζ's "
    "bars needs further assumptions.",
    "  - For the specified Hardy-Z primitive (the antiderivative of Z(t)/√(log(t/2π)) in the "
    "unfolded variable, endpoint-born bars dropped), on t ∈ [10⁵, 10⁶] with about 1.6M zeros, "
    "the Hill estimate at cut 0.01 is −0.020 [−0.041, +0.001] (sampling-only). The finite-tail "
    "estimates are substantially above −1/3 and near 0; sampling-only block-bootstrap intervals "
    "exclude −1/3 at the reported cutoffs.",
    "  - This is exploratory evidence about that finite-window statistic. It does not establish a "
    "limiting density exponent or a calibrated rejection accounting for all errors; the controls "
    "do not calibrate those errors either (at cut 0.01 they sit −0.017 to −0.065 from their "
    "theoretical expectations). It says nothing about the ξ-weighted primitive or a "
    "de Bruijn–Newman barcode.",
]
LINE_19 = ("  - The data were inspected before any prediction was frozen, so this is exploration, "
           "not confirmation.")


def fix_body(path):
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    assert lines[11] == OLD_12, (path, lines[11])
    assert lines[15:18] == OLD_16_18, (path, lines[15:18])
    assert lines[18] == LINE_19, (path, lines[18])
    lines[11] = NEW_12
    lines[15:18] = NEW_16_18
    path.write_text("\n".join(lines), encoding="utf-8")


OLD_C10_11 = [
    "  collisions are computed to about 40 digits, both at t < 0. The fold law",
    "  r^2 = 8 tau and lobe area |A| r^3 / 6 hold near the collision.",
]
NEW_C10_11 = [
    "  collisions are computed to about 40 digits, both at t < 0. The fold law",
    "  r^2 = 8 tau (1 + O(tau)) and lobe area (|A|/6) r^3 (1 + O(tau)) hold",
    "  near the collision.",
]
OLD_C15_19 = [
    "- stats/: Riemann-Siegel zeros, Gaussian and CUE controls, barcodes and",
    "  exponent estimates. Short-bar density exponent for zeta: -0.020",
    "  [-0.041, +0.001] (sampling-only). Decisive against -1/3 at the heights",
    "  tested; consistent with about 0 given the estimator's control bias.",
    "  The data were inspected first, so this is exploration, not confirmation.",
]
NEW_C15_19 = [
    "- stats/: Riemann-Siegel zeros, Gaussian and CUE controls, barcodes and",
    "  finite-tail exponent estimates. For the Hardy-Z primitive (t in",
    "  [1e5, 1e6], cut 0.01) the Hill estimate of the short-bar density",
    "  exponent is -0.020 [-0.041, +0.001] (sampling-only). Finite-tail",
    "  estimates for the Hardy-Z primitive are substantially above -1/3 and",
    "  near 0; sampling-only intervals exclude -1/3; no limiting exponent or",
    "  calibrated total-error rejection. -1/3 and 0 are theoretical",
    "  expectations under stated conditions, not calibrated control answers.",
    "  The data were inspected first, so this is exploration, not confirmation.",
]


def fix_commit(path):
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    assert lines[9:11] == OLD_C10_11, lines[9:11]
    assert lines[14:19] == OLD_C15_19, lines[14:19]
    # replace the later block first so earlier indices stay valid
    lines[14:19] = NEW_C15_19
    lines[9:11] = NEW_C10_11
    path.write_text("\n".join(lines), encoding="utf-8")


for name in ("pr_body.md", "pr_body_live.md"):
    fix_body(D / name)
    print("fixed", name)
fix_commit(D / "commit_msg.txt")
print("fixed commit_msg.txt")
