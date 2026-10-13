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


C = "main#307 comment 6116121839"

# version label
rep("draft v1 (v0 plus the two internal review passes, §10).",
    "draft v1 (v0 plus the same-provider review and check passes, §10).")

# section 1 tag legend
rep("**[memory]** means recalled from memory and unchecked. A full list is at the end of this\nsection.",
    "**[memory]** means recalled from memory and unchecked; **[via %s]** means taken\n"
    "second-hand from that comment (§2) and not checked against the source. A full list is at the\n"
    "end of this section." % C)

# Rodgers-Tao sentence (verbatim finding, README:134)
rep("The Rodgers–Tao energy argument is a technique for\n"
    "    proving the lower bound; it is not a global constraint that rules out configurations for\n"
    "    t > 0.",
    "The Rodgers–Tao energy argument is a technique for\n"
    "    proving the lower bound; it does not forbid any particular arrangement of zeros for t > 0.")

# section 1.3 bullet lead
rep("- **A theta functional equation does not locate zeros.** Z_4 has zeros",
    "- **Theta structure plus a symmetry does not place zeros on the symmetry line.** Z_4 has zeros")

# section 1 provenance: move Lagarias-Montague out of the snippet list
rep("  Montgomery 1973, Rudnick–Sarnak 1996, Davenport–Heilbronn 1936, Newman–Wu 2020,\n"
    "  Lagarias–Montague (arXiv:1106.4348, the integral of ξ; statement to be checked), and the\n"
    "  historical table of Λ lower bounds including the −0.385 entry attributed to\n"
    "  Norfolk–Ruttan–Varga 1992.\n",
    "  Montgomery 1973, Rudnick–Sarnak 1996, Davenport–Heilbronn 1936, Newman–Wu 2020, and the\n"
    "  historical table of Λ lower bounds including the −0.385 entry attributed to\n"
    "  Norfolk–Ruttan–Varga 1992.\n"
    "- *Second-hand, via %s (no snippet seen):* Lagarias–Montague\n"
    "  (arXiv:1106.4348, the integral of ξ; statement to be checked).\n" % C)

# section 2: related open proposal
rep("  (an OpenAI lane, 12 October 2026) proposes polynomial-first deterministic heat-flow fold\n"
    "  controls with finite-interval H0 in `experiments/universality`. This packet does not take\n"
    "  that scope; it touches only `experiments/zeta_fold_exploration/**`.\n",
    "  (an OpenAI lane, 12 October 2026) proposes polynomial-first deterministic heat-flow fold\n"
    "  controls with finite-interval H0 within the universality program (no path named). This\n"
    "  packet does not take that scope; it touches only `experiments/zeta_fold_exploration/**`.\n"
    "  The triple-collision example (§3 Remark 1), the gap-versus-bar countercontrol (§5.2) and\n"
    "  the information-loss example (§7) are taken from that comment and re-checked by hand here\n"
    "  in exact rational arithmetic. The density-transfer caveat (`CLAIMS.md` 5.4) and the\n"
    "  Lagarias–Montague pointer (§4 item 3) are also taken from it; the latter is unchecked. No\n"
    "  reproducible controls are built.\n")

# section 3 remark 1: credit
rep("   zeros r apart, and is not covered.\n",
    "   zeros r apart, and is not covered (example taken from %s, §2).\n" % C)

# section 4 item 3: Lagarias-Montague tag
rep("   on the zeros of the integral of ξ [snippet; statement to be checked]. Its zeros are not the\n"
    "   critical points of F used here, so the barcode question is distinct.\n",
    "   on the zeros of the integral of ξ [via %s; statement to be checked]. Its\n"
    "   zeros are not the critical points of F used here, so the barcode question is distinct.\n" % C)

# section 5.2 footnote: credit
rep("need not be a finite bar: on [−3/2, 3/2], x³/3 − x has critical-value gap 4/3, but its only\n"
    "finite H0 bar, born at the right end, has length 7/24.\n",
    "need not be a finite bar: on [−3/2, 3/2], x³/3 − x has critical-value gap 4/3, but its only\n"
    "finite superlevel H0 bar, born at the right end, has length 7/24 (countercontrol taken from\n"
    "%s, §2).\n" % C)

# section 7: exponent bullet
rep("- **That −1/3 explains or relates to the critical line.** It does not, and the data contradict\n"
    "  it for ζ: the zeta short-bar density exponent is about 0, not −1/3.\n",
    "- **That −1/3 has any bearing on where ζ's zeros lie.** It has none. Nor does −1/3 describe\n"
    "  ζ's short bars: the zeta short-bar density exponent is about 0.\n")

# section 7: theta constraints bullet
rep("- **Hunting for new global theta constraints on H_t.** The constraints the note asks for exist\n"
    "  already (de Bruijn's strip lemma, Newman, Ki–Kim–Lee, Rodgers–Tao; §1.2), and the question is\n"
    "  outside the project's competence and mission.\n",
    "- **Looking for new theta-derived restrictions on the zeros of H_t.** Restrictions of the kind\n"
    "  the note proposes exist already (de Bruijn's strip lemma, Newman, Ki–Kim–Lee, Rodgers–Tao;\n"
    "  §1.2), and the question is outside the project's competence and mission.\n")

# section 7: information-loss bullet
rep("  is verified numerically. Real-axis barcodes alone discard this information: H = 1 and\n"
    "  H = z² + 1 give antiderivatives z and z³/3 + z, both with no real critical point and the\n"
    "  same barcode (no finite bar), yet the second has zeros ±i.\n",
    "  is verified numerically. Real-axis barcodes alone discard this information: on [−L, L],\n"
    "  H = 1 + L²/3 and H = z² + 1 have increasing antiderivatives, (1 + L²/3)z and z³/3 + z, with\n"
    "  the same endpoint values, hence identical barcodes (one essential bar, no finite bar), yet\n"
    "  H = z² + 1 has zeros ±i (example taken from %s, §2).\n" % C)

# section 8: fold_controls producer
rep("- `theta/check_theta.py` and `stats/check_gue_transfer.py` (seconds),\n"
    "  `theta/project_theta_checks.py` (~7 min).\n",
    "- `theta/check_theta.py` and `stats/check_gue_transfer.py` (seconds),\n"
    "  `theta/project_theta_checks.py` (~7 min).\n"
    "- `python3 -B experiments/universality/fold_controls.py`, run from the repository root\n"
    "  (seconds; a repository script outside this packet); its stdout is\n"
    "  `results/theta/fold_controls_rerun.out`.\n")
rep("`count_check` (window [1, 103] at t = −1.89), `check_gue_transfer`, `analyze`, `matched_tail`,\n"
    "`make_plots`, `check_zeta_quad`, `barcode_bruteforce`, `gauss_hidden`, `localamp_baseline`,\n"
    "`rs_spot_check`.",
    "`count_check` (window [1, 103] at t = −1.89), `check_gue_transfer`, `analyze`, `matched_tail`,\n"
    "`make_plots`, `check_zeta_quad`, `barcode_bruteforce`, `gauss_hidden`, `localamp_baseline`,\n"
    "`rs_spot_check`, `fold_controls_rerun`.")

# section 9
rep("  confirming every item marked [snippet] or [memory], including the statement of\n"
    "  Lagarias–Montague (arXiv:1106.4348).\n",
    "  confirming every item marked [snippet] or [memory], and the statement of Lagarias–Montague\n"
    "  (arXiv:1106.4348), which this packet has only second-hand via %s.\n" % C)
rep("The two review passes already applied (§10) were Anthropic Claude reading Anthropic Claude\n"
    "work. They earn no organizational-independence credit.\n",
    "The review passes and checks already applied (§10) were all Anthropic Claude reading Anthropic\n"
    "Claude work. They earn no organizational-independence credit.\n")

# section 10
rep("| N12 | Tracker completeness; monotonicity (info) | §5.2 |\n",
    "| N12 | Tracker completeness; monotonicity (info) | §5.2 |\n"
    "\n"
    "Two further same-provider checks (accuracy, 16 findings A1–A16; reproducibility, 6 findings\n"
    "R1–R6) were then applied, followed by a third same-provider round (accuracy, wording overlap\n"
    "with the note, and reproducibility/CI; 18 findings, some overlapping). None of these earns\n"
    "independence credit.\n")

P.write_text(t, encoding="utf-8")
print("ok")
