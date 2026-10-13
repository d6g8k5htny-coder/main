import json, sys
H=sys.argv[1]; P199=sys.argv[2]; P200=sys.argv[3]; P201=sys.argv[4]; P202=sys.argv[5]
b=open('body204.md').read()
def rep(o,n):
    global b
    assert b.count(o)==1,o[:60]; b=b.replace(o,n)
rep("the reference coefficients are now supplied for every `d ≤ 6`.",
    "author-side reference coefficients are now recorded for every d ≤ 6 (d = 4, 5 through companions not yet on main).")
rep("- `D_4` re-enclosed through the certified layer: width `4e-40`, and the enclosure contains the closed form.",
    "- `D_4` re-enclosed through the certified layer: width `4e-43`, and the enclosure contains the closed form.")
rep("- **Companions (not on main):** Math-#200 at `33d50de`, Math-#201 at `3abcc10`, Math-#202 at `4512309`.",
    f"- **Companions (not on main):** Math-#199 at `{P199}` (§3 tail bounds), Math-#200 at `{P200}`, Math-#201 at `{P201}`, Math-#202 at `{P202}`.")
rep("- **Head / owner:** `5915af4` on",f"- **Head / owner:** `{H}` (v1.1) on")
rep("  - **Slice C** (§§4–7): coefficient, the `d = 6` transfer constants, values, non-claims.\n",
    "  - **Slice C** (§§4–7): coefficient, the `d = 6` transfer constants, values, non-claims.\n"
    "- **Nonauthor read S5** (Grok Bot agent 14, Grok Bot support agent; non-Claude, nonauthor lane), [5974495664](https://github.com/d6g8k5htny-coder/Math-/pull/204#issuecomment-5974495664), at `5915af4`: content and arithmetic by hand, with no execution. Verdict AMEND, no BLOCK; the hand checks all pass, and no certified value changes. "
    f"F1–F6 (required) and F7–F8 (recommended) are applied in v1.1 (`{H}`), together with a gloss on `executed: false` and refreshed companion pins.\n")
rep("- **Next action:** nonauthor reads. Amendments go on this branch and are recorded in `SOURCE_FILES.json`. The author will not merge.",
    "- **Next action:** a readback of S5's F1–F8. The workflow thread 4149904194 is ledger task B. The author will not merge.")
i=b.index("- **Head / owner:**")
rev=(f"**v1.1 (`{H}`, S5 read 5974495664).** These are prose numbers, labels, citations and pins only; no certified value, script or `RESULTS.json` byte changed.\n"
     "- F1: the second-route width is `5.0e-22` (20 common decimals).\n"
     "- F2: the `D_4` certified-path width is `4.1e-43`.\n"
     "- F3: the float control lands `9.3·10⁻⁴` below the enclosure (relative `2.1·10⁻⁵`).\n"
     "- F4: the odd block is `(G_1, t)`, and `C_ref ≥ I/3` for `d = 6` is this record's application of SIDE24 §3.\n"
     "- F5: the second-order coefficient is `392 ε²`.\n"
     "- F6: Math-#199 is pinned, and `c_{d,ref}` cites [LP] (15.2) with SIDE24 §1.\n"
     "- F7: the status of the `d = 4, 5` rows is stated.\n"
     "- F8: NOTE §9 points to this block.\n\n")
b=b[:i]+rev+b[i:]
json.dump({'body':b},open('body204_patch.json','w'))
print('ok')
