#!/usr/bin/env python3
"""Render the census of main's proof obligations, checking every citation.

The entries below are a manual reading of each source at REV, guided by
extract.py and digest.py. Building them checks, against Git objects at REV:
  - every cited path exists, with the recorded blob;
  - every cited line range lies inside that file;
  - every cited anchor text occurs on a line inside its range.
A failed check stops the build with exit status 1, so the census cannot cite
a location that does not exist at the revision (A7).

Writes census.json and CENSUS.md. Usage: build_census.py --git DIR --out DIR
"""

import argparse
import json
import posixpath
import re
import subprocess
import sys
from pathlib import Path

REV = "81acc060341e8a31cc4cd4f9523ef71fb6c06175"
PROSE = "QUARANTINED"
LEAN = "CLOSED-CANDIDATE"
ARCHIVAL = "QUARANTINED-ARCHIVAL"

NOT_KERNEL = "Not kernel-verified; prose review cannot establish kernel closure (GOVERNANCE.md Layer A)."

# Each cite is (path, first_line, last_line, anchor_text_or_None).
ENTRIES = [
    {
        "id": "OBL-MAIN-01", "path": "docs/research-translation/20260930/cap-assumption-ablation/PROOF.md",
        "blob": "7158f3788ad9", "kind": "counterexample (negative result)", "disposition": PROSE,
        "claim": "The interior-height hypothesis C1 of the D1 cap theorem cannot be dropped, even keeping C2, C3, "
                 "every pin jet and arbitrarily good uniform closeness, for smooth Morse functions with distinct "
                 "critical values on a fixed compact torus.",
        "statement": [(15, 51, "could C1 be replaced"), (231, 255, "cannot be dropped")],
        "nonclaims": [(3, 8, "not a second acceptance vote"), (244, 251, "No violation of the source cap theorem")],
        "reviews": "No adjacent review record. The text describes itself as one delegated AI derivation.",
        "depends": "D1's cap theorem (PR196, AUDIT.md), which this note does not re-audit.",
        "boundary": "Only the explicit perturbation family of sections 2-4 is shown to break C1 while keeping C2, "
                    "C3 and the pin jets. Nothing here refutes or accepts the D1 theorem. " + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-02", "path": "experiments/periodic_h0/spectral_shrinking_bins/PROOF.md",
        "blob": "1a4b544d018d", "kind": "conditional analytic theorem, author-side", "disposition": PROSE,
        "claim": "A finite error bound (5) and its asymptotic consequence (7) for shrinking lifetime bins of "
                 "grid samples coupled to a Gaussian Fourier field, under the imported hypotheses D, C and A.",
        "statement": [(15, 125, "consumed hypotheses"), (164, 206, "Theorem (finite error bound")],
        "nonclaims": [(441, 452, "No evaluated value of K_C")],
        "reviews": "Self-declared at lines 3-4: source-bound nonauthor review pending. No adjacent review record.",
        "depends": "The imported continuum density D and count C interfaces; the M4-M5 coefficient interface.",
        "boundary": "Conditional on D, C and A exactly as stated in section 1. No evaluated constant, grid "
                    "threshold or historical-experiment certification. Unreviewed. " + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-03", "path": "experiments/universality/finite_candidate_lifetime/PROOF.md",
        "blob": "a27a84150b3e", "kind": "conventional analytic theorem", "disposition": PROSE,
        "claim": "For each fixed finite Fourier Gaussian law (1) with K>=2, the ordered maximum-saddle candidate "
                 "measure has a finite continuous density nu_cand(ell) ~ C_loc ell^(-1/3) with the explicit "
                 "coefficient (4), and cumulative count (3/2) C_loc t^(2/3).",
        "statement": [(98, 156, "Candidate theorem")],
        "nonclaims": [(154, 156, "does not claim the stronger remainder"), (553, 577, "Exact remaining scope")],
        "reviews": "REVIEW.md: nonauthor PASS within the requested scope and a no-must-fix full-source verdict "
                   "(OpenAI/Codex, organizational independence 0).",
        "depends": "External: the Math- parent P (UNIFORM_MATRIX_CAP_AND_LIFETIME.md at 13af1089) with its errata E1 and E2.",
        "boundary": "Candidate pairs only: not actual persistence bars, no O(1) remainder, Gaussian coefficient "
                    "law only (not the finite-word sampler). " + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-04", "path": "experiments/universality/finite_contact_genericity/PROOF.md",
        "blob": "569a49dee2ec", "kind": "conventional analytic theorem", "disposition": PROSE,
        "claim": "For each fixed law, target, basepoint and frame, almost surely under the canonical conditional "
                 "law: the contact is an isolated critical point; every other critical point is Morse; the "
                 "critical set is finite; the other critical values are distinct and differ from b.",
        "statement": [(97, 120, "Precise result and quantifiers")],
        "nonclaims": [(109, 113, "No simultaneous probability-one"), (455, 473, "Remaining obligations")],
        "reviews": "REVIEW.md: full mathematical PASS, a bounded PASS, and a new full-source PASS (OpenAI/Codex).",
        "depends": "External: the frozen PR321 candidate and chart sources.",
        "boundary": "One fixed conditional law at a time; no statement over uncountably many targets or frames; "
                    "the whole conditional field is not claimed Morse. " + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-05", "path": "experiments/universality/finite_elder_transfer/PROOF.md",
        "blob": "69f84eae401b", "kind": "deterministic analytic theorem", "disposition": PROSE,
        "claim": "Under global C4 convergence, the exact finite pins (1) and the limit contact data (2) with "
                 "finite Morse, distinct other critical values: for large j the pinned pair (M_j, S_j) is the "
                 "actual finite H0 superlevel elder pair, with lifetime exactly k_j r_j^3.",
        "statement": [(116, 154, "Deterministic theorem")],
        "nonclaims": [(151, 154, "A deterministic"), (441, 458, "Remaining consumers")],
        "reviews": "REVIEW.md: fresh full exact-file nonauthor PASS of the incorporated proof (OpenAI/Codex).",
        "depends": "OBL-MAIN-04 supplies its hypotheses almost surely under the contact law.",
        "boundary": "The threshold may depend on the realization; no deterministic ensemble-wide threshold. "
                    + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-06", "path": "experiments/universality/finite_fold_chart/PROOF.md",
        "blob": "6febb6e697a7", "kind": "analytic theorem, research preparation", "disposition": PROSE,
        "claim": "Uniform derivative-controlled fold charts for C4 fields on the embedded box, under the pins, "
                 "the fixed windows and condition (A).",
        "statement": [(53, 134, "Theorem and exact assumptions")],
        "nonclaims": [(513, 551, "Formal API consumption")],
        "reviews": "REVIEW.md: no-must-fix successor verdict and a full-source PASS for the bounded analytic "
                   "theorem (OpenAI/Codex).",
        "depends": "Its stated assumptions; consumed by OBL-MAIN-09.",
        "boundary": "Exactly the stated assumptions and windows; no compiled Lean application of these bounds "
                    "(line 543). " + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-07", "path": "experiments/universality/finite_gaussian_contact/PROOF.md",
        "blob": "5f345201309c", "kind": "conventional analytic interfaces", "disposition": PROSE,
        "claim": "For positive finite Fourier Gaussian spectra (K>=2), the named contact, divided-difference, "
                 "regression, compact-mark normalizer, envelope, off-diagonal and local-jet interfaces "
                 "(sections 2-8), and a finite-r physical Hessian bridge (section 10).",
        "statement": [(10, 93, "Source, field and exact scope")],
        "nonclaims": [(425, 479, "Negative boundaries")],
        "reviews": "REVIEW.md: two ACCEPT_SCOPED readings and a PASS of the sections 9-10 physical-interface "
                   "scope (OpenAI/Codex).",
        "depends": "External: the Math- parent P (UNIFORM_MATRIX_CAP_AND_LIFETIME.md at 13af1089) with E1, CAP and E2.",
        "boundary": "The ten catalogue spectra at L=24, K=2 or 3, and the general finite law (1). Not an "
                    "admission to the full persistence theorem. " + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-08", "path": "experiments/universality/finite_h0_lifetime/PROOF.md",
        "blob": "7a3518edd9bd", "kind": "composition theorem", "disposition": PROSE,
        "claim": "Consuming OBL-MAIN-05 for the same fixed law, the finite H0 elder-bar intensity has a density "
                 "nu_bar ~ C_loc ell^(-1/3) with the candidate coefficient, and cumulative count "
                 "(3/2) C_loc t^(2/3).",
        "statement": [(89, 146, "Composition theorem")],
        "nonclaims": [(495, 526, "Finite family and remaining boundaries")],
        "reviews": "REVIEW.md: PASS readings of the preparation and fresh full nonauthor PASS readings of the "
                   "incorporated proof; metadata AMEND requests are recorded at lines 107-118 (OpenAI/Codex).",
        "depends": "The candidate, genericity and elder-transfer rows it composes (see its links).",
        "boundary": "The fixed finite law only; no numerical enclosure or angular/area factor. " + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-09", "path": "experiments/universality/finite_weighted_fold_window/PROOF.md",
        "blob": "f5473c0e4502", "kind": "conventional analytic estimate", "disposition": PROSE,
        "claim": "With constants chosen from the uniform moments and normalizer floor, for 0<r<=r_* the weighted "
                 "law gives Q^W(R<=T and |A|>=a_0) >= 1-epsilon, removing the deterministic fold-chart windows.",
        "statement": [(231, 290, "Explicit choice and exact quantifier")],
        "nonclaims": [(322, 357, "Chart consequence and remaining boundaries")],
        "reviews": "REVIEW.md: PASS, delta PASS and a new full-source PASS (OpenAI/Codex).",
        "depends": "External: the compact-mark normalizer theorem.",
        "boundary": "The fold theorem's continuous parameter dependence remains conditional (line 339). "
                    + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-10", "path": "experiments/universality/fold_structural_universality/PROOF.md",
        "blob": "483d779cfdfd", "kind": "structural theorem", "disposition": PROSE,
        "claim": "For a regular cubic fold on a compact manifold, the five interfaces COUNT, CONTACT, SELECT, "
                 "ENVELOPE and FAR are sufficient for the H0 lifetime exponent -1/3, through a dimensional "
                 "cancellation (section 6).",
        "statement": [(19, 73, "Structural claim"), (379, 449, "Proof of the structural theorem")],
        "nonclaims": [(25, 27, "remains a substantive admission task"), (691, 750, "Conditional order taxonomy")],
        "reviews": "REVIEW.md: fresh full PASS of the original preparation and a fresh FULL PASS of the current "
                   "file (OpenAI/Codex).",
        "depends": "Its sufficiency statement uses no other row; the rows it links are adapters and examples. Admission for a given field is a separate obligation.",
        "boundary": "Sufficiency only. Verifying the five interfaces for a particular field is not done here; "
                    "no higher-singularity class law. " + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-11", "path": "experiments/universality/higher_homology_fold/PROOF.md",
        "blob": "8e18a7900e7d", "kind": "deterministic and probabilistic theorems", "disposition": PROSE,
        "claim": "Attaching obstructions and isolated-fold unit incidence in degree p for C4 fields with one "
                 "isolated degenerate critical point; an actual Hp lifetime density (section 10) and a "
                 "degree-marked iid-copy corollary (section 11).",
        "statement": [(25, 78, "the exact deterministic claim"), (562, 677, "Actual Hp lifetime density")],
        "nonclaims": [(755, 796, "Failure boundaries")],
        "reviews": "REVIEW.md: two FULL PASS readings of the preparation, a FULL PASS of the current proof, and "
                   "a README wording AMEND that is retained (OpenAI/Codex).",
        "depends": "Its section 13 binds the operational sources; SG is OBL-MAIN-15.",
        "boundary": "Ordinary finite Hp intervals only; no extended, relative, zigzag or torsion-valued claim "
                    "(line 440). " + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-12", "path": "experiments/universality/iid_short_bar_process/PROOF.md",
        "blob": "cd7ef8b8a364", "kind": "process limit theorem", "disposition": PROSE,
        "claim": "For the periodized Gaussian law (1), fixed d>=2 and L: a single field's short-bar process has "
                 "an empty limit, while independent whole copies give a nonempty marked Poisson limit; with an "
                 "explicit first-moment counterexample (section 8).",
        "statement": [(19, 74, "Scope: fixed law")],
        "nonclaims": [(519, 618, "Existing lanes and residual scope")],
        "reviews": "REVIEW.md: metadata PASS and scoped mathematical PASS readings (OpenAI/Codex).",
        "depends": "The all-bar first moment IA (its text, lines 30-41).",
        "boundary": "Independent whole copies of one fixed law; qualitative dominated convergence, not a "
                    "certified numerical window (line 536). " + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-13", "path": "experiments/universality/periodized_gaussian_h0/PROOF.md",
        "blob": "b27e2bc84839", "kind": "admission theorem for one law", "disposition": PROSE,
        "claim": "For the exact periodized Gaussian law K_L (fixed d>=2, L>0), the five interfaces are derived and "
                 "compose to the actual H0 lifetime intensity, with an exact covariance-derived coefficient "
                 "(section 8).",
        "statement": [(17, 84, "exact theorem scope"), (498, 568, "Exact covariance-derived coefficient")],
        "nonclaims": [(569, 637, "residual scope")],
        "reviews": "REVIEW.md: fresh full PASS of the preparation and a FULL PASS of the current file "
                   "(OpenAI/Codex).",
        "depends": "OBL-MAIN-16 for the L=24 instance (its lines 17-27).",
        "boundary": "The one law K_L; it does not import a finite-cutoff limit or a historical remainder. "
                    + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-14", "path": "experiments/universality/spectral_amplitude_crossover/PROOF.md",
        "blob": "5d5b7e36beea", "kind": "exact regime theorem", "disposition": PROSE,
        "claim": "Three amplitude regimes (sub-, critical and supercritical) with exact density and cumulative "
                 "asymptotics and process limits for independent whole copies (the section 11 table).",
        "statement": [(669, 712, "Summary of exact regimes")],
        "nonclaims": [(686, 689, "There is no spatial expanding-domain law")],
        "reviews": "REVIEW.md: two AMEND verdicts on the original preparation, retained; PASS on the corrected "
                   "preparation; a fresh FULL PASS of the current file (OpenAI/Codex).",
        "depends": "The stated base premises; see its section 1.",
        "boundary": "Independent whole copies only; no within-one-field Poisson statement or coalescing-contact "
                    "factorial estimate. " + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-15", "path": "experiments/universality/spectral_gaussian_admission/PROOF.md",
        "blob": "72c896474073", "kind": "admission theorem", "disposition": PROSE,
        "claim": "Admission of regular-fold H0 lifetimes and iid-copy processes for declared stationary Gaussian "
                 "laws, with the concrete spectral verifier (P5)+(A4).",
        "statement": [(19, 92, "Admission theorem")],
        "nonclaims": [(706, 755, "Refusals"), (698, 698, "does not claim samples C-infinity")],
        "reviews": "REVIEW.md: two FULL PASS readings, a FULL PASS of the current source and a metadata PASS "
                   "(OpenAI/Codex).",
        "depends": "The rows it links.",
        "boundary": "Fixed declared laws; family uniformity is a further condition; C4 regularity, not smooth "
                    "samples. " + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-16", "path": "formal/sources/side24_v1/PROOF.md",
        "blob": "44b66f04f89f", "kind": "coefficient evaluation with outward rational bounds", "disposition": PROSE,
        "claim": "Rational-endpoint enclosures 0.07340691930603427103 < c_2,24 < 0.07340691930603427104 and "
                 "0.04177593184059834334 < c_3,24 < 0.04177593184059834335 of the coefficient defined in the "
                 "parent's equation (15.2).",
        "statement": [(6, 31, "evaluates the coefficient defined in equation (15.2)")],
        "nonclaims": [(27, 31, "No finite-radius error estimate"), (175, 175, "NOT claimed equal")],
        "reviews": "Its own line 4 still reads 'nonauthor review open', but STATUS.md's D3 row records the "
                   "review on main #65. That status disagreement belongs on the G8 status-truth list.",
        "depends": "The Math- parent UNIFORM-MATRIX-CAP-LIFETIME (sha256 9350ad6e...), whose elder-selection and "
                   "Kac-Rice proof this note does not accept.",
        "boundary": "The enclosures of the specified expression only; interpreting it as the finite-bar constant "
                    "stays conditional on the parent. Its arithmetic skeleton is OBL-MAIN-21; the analytic "
                    "steps are prose. " + NOT_KERNEL,
    },
    {
        "id": "OBL-MAIN-17", "path": "incoming/chatgpt-c022-q0-recovery-20260928/context/Lemma I Package.md",
        "blob": "5ec0e33efe98", "kind": "incoming custody copy: lemma package", "disposition": ARCHIVAL,
        "claim": "Its own text calls Lemma I 'resolved', but by measured or numerically verified components.",
        "statement": [(1, 13, "Lemma I")],
        "nonclaims": [(32, 32, "measured directly")],
        "reviews": "None on main.",
        "depends": "C007-C009 snapshots off main.",
        "boundary": "A historical incoming copy, not adopted by main. Measured or numerical components are not "
                    "proof. Nothing on main depends on it.",
    },
    {
        "id": "OBL-MAIN-18", "path": "incoming/chatgpt-c022-q0-recovery-20260928/context/Lemma MODZONE CountLaw Package.md",
        "blob": "55c8712a17dd", "kind": "incoming custody copy: adjudication", "disposition": ARCHIVAL,
        "claim": "OBL-BETA-MODZONE adjudicated FAILED-AS-WRITTEN on both routes and superseded by "
                 "OBL-BETA-RELEVANCE.",
        "statement": [(1, 14, "FAILED-AS-WRITTEN")],
        "nonclaims": [(117, 134, "superseding obligation")],
        "reviews": "None on main.",
        "depends": "C023 snapshots off main.",
        "boundary": "A historical incoming copy of a failed-as-written obligation. Nothing on main depends on it.",
    },
    {
        "id": "OBL-MAIN-19", "path": "incoming/claude-bundle-v3-native-copies-20260928/K3_SIDE24_LB/RETURN_06/THEOREM_TABLE.md",
        "blob": "5a800ba09283", "kind": "incoming custody copy: theorem table", "disposition": ARCHIVAL,
        "claim": "A graded table of carrier statements (EXACT, CERTIFIED INTERVAL, conservative bound, measured "
                 "estimate, diagnostic), including conditional ones.",
        "statement": [(1, 28, "THEOREM_TABLE")],
        "nonclaims": [(13, 13, "CONDITIONAL")],
        "reviews": "None on main.",
        "depends": "Carriers in its SOURCE_CAPSULE, off main.",
        "boundary": "A historical incoming copy; its rows carry their own grades and are not adopted by main.",
    },
    {
        "id": "OBL-MAIN-20", "path": "reviews/lb_rate_thm023_landing_20260926/REVIEW.md",
        "blob": "b612adab", "kind": "review with HOLD disposition", "disposition": PROSE,
        "claim": "LB-RATE / KIMI-THM-023: the constant c = 0.9144 is not accepted as a proved uniform theorem "
                 "constant.",
        "statement": [(1, 12, "HOLD")],
        "nonclaims": [(8, 8, "Do not accept")],
        "reviews": "This file is the review: HOLD, a conditional assembly draft (xAI, organizational "
                   "independence 0).",
        "depends": "Its float-based verifiers under history/carrier20_recovery_20260928/ "
                   "(01XJMJ's A3 census, issue 684 comment 6503611192).",
        "boundary": "The obligation is open and must not be indexed as CLOSED; its history/ verifiers are not "
                    "A3-admissible certificates.",
    },
]

LEAN_ENTRY = {
    "id": "OBL-MAIN-21", "path": "formal/manifest.json", "kind": "Lean 4 targets (31)", "disposition": LEAN,
    "boundary": "Each target's exact Lean statement, with the manifest's allowed axioms propext, Classical.choice "
                "and Quot.sound. CLOSED only once G3 (build), G4 (kernel recheck) and G2 (axioms) are observed in "
                "the final run. Root reports a fresh build, leanchecker, axioms and negative controls passing at "
                "c3377cc8 (issue 684 comment 6503255828), which is not yet the final run. Alignment of each target "
                "with its informal anchor stays PENDING_INDEPENDENT_REVIEW, and each target's does_not_claim text "
                "bounds it.",
}


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE, text=True).stdout


def check(repo, entry):
    blob = git(repo, "rev-parse", f"{REV}:{entry['path']}").strip()
    assert blob.startswith(entry["blob"]), f"{entry['id']}: blob {blob} does not match {entry['blob']}"
    lines = git(repo, "show", f"{REV}:{entry['path']}").splitlines()
    for first, last, anchor in entry["statement"] + entry["nonclaims"]:
        assert 1 <= first <= last <= len(lines), f"{entry['id']}: lines {first}-{last} outside 1-{len(lines)}"
        if anchor:
            window = "\n".join(lines[first - 1:last])
            assert anchor in window, f"{entry['id']}: anchor {anchor!r} not in lines {first}-{last}"
    entry["blob"] = blob
    entry["lines"] = len(lines)


PROOF_LINK = re.compile(r"(?<![\w])(?:[\w.-]+/)*(?:PROOF|THEOREM|LEMMA)[\w.-]*\.md")


def links(repo, entry, by_path):
    """Census rows this document links to, resolved from its own relative links."""
    text = git(repo, "show", f"{REV}:{entry['path']}")
    base = posixpath.dirname(entry["path"])
    found = set()
    for raw in PROOF_LINK.findall(text):
        for candidate in (posixpath.normpath(posixpath.join(base, raw)), posixpath.normpath(raw)):
            if candidate in by_path and candidate != entry["path"]:
                found.add(by_path[candidate])
    return sorted(found)


def lean_targets(repo):
    manifest = json.loads(git(repo, "show", f"{REV}:formal/manifest.json"))
    return manifest, [{k: t[k] for k in ("name", "module", "title", "does_not_claim") if k in t}
                      for t in manifest["targets"]]


def render(entries, lean, targets):
    out = ["# Census of main's proof obligations", "",
           f"Read at `main` `{REV}`. These are draft rows for Root's `PROOF_INDEX.md`, and they are not dispositions. "
           "Every cited path, blob, line range and anchor below was checked mechanically against that revision by "
           "`build_census.py`. Each claim restates the document's own words; it is not a new mathematical review.", "",
           "| ID | Draft disposition | Kind | Path |", "|---|---|---|---|"]
    for e in entries:
        out.append(f"| {e['id']} | {e['disposition']} | {e['kind']} | `{e['path']}` |")
    out.append(f"| {lean['id']} | {lean['disposition']} | {lean['kind']} | `{lean['path']}` |")
    out += ["", "## Prose and archival obligations", ""]
    for e in entries:
        cite = lambda rows: "; ".join(f"lines {a}-{b}" if a != b else f"line {a}" for a, b, _ in rows)
        out += [f"### {e['id']}: `{e['path']}`", "",
                f"- **Blob** `{e['blob'][:12]}`, {e['lines']} lines. **Kind:** {e['kind']}.",
                f"- **Claim:** {e['claim']}",
                f"- **Statement:** {cite(e['statement'])}. **Stated limits:** {cite(e['nonclaims'])}.",
                f"- **Reviews:** {e['reviews']}",
                f"- **Inputs:** {e['depends']}",
                f"- **Links, from its own text:** {', '.join(e['links']) or 'none'}. Links include references "
                "and examples as well as logical dependencies.",
                f"- **Draft disposition:** {e['disposition']}. **Boundary:** {e['boundary']}", ""]
    out += [f"## {lean['id']}: the 31 Lean targets in `formal/`", "", lean["boundary"], "",
            "| Target | Module | Does not claim |", "|---|---|---|"]
    for t in targets:
        out.append(f"| `{t['name']}` | `{t['module']}` | {t.get('does_not_claim', '')} |")
    out += ["", "## Pointers, not new rows", "",
            "- **`STATUS.md` rows D1-D6, SARD-G and C6** are Math- obligations, with their sources in Math-. "
            "They belong to Math-'s index, whose candidates DB642 extracted (issue 684 comment 6503371863). "
            "`STATUS.md` is dated 2026-09-29, so it needs the G8 refresh.",
            "- **The two float-based `history/carrier20_recovery_20260928/public_carriers/` verifiers** should be "
            "labelled as 01XJMJ proposed. They back OBL-MAIN-20.", ""]
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--git", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        by_path = {e["path"]: e["id"] for e in ENTRIES}
        for e in ENTRIES:
            check(args.git, e)
            e["links"] = links(args.git, e, by_path)
        manifest, targets = lean_targets(args.git)
        assert len(targets) == 31, f"expected 31 Lean targets, found {len(targets)}"
        assert manifest["allowed_axioms"] == ["propext", "Classical.choice", "Quot.sound"]
    except AssertionError as exc:
        print(f"CENSUS CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    data = {"rev": REV, "entries": ENTRIES, "lean": LEAN_ENTRY | {"targets": targets}}
    (args.out / "census.json").write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    (args.out / "CENSUS.md").write_text(render(ENTRIES, LEAN_ENTRY, targets))
    print(f"census OK: {len(ENTRIES)} prose/archival rows and 31 Lean targets, all citations checked at {REV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
