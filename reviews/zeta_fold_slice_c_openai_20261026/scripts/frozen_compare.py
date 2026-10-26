#!/usr/bin/env python3
"""Compare the slice-C rerun (in the worktree packet) against the committed outputs.

Classifies every changed committed file as IDENTICAL, TIMING-ONLY (equal after masking the
timing tokens listed below) or DIFFERENT (unified diff excerpt kept); hashes regenerated raw
samples against MANIFEST.json; rebuilds track_low10_dt_sensitivity.out from the five dt runs
the way its header describes. Writes COMPARE.md and compare.json into ZF_LOG.
"""
import difflib
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

PKT = Path(os.environ["ZF_PKT"]).resolve()
LOG = Path(os.environ["ZF_LOG"]).resolve()
REL = "experiments/zeta_fold_exploration"

TIMING = [
    (re.compile(r"(setup )[0-9.]+s"), r"\1<T>s"),
    (re.compile(r"(zeta zeros in )[0-9.]+s"), r"\1<T>s"),
    (re.compile(r"^time [0-9.eE+-]+\s*$", re.M), "time <T>"),
    (re.compile(r"(scan time )[0-9.eE+-]+"), r"\1<T>"),
    (re.compile(r"( time )[0-9.eE+-]+$", re.M), r"\1<T>"),
    (re.compile(r'("seconds(?:_zero_finding|_total)?": )[0-9.eE+-]+'), r"\1<T>"),
    (re.compile(r'("sec_per_1e5_evals_at_9e5": )[0-9.eE+-]+'), r"\1<T>"),
    (re.compile(r"\([0-9.]+ ?s\)"), "(<T>s)"),
    (re.compile(r"(samples in )[0-9.]+s"), r"\1<T>s"),
    (re.compile(r"^(init )[0-9.]+( N \d+)$", re.M), r"\1<T>\2"),
]


def mask(s):
    for rx, rep in TIMING:
        s = rx.sub(rep, s)
    return s


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=PKT, text=True, capture_output=True, check=True).stdout


def rebuild_dt_sensitivity():
    target = PKT / "results/dbn/track_low10_dt_sensitivity.out"
    committed = git("show", f"HEAD:{REL}/results/dbn/track_low10_dt_sensitivity.out").splitlines()
    header = [l for l in committed[:2] if l.startswith("#")]
    lines = list(header)
    for dt in ["0.01", "0.02", "0.025", "0.04", "0.05"]:
        p = LOG / f"track_dt{dt}.stdout"
        if not p.exists():
            return f"missing {p.name}"
        lines.append(f"== DT={dt}")
        for l in p.read_text().splitlines():
            lines.append(l[:160] if l.startswith("EVENT") else l)
    target.write_text("\n".join(lines) + "\n")
    return "rebuilt"


def main():
    report = {"dt_sensitivity": rebuild_dt_sensitivity()}
    status = git("status", "--porcelain", "--", ".").splitlines()
    changed = sorted({l[3:] for l in status if l[:2].strip() in ("M", "MM", "AM")})
    untracked = sorted(l[3:] for l in status if l.startswith("??"))
    deleted = sorted(l[3:] for l in status if l[:2].strip() == "D")
    rows = []
    for rel in changed:
        rel_pkt = rel[len(REL) + 1:] if rel.startswith(REL + "/") else rel
        new = PKT / rel_pkt
        old_bytes = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=PKT, capture_output=True, check=True).stdout
        new_bytes = new.read_bytes()
        row = dict(path=rel_pkt, old_bytes=len(old_bytes), new_bytes=len(new_bytes))
        if old_bytes == new_bytes:
            row["class"] = "IDENTICAL"
        else:
            try:
                o, n = old_bytes.decode(), new_bytes.decode()
                if mask(o) == mask(n):
                    row["class"] = "TIMING-ONLY"
                else:
                    row["class"] = "DIFFERENT"
                    d = list(difflib.unified_diff(o.splitlines(), n.splitlines(), "committed", "rerun", lineterm="", n=0))
                    row["diff_lines"] = sum(1 for l in d if l[:1] in "+-" and not l.startswith(("+++", "---")))
                    row["diff_excerpt"] = d[:60]
            except UnicodeDecodeError:
                row["class"] = "DIFFERENT(binary)"
        rows.append(row)
    report["changed"] = rows
    report["untracked"] = untracked
    report["deleted"] = deleted
    tracked_outputs = [l for l in git("ls-files", "results").splitlines()]
    report["committed_output_files"] = len(tracked_outputs)
    report["unchanged_output_files"] = len(tracked_outputs) - len([r for r in rows if r["path"].startswith("results/")])

    man = json.loads((PKT / "MANIFEST.json").read_text())
    raw_rows = []
    for r in man["raw_files"]:
        p = PKT / r["path"]
        if p.exists():
            h = sha(p)
            raw_rows.append(dict(path=r["path"], expected_sha256=r["sha256"], got_sha256=h,
                                 expected_bytes=r["bytes"], got_bytes=p.stat().st_size,
                                 match=(h == r["sha256"] and p.stat().st_size == r["bytes"])))
        else:
            raw_rows.append(dict(path=r["path"], expected_sha256=r["sha256"], got_sha256=None, match=None))
    report["raw"] = raw_rows

    ledger = []
    for lane in ("dbn", "stats"):
        p = LOG / f"ledger_{lane}.jsonl"
        if p.exists():
            ledger += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    report["ledger"] = ledger
    report["jobs_total"] = len(ledger)
    report["jobs_rc_nonzero"] = [j["name"] for j in ledger if j["rc"] != 0]
    report["wall_total_s"] = round(sum(j["wall_s"] for j in ledger), 1)

    (LOG / "compare.json").write_text(json.dumps(report, indent=1) + "\n")
    md = ["# Slice C rerun vs committed outputs", ""]
    md.append(f"Jobs run: {len(ledger)}; nonzero rc: {report['jobs_rc_nonzero'] or 'none'}; "
              f"summed wall time: {report['wall_total_s']} s.")
    md.append(f"Committed output files under results/: {len(tracked_outputs)}; byte-unchanged after rerun: "
              f"{report['unchanged_output_files']}.")
    md.append(f"dt-sensitivity file: {report['dt_sensitivity']}.")
    md += ["", "| changed file | class | old bytes | new bytes | diff lines |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| `{r['path']}` | {r['class']} | {r['old_bytes']} | {r['new_bytes']} | {r.get('diff_lines', '')} |")
    md += ["", "| raw sample | manifest sha256 | rerun sha256 | match |", "|---|---|---|---|"]
    for r in raw_rows:
        md.append(f"| `{r['path']}` | `{r['expected_sha256'][:16]}…` | `{(r['got_sha256'] or 'absent')[:16]}…` | {r['match']} |")
    if untracked:
        md += ["", "Untracked files left by the rerun: " + ", ".join(f"`{u}`" for u in untracked)]
    for r in rows:
        if r["class"] == "DIFFERENT":
            md += ["", f"## diff excerpt: `{r['path']}`", "```"] + r["diff_excerpt"] + ["```"]
    (LOG / "COMPARE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
