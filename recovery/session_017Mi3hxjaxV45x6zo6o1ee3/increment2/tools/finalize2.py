"""Write LABEL files, RECOVERY_MAP.tsv, EXCLUDED.json, MANIFEST2.json, MANIFEST2.md and SHA256SUMS2.txt (stdlib)."""
import collections
import hashlib
import json
import os
import sys
import time

SCRATCH = os.path.abspath(sys.argv[1])
STAGE = os.path.abspath(sys.argv[2])
REC2 = os.path.join(SCRATCH, "rec2")
SESSION = "session_017Mi3hxjaxV45x6zo6o1ee3"
CK1 = "d579bcc05aa8173986fdb2634fbcf75df7e9f53b"
NOW = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
res = json.load(open(os.path.join(REC2, "stage2_result.json")))
rows, uploaded, excluded = res["rows"], res["uploaded"], res["excluded"]
GH = "https://github.com/d6g8k5htny-coder/"

LABEL_TORUS = """# LABEL: LOCAL, NON-AUTHORITATIVE, NOT HOSTED EVIDENCE

These files are the untracked outputs of a local run of `companions/torus_filtration_20261011/replay.sh`
(Math- repository) performed in the Claude Code cloud container of session
`session_017Mi3hxjaxV45x6zo6o1ee3` on 12 October 2026, 22:52–22:57 UTC, at Math- commit
`40440ed8bcb80913426042c80f6facedc6a4965c` (`checked_commit` in `source.json` and `receipt.json`), with
`LEAN_NUM_THREADS=2`; elan 4.2.4, `leanprover/lean4:v4.34.1`, Lake 5.0.0, Mathlib
`d13f23b723b8a846827a245b89c10fc7d3f11612`; container 15 GB RAM, 4 CPUs. The run exited 0 in 5 m 5 s with a
peak cgroup memory of 12,645,441,536 bytes (`toolchain_readiness/torus_replay.log` and `torus_replay_peak.txt`
in checkpoint 1, commit d579bcc0).

Status of every file here, by the formal lane's rule: **local diagnostic output**. `receipt.json` is a locally
produced receipt and is **invalid as a receipt**: only the hosted workflow run named in the companion's own
records is the receipt for this companion. Nothing here promotes, accepts or changes any scientific or
formalization status; scientific effect NONE; organizational-independence credit 0. The files are preserved
because the owner asked that local Lean logs be kept rather than discarded (main#307 comments 6136210973 and
6136304987). The `<SCRATCH>` placeholder replaces the container's private scratch path where it occurred.
"""
LABEL_FORMAL = """# LABEL: LOCAL, NON-AUTHORITATIVE, PARTIAL RUN, NOT A GATE PASS

These files are the untracked outputs of a local, deliberately partial run of the Math- `formal/` gate in the
Claude Code cloud container of session `session_017Mi3hxjaxV45x6zo6o1ee3` on 12 October 2026, 22:45–22:47 UTC,
at Math- commit `40440ed8bcb80913426042c80f6facedc6a4965c`, with the pinned toolchain (elan 4.2.4,
`leanprover/lean4:v4.34.1`, Lake 5.0.0, Mathlib `d13f23b723b8a846827a245b89c10fc7d3f11612`).

The full gate (`formal/gate.py --execute`) had been attempted first and its `leanchecker` step was killed by the
container's memory cgroup (exit 137, 13.49 GB anonymous RSS against a 14,343,688,192-byte limit; see
`toolchain_readiness/math_gate_execute.log` in checkpoint 1). This tree comes from a second run in which only the
leanchecker step was replaced by a logged SKIP (`leanchecker.log` says so); build, axiom audit, type checks and
negative controls ran. **This is not a gate pass**, `receipt.json` is **invalid as a receipt**, and the local
copy of the modified gate (`toolchain_readiness/gate_partial_LOCAL_DIAGNOSTIC_ONLY.py`, checkpoint 1) is not the
project gate and must not be installed. The hosted `formal-verification` runs are the only receipts.
Scientific effect NONE; organizational-independence credit 0. Preserved at the owner's request (main#307
comments 6136210973 and 6136304987); `<SCRATCH>` replaces the container's private scratch path.
"""
open(os.path.join(STAGE, "local_lean_evidence/torus_filtration_20261011_replay/LABEL.md"), "w").write(LABEL_TORUS)
open(os.path.join(STAGE, "local_lean_evidence/math_formal_partial_gate/LABEL.md"), "w").write(LABEL_FORMAL)

AREA_NOTES = {
    "(root)": "working files of every round, 26 September – 13 October: PR bodies and body-patch payloads, comment drafts and readbacks, probe and exact-check scripts, successive versions of edited records (for example `audit_new*.md` / `audit_rb*.md` are the successive texts of the edited audit comment main#229 6004355577, whose earlier versions GitHub does not keep), small logs",
    "v13": "draft of Lean packet v1.3 (cap first-exit stack); superseded by the landed packets of Math-#253–#266",
    "v15": "draft of Lean packet v1.5 before the K-stack amendments; landed in Math-#253",
    "v16": "draft of Lean packet v1.6; landed in Math-#255", "v17": "draft of Lean packet v1.7; landed in Math-#257",
    "v18": "draft of Lean packet v1.8; landed in Math-#258", "v19": "draft of Lean packet v1.9; landed in Math-#259",
    "v20": "draft of Lean packet v2.0; landed in Math-#260", "v21": "draft of Lean packet v2.1; landed in Math-#262",
    "v22": "draft of Lean packet v2.2 (`trace.txt` is the local Lean build trace); landed in Math-#264",
    "v23": "draft of Lean packet v2.3; landed in Math-#266",
    "v19m": "amendment working set for v1.9 (K-stack AMENDs)", "v20m": "amendment working set for v2.0",
    "v21m": "amendment working set for v2.1", "v22m": "amendment working set for v2.2", "v23m": "amendment working set for v2.3",
    "lean25": "Lean sources of the v2.5 draft; landed in Math-#285", "lean26": "Lean sources of the v2.6 draft; landed in Math-#295",
    "leancap": "Lean sources of the cap first-exit packet draft",
    "r169": "probe scripts for the nonauthor Slice A/B/C reviews of Math-#169",
    "r178": "probe scripts, logs and JSON outputs behind the certified tail-constant work (Math-#178 thread, Math-#190)",
    "r181": "probe scripts for the Slice B review of Math-#181", "r185": "probe scripts for the Slice A review of Math-#185",
    "rw3": "review-round probe scripts and logs (rw series); the posted reviews are the records",
    "rw4": "review-round probe scripts and logs (rw series)", "rw5": "review-round probe scripts and logs (rw series)",
    "rw6": "review-round probe scripts and logs (rw series)", "rw7": "review-round probe scripts and logs (rw series)",
    "rd3cf": "read-round probe scripts and outputs (rd series)", "rd4": "read-round probe scripts and logs (rd series)",
    "rd4cf": "read-round probe scripts and outputs (rd series)", "rd5": "read-round probe scripts and logs (rd series)",
    "rd6": "read-round probe scripts and outputs (rd series)",
    "pr105": "replay probe scripts and stderr captures for the review work on the PR named by the directory",
    "pr125": "replay stdout captures for the nonauthor review of Math-#125", "pr128": "replay stdout captures for the nonauthor review of Math-#128",
    "pr132": "replay stdout captures for the nonauthor review of Math-#132",
    "l331": "replay stdout/stderr for the engineering read of Math-#331", "l356": "replay stdout/stderr for the read of Math-#356",
    "l365": "stderr captures from the main#365 dry-run builds", "ev305_normal": "evidence replay of Math-#305 (normal mode)",
    "ev305_opt": "evidence replay of Math-#305 (optimized mode)", "ev331": "evidence replay invocations for Math-#331 (process.json, stdout.bin, stderr.bin per invocation)",
    "p146m": "mutant replay outputs for the Math-#146 review", "p146m-O": "mutant replay outputs for the Math-#146 review (-O)",
    "p146out": "replay outputs for the Math-#146 review", "p146out-O": "replay outputs for the Math-#146 review (-O)",
    "p15_tail_replay_out": "replay outputs for the P15 sharp even tail review", "p157_verify_out.json": "replay outputs for the Math-#157 Slice A review (directory despite its name)",
    "rcsout": "replay outputs for the Math-#135 R-slice review", "rcsout-o": "replay outputs for the Math-#135 R-slice review (-O)",
    "pinout": "mutation-run outputs (per-mutant stdout/stderr and summaries)", "iw_out": "mutation-run outputs (per-mutant stdout/stderr and summaries)",
    "gate-run": "main `frontiers/downstream_gate_20260925` mutation run of 28 September 23:01Z (REPORT.json names the source hashes; mutant trees excluded as regenerable)",
    "gate-run2": "second mutation run of the same packet, 28 September 23:35Z",
    "rt": "copy of the downstream-gate packet with `downstream-gate-results` (28 September); mutant trees excluded",
    "rtfOEb": "copy of the downstream-gate packet with `downstream-gate-results` and `log.txt` (28 September); mutant trees excluded",
    "a022": "mutant control files and workflow script of a 28–30 September packet", "a41": "review texts and API readbacks of early-October reads",
    "c5051": "probe scripts (C50/C51 lemma work on Math-#233)", "c235": "probe script for the Math-#235 read", "collar": "collar probe script",
    "collision_packet": "draft PROOF.md and SOURCE_FILES.json of the witness-collision packet (28 September)",
    "d5_collar_20260928": "draft of the D5 collar packet (28 September)", "elder_dimension_lift_claude_20260928": "draft of the elder dimension lift note (28 September)",
    "remote_pair_law_claude_20260928": "draft of the remote pair law note (28 September)", "synth": "synthesis note, results and certified d=2 script",
    "kreplies": "replies to the K-stack reviews of Math-#257–#260 (text and payload)", "replies213": "replies on Math-#213",
    "dry": "dry-run outputs", "dl320": "files downloaded from Math-#320 for a read (non-identical to any fetched ref; kept)", "dl328": "files downloaded from Math-#328 for a read, with local test outputs",
    "landing_fix": "patch used for a landing fix", "rwin": "window-check log and stderr",
    "takeover": "12 October takeover-round notes and posted texts (API dumps excluded)",
    "pr282": "main#282 custodian working texts (drafts, templates, resolve script)", "pr383": "main#383 author working texts",
    "red282": "main#282 rerun working texts", "m443": "Math-#443 read working texts", "m444": "Math-#444 review working texts",
    "mf27": "meta-framework#27 read working texts", "mf15": "meta-framework#15 decision working texts and parity-probe outputs (local repository copy excluded)",
    "pr384": "main#384 notes", "s307": "hub #307 posting helpers", "toolchain": "toolchain-readiness notes (owner approval relay, executor confirmation, PR body)",
    "verdicts": "verdict texts as drafted for posting (the posted comments are authoritative)",
    "torus_filtration_20261011_replay": "LOCAL Lean replay outputs of the torus-filtration companion (see LABEL.md): non-authoritative, receipt invalid",
    "math_formal_partial_gate": "LOCAL partial Math- formal gate outputs (see LABEL.md): not a gate pass, receipt invalid",
}

# RECOVERY_MAP.tsv
cols = ["path", "bytes", "sha256", "git_blob", "disposition", "repo", "remote_path", "commit", "pr", "on_default_branch", "locator_or_reason", "published_sha256", "scrubbed"]
disp_count = collections.Counter()
with open(os.path.join(STAGE, "RECOVERY_MAP.tsv"), "w") as f:
    f.write("# RECOVERY_MAP for " + SESSION + " local scratch files, generated " + NOW + "; one row per file (directories only for unwalked trees)\n")
    f.write("\t".join(cols) + "\n")
    for r in rows:
        d = r.get("disposition", "")
        disp_count[d] += 1
        loc = ""
        if d == "MATCHED_SESSION_COMMIT":
            loc = f"{GH}{r['repo']}/blob/{r['commit']}/{r['remote_path']}" + (f" (PR #{r['pr']})" if r.get("pr") else "")
        elif d == "MATCHED_REPO_HISTORY":
            loc = f"byte-identical blob in {r['repo']} history at path {r['remote_path']}"
        elif d == "UPLOADED_CHECKPOINT1":
            loc = f"{GH}main/blob/{CK1}/{r['remote_path']}"
        elif d == "UPLOADED_INCREMENT2":
            loc = "this increment: " + r["remote_path"]
        else:
            loc = str(r.get("reason", "")) + ((" | " + str(r["locator"])) if r.get("locator") else "") + ((" | " + r["note"]) if r.get("note") else "")
        vals = [r.get("path", ""), r.get("bytes", ""), r.get("sha256", ""), r.get("blob", ""), d, r.get("repo", ""), r.get("remote_path", "") if d.startswith("MATCHED") else "",
                r.get("commit", ""), r.get("pr", ""), r.get("on_default"), loc.replace("\t", " ").replace("\n", " "), r.get("published_sha256", ""), r.get("scrubbed", "")]
        f.write("\t".join("" if v is None else str(v) for v in vals) + "\n")

json.dump({"generated": NOW, "session": SESSION, "entries": excluded}, open(os.path.join(STAGE, "EXCLUDED.json"), "w"), indent=1)

# MANIFEST2.json (published files; the label files are added below)
by_top = collections.defaultdict(lambda: {"files": 0, "bytes": 0, "scrubbed": 0})
for e in uploaded:
    top = e["top"] if e["published_path"].startswith("scratch/") else e["published_path"].split("/")[1]
    e["area"] = top
    b = by_top[top]; b["files"] += 1; b["bytes"] += e["bytes"]; b["scrubbed"] += int(e["scrubbed"])
label_entries = []
for rel in ("local_lean_evidence/torus_filtration_20261011_replay/LABEL.md", "local_lean_evidence/math_formal_partial_gate/LABEL.md"):
    data = open(os.path.join(STAGE, rel), "rb").read()
    label_entries.append({"published_path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "scrubbed": False, "area": rel.split("/")[1], "kind": "label"})
manifest = {
    "schema": "recovery-increment/1", "generated": NOW, "session": SESSION, "performer": "Anthropic Claude, Claude Code cloud session " + SESSION,
    "owner_request": ["main#307 issuecomment 6136156988", "6136210973", "6136231457", "6136304987"],
    "parent_checkpoint_commit": CK1, "scientific_effect": "NONE", "organizational_independence_credit": 0,
    "status_of_everything_here": "scratch, drafts, local diagnostic outputs and local logs; nothing is a receipt, review record or landed object; local Lean receipts are labeled invalid as receipts",
    "scrubbing": "text files: the container's private scratch path replaced by <SCRATCH> and its UUID by <SESSION-UUID>; nothing else edited; original_sha256 is the as-run byte identity",
    "screening": "excluded before copy: GitHub API responses (locators recorded), binaries, regenerable mutant and clone trees, third-party documents, any file carrying an e-mail address other than the owner's public addresses and well-known no-reply addresses, any Drive locator other than the two public docs/WORKSPACE.md links, any identifier from the excluded main#365 private material; the public-intake guard's credential patterns matched nothing",
    "counts": {"map_rows": len(rows), **disp_count}, "uploaded_files": len(uploaded) + len(label_entries),
    "uploaded_bytes": sum(e["bytes"] for e in uploaded) + sum(e["bytes"] for e in label_entries),
    "areas": {k: {**v, "note": AREA_NOTES.get(k, "working files of this session; see RECOVERY_MAP.tsv")} for k, v in sorted(by_top.items())},
    "files": sorted(uploaded + label_entries, key=lambda e: e["published_path"]),
}
json.dump(manifest, open(os.path.join(STAGE, "MANIFEST2.json"), "w"), indent=1, ensure_ascii=False)

# MANIFEST2.md
lines = []
lines.append("# Recovery increment 2 — Anthropic Claude session `" + SESSION + "`\n")
lines.append(f"Generated {NOW}. Follows checkpoint 1 (commit `{CK1}`, verified by the coordinator in main#307 comment 6136304987) and answers its three asks: map every local file to an exact remote artifact or upload it, preserve local Lean logs with labels, register the checkpoint. Performer: Anthropic Claude, Claude Code cloud session `{SESSION}`, delegated for Dylan Roy. Organizational-independence credit 0. Scientific effect NONE. Everything here is scratch: drafts, probes, local diagnostic outputs, logs. Nothing is a receipt, a review record or a landed object.\n")
lines.append("## 1. The map (`RECOVERY_MAP.tsv`)\n")
lines.append("One row per local file of the session's scratch area (plus one row per unwalked directory). Dispositions:\n")
lines.append("| disposition | rows | meaning |\n|---|---:|---|")
meaning = {"MATCHED_SESSION_COMMIT": "byte-identical to a blob in a commit carrying this session's trailer; column `locator_or_reason` gives repo, commit, path and PR. All 191 Math- and 2 meta-framework session commits are on their default branches; 14 of the 15 main session commits are (the fifteenth is checkpoint 1 on the retention branch)",
           "MATCHED_REPO_HISTORY": "byte-identical to a blob somewhere in the history of Math-, meta-framework or main (every branch and pull-request head fetched on 13 October 01:55Z); path given",
           "UPLOADED_CHECKPOINT1": "same SHA-256 as a file published in checkpoint 1 and not otherwise matched (0 here: the checkpoint-1 originals match commit d579bcc0 directly and appear under MATCHED_SESSION_COMMIT with `on_default_branch` False)", "UPLOADED_INCREMENT2": "published in this increment under `scratch/<original relative path>` or `local_lean_evidence/`",
           "EXCLUDED": "not uploaded; the reason and, where one exists, an exact remote locator are in the row and in `EXCLUDED.json`"}
for k in ("MATCHED_SESSION_COMMIT", "MATCHED_REPO_HISTORY", "UPLOADED_CHECKPOINT1", "UPLOADED_INCREMENT2", "EXCLUDED"):
    lines.append(f"| {k} | {disp_count.get(k, 0)} | {meaning[k]} |")
lines.append("\nA match is a byte identity (git blob id), not a name match: an older draft of a landed file is unmatched and therefore uploaded, with `related_landed_paths` in `MANIFEST2.json` naming landed files of the same name where they exist.\n")
lines.append("## 2. What this increment publishes\n")
lines.append(f"{manifest['uploaded_files']} files, {manifest['uploaded_bytes']:,} bytes, each listed in `MANIFEST2.json` with published SHA-256, as-run SHA-256, byte counts and a scrubbed flag, and in `SHA256SUMS2.txt`. Text files had the container's private scratch path replaced by `<SCRATCH>` and its UUID by `<SESSION-UUID>` ({sum(1 for e in uploaded if e['scrubbed'])} files changed by that alone); nothing else was edited.\n")
lines.append("### 2a. Local Lean logs and receipts (`local_lean_evidence/`), labeled\n")
lines.append("- `torus_filtration_20261011_replay/`: the untracked outputs of the local `companions/torus_filtration_20261011/replay.sh` run of 12 October 22:52–22:57Z at Math- `40440ed8` (Lean sources, build/census/axiom/type logs, negative-control logs, `source.json`, `receipt.json`). **Local, non-authoritative; `receipt.json` invalid as a receipt.** See its `LABEL.md`.")
lines.append("- `math_formal_partial_gate/`: the untracked outputs of the partial local Math- `formal/` gate run of 12 October 22:45–22:47Z at `40440ed8` with the leanchecker step replaced by a logged SKIP after the full gate was OOM-killed. **Not a gate pass; `receipt.json` invalid as a receipt.** See its `LABEL.md`.")
lines.append("- `scratch/gate-run/`, `scratch/gate-run2/`: the two 28 September mutation runs of main `frontiers/downstream_gate_20260925` (REPORT.json, outputs, per-mutant stdout/stderr); mutant trees excluded as regenerable. Software replay only; promotion permission false in both reports.\n")
lines.append("### 2b. Areas (`scratch/<directory>`)\n")
lines.append("| area | files | bytes | what it is |\n|---|---:|---:|---|")
for k, v in sorted(by_top.items(), key=lambda kv: -kv[1]["bytes"]):
    lines.append(f"| `{k}` | {v['files']} | {v['bytes']:,} | {AREA_NOTES.get(k, 'working files of this session; see RECOVERY_MAP.tsv')} |")
lines.append("\nDirectory names follow the pull request, packet or round they served; where a note above is generic, the files themselves and `RECOVERY_MAP.tsv` carry the detail. Posted texts (verdicts, readbacks, bodies) are drafts: the GitHub comment or body is the authoritative record.\n")
lines.append("## 3. Excluded (`EXCLUDED.json`)\n")
reason_counts = collections.Counter()
for e in excluded:
    reason_counts[str(e.get("reason", ""))] += 1
lines.append("| files | reason |\n|---:|---|")
for why, n in reason_counts.most_common():
    lines.append(f"| {n} | {why} |")
lines.append("\nGitHub API responses are listed with the URL of the first and last record they hold, so each can be re-fetched. The main#365 private material (`d365/`) stays excluded: its public identities are on main#365 and its bytes are in the owner-only Drive location recorded in the custody note.\n")
lines.append("## 4. Verify\n")
lines.append("```\ncd recovery/" + SESSION + "/increment2 && sha256sum -c SHA256SUMS2.txt\n```\n`MANIFEST2.json` → `files[].original_sha256` is the as-run identity of each scrubbed file; an unscrubbed file has equal published and original hashes.\n")
lines.append("## 5. Retention, registration, obligations\n")
lines.append(f"- Retention: branch `claude/nice-maxwell-7i7ok0` of `main` is a retention branch, not for merge; retain this branch and its commits (checkpoint 1 `{CK1}` and this increment) while any work relies on them, until a verified durable replacement holds the actual bytes and the shared index is updated; an index alone is not a replacement. No deletion is requested.")
lines.append("- Registration: one screened Work Events row for the existing R17 register is requested from the appending lane in the hub post that announces this commit; this session has no native append route and makes no whole-file register write.")
lines.append("- Obligations held by this session: none open beyond the standing offers in checkpoint 1 section D. Math-#444 delta read follows if routed.")
lines.append("- Blockers: none. Platform limits unchanged (no scheduling from this session; hosted job logs not fetchable).\n")
open(os.path.join(STAGE, "MANIFEST2.md"), "w").write("\n".join(lines))

# SHA256SUMS2.txt over every published file except itself
sums = []
for dp, dns, fns in os.walk(STAGE):
    for fn in sorted(fns):
        p = os.path.join(dp, fn); rel = os.path.relpath(p, STAGE)
        if rel == "SHA256SUMS2.txt":
            continue
        sums.append((hashlib.sha256(open(p, "rb").read()).hexdigest(), rel))
sums.sort(key=lambda t: t[1])
with open(os.path.join(STAGE, "SHA256SUMS2.txt"), "w") as f:
    for h, rel in sums:
        f.write(f"{h}  ./{rel}\n")
print("published files incl. manifests:", len(sums) + 1, "disp:", dict(disp_count))
print("manifest uploaded_files:", manifest["uploaded_files"], "bytes:", manifest["uploaded_bytes"])
