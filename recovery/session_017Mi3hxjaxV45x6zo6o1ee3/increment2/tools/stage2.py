"""Stage recovery increment 2 for session 017Mi3hxjaxV45x6zo6o1ee3 (standard library only).

Every local scratch file gets exactly one disposition:
  MATCHED_SESSION_COMMIT  byte-identical to a blob in a commit carrying this session's trailer (repo, path, commit, PR)
  MATCHED_REPO_HISTORY    byte-identical to a blob somewhere in Math-, meta-framework or main history
  UPLOADED_CHECKPOINT1    already published in commit d579bcc0 (same SHA-256)
  UPLOADED_INCREMENT2     copied into increment2/scratch/<original relative path> (scratch root scrubbed)
  EXCLUDED                not uploaded; reason and, where one exists, an exact remote locator are recorded
"""
import collections
import hashlib
import json
import os
import re
import shutil
import sys
import time

SCRATCH = os.path.abspath(sys.argv[1])
STAGE = os.path.abspath(sys.argv[2])
REC2 = os.path.join(SCRATCH, "rec2")
SESSION = "session_017Mi3hxjaxV45x6zo6o1ee3"
SCRATCH_UUID = os.path.basename(os.path.dirname(SCRATCH))  # <SESSION-UUID>
CHECKPOINT1 = "d579bcc05aa8173986fdb2634fbcf75df7e9f53b"

EXCLUDE_DIRS = {  # top-level directories never walked: clones, caches, private material, this tool's own output
    "mathf": "clone of Math- (regenerable; its untracked local Lean evidence is published separately below)",
    "mfrepo": "clone of meta-framework (regenerable)",
    ".venv-audit": "Python virtual environment (third-party packages)",
    "pylib": "third-party Python package copies (pypdf)",
    "d365": "main#365 private delivery material: owner-only Drive locators and permission observations; "
            "public identities on main#365 (bundle 19034 B afd1fff8...)",
    "rec2": "this increment's build directory",
    "recovery": "checkpoint-1 staging copy (published as commit " + CHECKPOINT1 + ")",
}
EXCLUDE_SUBTREES = {  # regenerable derived trees inside walked directories
    "rt/downstream-gate-results/mutants": "mutation run output tree (each mutant is the packet with one mutation applied; regenerable from the packet named in rt/downstream-gate-results/REPORT.json)",
    "rtfOEb/downstream-gate-results/mutants": "mutation run output tree (regenerable, see rtfOEb/downstream-gate-results/REPORT.json)",
    "gate-run/mutants": "mutation run output tree (regenerable from frontiers/downstream_gate_20260925 at the hashes in gate-run/REPORT.json)",
    "gate-run2/mutants": "mutation run output tree (regenerable, see gate-run2/REPORT.json)",
    "mut": "mutant copies of main packets used for local mutation checks (regenerable)",
    "dl_r17": "downloaded CSV exports of the shared R17 registers (the registers themselves are the source; linked from docs/WORKSPACE.md)",
    "mf15/mathman": "local copy of repository material used by the meta-framework#15 parity probe (regenerable)",
    "leanbuild/formal/.lake": "Lean build products (generated IR and setup files; regenerable; also matched by the repository .gitignore rule .lake/)",
}
THIRD_PARTY_FILES = {
    "aal_v3.pdf": "owner-supplied third-party document (the owner holds the original); not project work product",
    "aal_v4.pdf": "owner-supplied third-party document (the owner holds the original); not project work product",
    "aal_v3.txt": "text extraction of the third-party document above; contains its authors' e-mail addresses",
    "aal_v4.txt": "text extraction of the third-party document above; contains its authors' e-mail addresses",
    "aal_v3.html": "HTML rendering of the third-party document above",
    "aal_v4.html": "HTML rendering of the third-party document above",
    "toolchain/Cache_Hashing.lean": "copy of the Mathlib cache tool source (upstream Mathlib; regenerable)",
    "toolchain/Cache_IO.lean": "copy of the Mathlib cache tool source (upstream Mathlib; regenerable)",
    "toolchain/Cache_Main.lean": "copy of the Mathlib cache tool source (upstream Mathlib; regenerable)",
    "toolchain/Cache_Requests.lean": "copy of the Mathlib cache tool source (upstream Mathlib; regenerable)",
}
BINARY_EXT = {".pdf", ".olean", ".zip", ".pyc", ".npz", ".png", ".ilean", ".trace", ".hash"}
PUBLIC_DRIVE_IDS = {"180yfvocozAaFRxf7tY8CDrobnpi17Sv-UkQGBBWCiD8", "1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no"}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SAFE_EMAIL = re.compile(r"(example\.(invalid|com|org|net)$|^noreply@anthropic\.com$|users\.noreply\.github\.com$|^noreply@github\.com$|^th3realdyll@gmail\.com$)")
DRIVE = re.compile(r"(?:drive\.google\.com|docs\.google\.com)/[^\s\"')>\]]*")
DRIVE_ID = re.compile(r"/(?:d|folders|file/d)/([A-Za-z0-9_-]{20,})")
CREDENTIALS = [  # the public-intake guard's own patterns (tools/public_intake_check.py)
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    re.compile(rb"(?<![0-9A-Za-z_-])AIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])"),
    re.compile(rb"\bnpm_[A-Za-z0-9]{36}\b"),
    re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bsk_live_[0-9A-Za-z]{20,}\b"),
    re.compile(rb"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"),
    re.compile(rb"hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+"),
]
API_KEYS = {"node_id", "html_url", "url", "user", "created_at"}

private = json.load(open(os.path.join(REC2, "private_markers.json")))
PRIVATE_EMAILS = set(private["emails"])
PRIVATE_IDS = set(private["ids"])
blobmaps = {n: json.load(open(os.path.join(REC2, f"blobmap_{n}.json"))) for n in ("Math-", "meta-framework", "main")}
prev = {e["blob"]: e for e in json.load(open(os.path.join(REC2, "local_map.json")))}
ck1 = {}
for line in open(os.path.join(SCRATCH, "recovery/stage/SHA256SUMS.txt")):
    h, p = line.split(maxsplit=1)
    ck1[h] = p.strip().lstrip("./")


def git_blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def is_api_dump(data, path):
    if not path.endswith((".json", ".jsonl")):
        return None
    try:
        txt = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        obj = json.loads(txt)
    except ValueError:
        if path.endswith(".jsonl"):
            first = txt.split("\n", 1)[0]
            try:
                obj = [json.loads(first)]
            except ValueError:
                return None
        else:
            return None
    probe = obj[0] if isinstance(obj, list) and obj and isinstance(obj[0], dict) else obj
    if isinstance(probe, dict) and len(API_KEYS & set(probe)) >= 3:
        loc = probe.get("html_url") or probe.get("url")
        if isinstance(obj, list):
            last = obj[-1] if isinstance(obj[-1], dict) else {}
            loc = f"{loc} .. {last.get('html_url') or last.get('url')} ({len(obj)} records)"
        return str(loc)
    return None


def privacy_issues(text):
    issues = []
    emails = sorted({e for e in EMAIL.findall(text) if not SAFE_EMAIL.search(e)})
    if emails:
        issues.append(f"third-party e-mail address ({len(emails)})")
    if "privaterelay" in text:
        issues.append("relay address")
    for m in DRIVE.findall(text):
        ids = DRIVE_ID.findall("/" + m)
        if any(i not in PUBLIC_DRIVE_IDS for i in ids) or (not ids and "folders" in m):
            issues.append("Drive locator other than the public WORKSPACE links")
            break
    if any(e in text for e in PRIVATE_EMAILS) or any(i in text for i in PRIVATE_IDS):
        issues.append("identifier from the excluded main#365 private material")
    return issues


def credential_hits(data):
    return [i for i, pat in enumerate(CREDENTIALS) if pat.search(data)]


def scrub(text):
    out = text.replace(SCRATCH, "<SCRATCH>").replace(SCRATCH_UUID, "<SESSION-UUID>")
    return out, out != text


rows = []            # RECOVERY_MAP rows
uploaded = []        # MANIFEST2 entries
excluded = []
counts = collections.Counter()
bytes_by = collections.Counter()
cred_review = []

def dispose(rel, data, area_root):
    """Classify one file; copy it when it is to be uploaded."""
    blob = git_blob(data); h = sha256(data); size = len(data)
    row = {"path": rel, "bytes": size, "sha256": h, "blob": blob}
    top = rel.split("/")[0] if "/" in rel else "(root)"
    # 1-2 exact remote matches
    m = prev.get(blob, {}).get("match") if blob in prev else None
    if m is None:
        for name in blobmaps:
            if blob in blobmaps[name]:
                m = {"kind": "repo-history", "repo": name, "path": blobmaps[name][blob]}; break
    if m and m["kind"] == "session-commit":
        row.update(disposition="MATCHED_SESSION_COMMIT", repo=m["repo"], remote_path=m["path"], commit=m["commit"],
                   pr=m.get("pr"), on_default=m.get("on_default"))
        counts["MATCHED_SESSION_COMMIT"] += 1; return row
    if m and m["kind"] == "repo-history":
        row.update(disposition="MATCHED_REPO_HISTORY", repo=m["repo"], remote_path=m["path"])
        counts["MATCHED_REPO_HISTORY"] += 1; return row
    if h in ck1:
        row.update(disposition="UPLOADED_CHECKPOINT1", remote_path="recovery/" + SESSION + "/" + ck1[h], commit=CHECKPOINT1)
        counts["UPLOADED_CHECKPOINT1"] += 1; return row
    # exclusions
    for sub, why in EXCLUDE_SUBTREES.items():
        if rel == sub or rel.startswith(sub + "/"):
            row.update(disposition="EXCLUDED", reason=why); excluded.append(row); counts["EXCLUDED"] += 1; return row
    if rel in THIRD_PARTY_FILES:
        row.update(disposition="EXCLUDED", reason=THIRD_PARTY_FILES[rel]); excluded.append(row); counts["EXCLUDED"] += 1; return row
    ext = os.path.splitext(rel)[1].lower()
    if ext in BINARY_EXT or (rel.endswith(".git") and size < 200):
        row.update(disposition="EXCLUDED", reason=f"binary or checkout pointer ({ext or 'no extension'}); regenerable")
        excluded.append(row); counts["EXCLUDED"] += 1; return row
    loc = is_api_dump(data, rel)
    if loc:
        row.update(disposition="EXCLUDED", reason="GitHub API response saved during work; regenerable from the API", locator=loc)
        excluded.append(row); counts["EXCLUDED"] += 1; return row
    hits = credential_hits(data)
    if hits:
        cred_review.append((rel, hits))
    binary = b"\0" in data[:8192]
    if binary:
        text = None
        issues = []
    else:
        text = data.decode("utf-8", "replace")
        issues = privacy_issues(text)
    if issues:
        row.update(disposition="EXCLUDED", reason="privacy screen: " + "; ".join(issues))
        excluded.append(row); counts["EXCLUDED"] += 1; return row
    # upload
    if binary:
        pub = data; scrubbed = False
    else:
        out, scrubbed = scrub(text)
        pub = out.encode("utf-8")
    dest_rel = f"{area_root}/{rel}"
    dest = os.path.join(STAGE, dest_rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(pub)
    entry = {"published_path": dest_rel, "bytes": len(pub), "sha256": sha256(pub), "original_sha256": h,
             "original_bytes": size, "scrubbed": scrubbed, "original_relpath": rel, "top": top}
    rel_landed = prev.get(blob, {}).get("superseded_by_candidates")
    if rel_landed:
        entry["related_landed_paths"] = rel_landed
    uploaded.append(entry)
    row.update(disposition="UPLOADED_INCREMENT2", remote_path="recovery/" + SESSION + "/increment2/" + dest_rel,
               scrubbed=scrubbed, published_sha256=entry["sha256"])
    counts["UPLOADED_INCREMENT2"] += 1; bytes_by[top] += len(pub)
    return row


if os.path.exists(STAGE):
    shutil.rmtree(STAGE)
os.makedirs(STAGE)
started = time.time()
# walk the scratch area
for dp, dns, fns in os.walk(SCRATCH):
    reld = os.path.relpath(dp, SCRATCH)
    if reld == ".":
        dns[:] = sorted(d for d in dns if d not in EXCLUDE_DIRS and d != "__pycache__")
    else:
        dns[:] = sorted(d for d in dns if d not in ("__pycache__", ".git"))
    for fn in sorted(fns):
        p = os.path.join(dp, fn)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        rel = os.path.relpath(p, SCRATCH)
        try:
            data = open(p, "rb").read()
        except OSError as exc:
            rows.append({"path": rel, "disposition": "EXCLUDED", "reason": f"unreadable: {exc}"}); continue
        rows.append(dispose(rel, data, "scratch"))
for top, why in EXCLUDE_DIRS.items():
    if os.path.isdir(os.path.join(SCRATCH, top)):
        n = sum(len(f) for _, _, f in os.walk(os.path.join(SCRATCH, top)))
        rows.append({"path": top + "/", "disposition": "EXCLUDED", "reason": why, "bytes": None, "note": f"{n} files, directory not walked"})
        excluded.append(rows[-1])
# local Lean evidence trees from the Math- clone (untracked local outputs)
LEAN = {
    "mathf/.lake/torus-filtration-evidence": "local_lean_evidence/torus_filtration_20261011_replay",
    "mathf/formal/.lake/formal-evidence": "local_lean_evidence/math_formal_partial_gate",
}
for src, dst in LEAN.items():
    base = os.path.join(SCRATCH, src)
    for dp, dns, fns in os.walk(base):
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, base)
            data = open(p, "rb").read()
            r = dispose(src + "/" + rel, data, dst.rsplit("/", 1)[0] + "/" + dst.rsplit("/", 1)[1])
            # dispose() nests under <area_root>/<rel>; rewrite to the flat evidence path
            rows.append(r)
# flatten: dispose() placed lean files at local_lean_evidence/<name>/mathf/.lake/.../file; move them up
for e in uploaded:
    if e["published_path"].startswith("local_lean_evidence/"):
        parts = e["published_path"].split("/")
        name = parts[1]
        src_prefix = [k for k in LEAN if e["original_relpath"].startswith(k + "/")][0]
        new_rel = f"local_lean_evidence/{name}/" + e["original_relpath"][len(src_prefix) + 1:]
        old = os.path.join(STAGE, e["published_path"]); new = os.path.join(STAGE, new_rel)
        os.makedirs(os.path.dirname(new), exist_ok=True)
        os.replace(old, new)
        e["published_path"] = new_rel
        for r in rows:
            if r.get("path") == e["original_relpath"]:
                r["remote_path"] = "recovery/" + SESSION + "/increment2/" + new_rel
# remove emptied nested dirs
for dp, dns, fns in os.walk(os.path.join(STAGE, "local_lean_evidence"), topdown=False):
    if not dns and not fns:
        os.rmdir(dp)
json.dump({"rows": rows, "uploaded": uploaded, "excluded": excluded, "counts": counts, "bytes_by_top": bytes_by,
           "credential_pattern_hits": cred_review, "seconds": round(time.time() - started, 1)},
          open(os.path.join(REC2, "stage2_result.json"), "w"), indent=1)
print("counts:", dict(counts))
print("uploaded bytes:", sum(e["bytes"] for e in uploaded), "files:", len(uploaded))
print("credential-pattern hits to inspect:", cred_review)
print("excluded reasons:")
for why, n in collections.Counter(e.get("reason", "")[:70] for e in excluded).most_common(20):
    print(f"  {n:5d} {why}")
print("uploaded bytes by top dir:")
for top, b in bytes_by.most_common(25):
    print(f"  {b:9d} {top}")
