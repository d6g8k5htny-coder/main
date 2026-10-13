#!/usr/bin/env python3
"""Curated, scrubbed reading copy of this session's still-local project scratch (owner request main#307 6136156988).
Copies text and small binary files from the listed scratch areas into pub/scratch/<original relative path>, replacing the
private scratch root and session UUID in text files, and writes MANIFEST.json (published and as-run sha256 per file) plus
EXCLUDED.json (repo/tree copies, caches, symlinks, oversized or binary-with-private-bytes files, each with size and sha256).
Nothing here is a receipt; labels are not changed by this copy."""
import hashlib, json, os, re, sys
from pathlib import Path
S = Path(sys.argv[1]); PUB = S / 'recovery/pub'; OUT = PUB / 'scratch'
AREAS = {
 'renderer_versions_1_9': ['lvs/spec_doc', 'lvs/register_update7', 'lvs/register_update8', 'lvs/register_update9'],
 'binders_rounds_1_9': ['lvs/register_update2', 'lvs/register_update5', 'lvs/register_update6'],
 'binders_rounds_10_14_with_logs': ['lvs/register_update10', 'lvs/register_update11', 'lvs/register_update12', 'lvs/register_update13', 'lvs/register_update14'],
 'round4_triage_probes': ['lvs/round4'],
 'proof_packet_rounds_5_6_verification': ['lvs/r6commit', 'r6', 'r6d2', 'r6verify', 'verify_r5', 'verify_r5reg', 'verif_r5d', 'r5cone_verify_py', 'r5cov_verify_scripts', 'vr9', 'vr9log', 'verify13d', 'verify_reg11', 'side24', 'p15', 'dv2'],
 'museum_review_probes': ['museum/probes', 'museum/synth', 'a2'],
 'torus_companion_c254_c26x': ['lvs/c254', 'lvs/c262', 'lvs/c263', 'lvs/c264', 'lvs/c264_controls', 'lvs/c264_controls_integ', 'lvs/c26x', 'lvs/c26x_fix', 'lvs/c419', 'lvs/c428push', 'verify_c26x', 'c26x_verify_old', 'verif_ec7', 'verif_logs', 'verif_tools', 'review_cov'],
 'lvs_misc_notes_and_fixes': ['lvs/bind358', 'lvs/bind434', 'lvs/body428', 'lvs/crawl', 'lvs/elab', 'lvs/field_lifetime_work', 'lvs/fixer_fl', 'lvs/fixer_infra', 'lvs/fixer_rn', 'lvs/gov', 'lvs/guide_row', 'lvs/infra_copy', 'lvs/infra_fixture', 'lvs/infra_scratch'],
 'frontier_round_and_handoff_tooling': ['lvs/r7', 'handoff'],
 'recovery_tooling': ['recovery'],
}
LVS_TOP_FILES = True  # also the loose files directly under lvs/
EXCL_DIR = {'.lake', 'node_modules', '__pycache__', '.git', 'wt', 'wt2', 'wt3', 'base_wt', 'idx_wt', 'idx_wt2', 'verify_pub', 'pp', 'clone', 'mirror'}
EXCL_REL = {'recovery/pub'}
EXCL_EXT = {'.olean', '.ilean', '.pyc', '.o', '.so', '.trace', '.hash'}
MAXB = 20 * 1024 * 1024
ROOT = str(S); UUID = '<SESSION-UUID>'
PRIV = re.compile(rb'<OWNER-RELAY-LOCALPART>|privaterelay|' + UUID.encode())
CRED = [re.compile(p) for p in (rb'-----BEGIN [A-Z ]*PRIVATE KEY', rb'\bgh[pousr]_[A-Za-z0-9]{36,}\b', rb'\bgithub_pat_[A-Za-z0-9_]{60,}\b', rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b', rb'\bxox[baprs]-[A-Za-z0-9-]{10,}\b', rb'\bAKIA[A-Z0-9]{16}\b')]
def is_repo_copy(d):
    try: names = set(os.listdir(d))
    except OSError: return False
    return '.git' in names or 'lakefile.toml' in names or 'lake-manifest.json' in names or ('tools' in names and 'tests' in names and ('docs' in names or 'engine' in names or 'claims' in names))
def sha(b): return hashlib.sha256(b).hexdigest()
manifest, excluded = [], []
def kind_of(p):
    e = p.suffix.lower()
    return {'.py': 'script', '.sh': 'script', '.mjs': 'script', '.js': 'script', '.lean': 'lean', '.log': 'log', '.out': 'output', '.stdout': 'output', '.err': 'output', '.exit': 'output', '.status': 'output', '.json': 'data', '.csv': 'data', '.md': 'note', '.txt': 'note', '.msg': 'note', '.diff': 'patch', '.patch': 'patch', '.yml': 'config', '.toml': 'config'}.get(e, 'other')
def handle(src: Path, rel: str, area: str):
    if src.is_symlink():
        excluded.append({'path': rel, 'reason': 'symlink', 'target': os.readlink(src)}); return
    if src.suffix.lower() in EXCL_EXT:
        excluded.append({'path': rel, 'reason': 'build artifact (' + src.suffix + ')'}); return
    raw = src.read_bytes(); n = len(raw)
    if n > MAXB:
        excluded.append({'path': rel, 'reason': f'oversized ({n} bytes)', 'bytes': n, 'sha256': sha(raw)}); return
    if any(c.search(raw.replace(b'sk-item-4-instantiated-ordinary-h0-controls', b'ITEM4-SLUG')) for c in CRED):  # known task slug, not a key
        excluded.append({'path': rel, 'reason': 'credential-shaped bytes', 'bytes': n, 'sha256': sha(raw)}); return
    asrun = sha(raw); scrubbed = False
    try:
        text = raw.decode('utf-8'); is_text = True
    except UnicodeDecodeError:
        is_text = False
    if is_text:
        new = text.replace(ROOT, '<SCRATCH>').replace(UUID, '<SESSION-UUID>').replace('<OWNER-RELAY-EMAIL>', '<OWNER-RELAY-EMAIL>').replace('<OWNER-RELAY-LOCALPART>', '<OWNER-RELAY-LOCALPART>')
        if new != text: scrubbed = True
        out = new.encode('utf-8')
    else:
        if PRIV.search(raw):
            excluded.append({'path': rel, 'reason': 'binary with private bytes', 'bytes': n, 'sha256': asrun}); return
        out = raw
    assert UUID.encode() not in out and b'<OWNER-RELAY-LOCALPART>' not in out, rel
    dst = OUT / rel; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(out)
    manifest.append({'path': 'scratch/' + rel, 'area': area, 'kind': kind_of(src), 'bytes': len(out), 'sha256': sha(out),
                     'as_run_bytes': n, 'as_run_sha256': asrun, 'scrubbed': scrubbed})
for area, dirs in AREAS.items():
    for d in dirs:
        root = S / d
        if not root.is_dir():
            excluded.append({'path': d, 'reason': 'listed area directory absent'}); continue
        for dp, dn, fn in os.walk(root):
            keep = []
            for x in sorted(dn):
                p = Path(dp) / x; rel = str(p.relative_to(S))
                if x in EXCL_DIR or x.startswith('wt_') or rel in EXCL_REL:
                    excluded.append({'path': rel, 'reason': 'excluded directory (cache, clone, worktree or extracted tree)'}); continue
                if p.is_symlink():
                    excluded.append({'path': rel, 'reason': 'symlinked directory', 'target': os.readlink(p)}); continue
                if is_repo_copy(p):
                    excluded.append({'path': rel, 'reason': 'repository/tree copy (reproducible from the pinned commits named in its parent notes)'}); continue
                keep.append(x)
            dn[:] = keep
            for f in sorted(fn):
                handle(Path(dp) / f, str((Path(dp) / f).relative_to(S)), area)
if LVS_TOP_FILES:
    for f in sorted((S / 'lvs').iterdir()):
        if f.is_file(): handle(f, str(f.relative_to(S)), 'lvs_top_level_notes')
PUB.mkdir(parents=True, exist_ok=True)
by_area = {}
for m in manifest:
    a = by_area.setdefault(m['area'], {'files': 0, 'bytes': 0, 'scrubbed': 0}); a['files'] += 1; a['bytes'] += m['bytes']; a['scrubbed'] += m['scrubbed']
(PUB / 'MANIFEST.json').write_text(json.dumps({'note': 'Reading copy of scratch work. sha256/bytes identify the published copy; as_run_* the exact bytes this session executed or wrote. scrubbed=true means only the private scratch root, session UUID or owner relay address were replaced by placeholders. Nothing here is a receipt or evidence of any claim.', 'areas': by_area, 'files': manifest}, indent=1) + '\n')
(PUB / 'EXCLUDED.json').write_text(json.dumps({'note': 'Not copied: caches, clones, worktrees and extracted repository trees (reproducible from the commits their notes name), build artifacts, symlinks, oversized files and anything carrying credential-shaped or private bytes.', 'entries': excluded}, indent=1) + '\n')
print(json.dumps(by_area, indent=1)); print('files', len(manifest), 'bytes', sum(m['bytes'] for m in manifest), 'excluded entries', len(excluded))
print('excluded reasons:', json.dumps({r: sum(1 for e in excluded if e['reason'].startswith(r)) for r in ('symlink', 'build artifact', 'oversized', 'credential', 'binary with private', 'excluded directory', 'repository/tree copy', 'listed area directory absent', 'symlinked directory')}))
