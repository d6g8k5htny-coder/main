#!/usr/bin/env python3
"""Check tranche-5 Lean records against Math- bytes and assemble proposals.

Usage: python3 -I -B assemble_t5.py MATH_REPO FACTS_DIR VERDICTS_JSON OUTPUT_JSON

FACTS_DIR holds lbatch<N>.json. Fails (exit 1, nothing written) on a missing, duplicate or unknown group; a statement whose file
is not one of the group's Lean files, or whose text differs from that file's lines at REF; a record with no statements unless
NOT_OBLIGATION; a floating_point field not starting with USED, NONE or NO_SCRIPTS. Bindings (every Lean file of the group, with
blob, bytes and sha256) come only from Git.
"""
import glob, hashlib, json, os, subprocess, sys

REF = '3b11bff6479f3f40ac838fe1d8b87c7ed6814606'
REPO = 'd6g8k5htny-coder/Math-'
# Groups another lane holds: recorded with their verified record, but not proposed, so nothing competes with that lane.
HELD = {'frontiers/source_derivative_gaussian_20261025':
        'Math-#514 (head 86c00d4c, in integration on 2 Nov; all 10,998 changed paths are inside this directory; hosted '
        'formal / formal-evidence succeeded at that head). Its post-merge state supersedes this record.'}


def receipt_check(math_repo, group):
    """For every RECEIPT.json under the group: each source_before digest must equal the sha256 of a Math- file at REF with that
    file name; kernel receipts are leanchecker commands with returncode 0, no timeout, no integrity error, sources unchanged."""
    tree = [l.split('\t') for l in subprocess.run(['git', '-C', math_repo, 'ls-tree', '-r', REF], capture_output=True,
                                                   text=True).stdout.split('\n') if l]
    by_sha = {}
    for meta, p in tree:
        if p.endswith('.lean'):
            by_sha.setdefault(hashlib.sha256(blob_bytes(math_repo, p)).hexdigest(), []).append(p)
    receipts = [p for _, p in tree if p.startswith(group + '/') and p.endswith('RECEIPT.json')]
    out = {'receipts': len(receipts), 'digests_checked': 0, 'digests_unmatched': [], 'kernel_ok': [], 'kernel_failed': []}
    for p in receipts:
        d = json.loads(blob_bytes(math_repo, p))
        for name, digest in (d.get('source_before') or {}).items():
            out['digests_checked'] += 1
            if not any(q.endswith('/' + name) for q in by_sha.get(digest, [])):
                out['digests_unmatched'].append(f'{p}: {name}')
        cmd = ' '.join(d['command']) if isinstance(d.get('command'), list) else str(d.get('command'))
        if 'leanchecker' in cmd:
            good = d.get('returncode') == 0 and not d.get('timed_out') and d.get('integrity_error') is None and d.get('source_unchanged')
            (out['kernel_ok'] if good else out['kernel_failed']).append({'receipt': p, 'module': d.get('module'), 'returncode': d.get('returncode')})
    return out


def blob_bytes(repo, path):
    return subprocess.run(['git', '-C', repo, 'cat-file', 'blob', f'{REF}:{path}'], capture_output=True, check=True).stdout


def main(math_repo, facts_dir, verdicts_path, output):
    facts = {}
    for f in sorted(glob.glob(os.path.join(facts_dir, 'lbatch*.json'))):
        for g in json.load(open(f))['groups']:
            facts[g['group']] = g
    verdicts = [v for b in json.load(open(verdicts_path))['batches'] for v in (b['verdicts'] or [])]
    problems, seen = [], {}
    for v in verdicts:
        o = v['record']['object']
        if o in seen:
            problems.append(f'duplicate: {o}')
        seen[o] = v
    problems += [f'missing: {o}' for o in sorted(set(facts) - set(seen))]
    problems += [f'unknown: {o}' for o in sorted(set(seen) - set(facts))]
    quarantines, closed_candidates, not_obligations, held = [], [], [], []
    for g in sorted(facts):
        if g not in seen:
            continue
        v, fact = seen[g], facts[g]
        rec = v['record']
        lean = {f['path']: f for f in fact['lean_files']}
        for st in rec['statements']:
            if st['file'] not in lean:
                problems.append(f'{g}: statement file {st["file"]} is not a Lean file of the group')
                continue
            lines = blob_bytes(math_repo, st['file']).decode('utf-8').split('\n')
            a, b = st['line_start'], st['line_end']
            if not (1 <= a <= b <= len(lines)) or '\n'.join(lines[a - 1:b]) != st['text'] or not st['text'].strip():
                problems.append(f'{g}: statement {st["file"]}:{a}-{b} is not verbatim')
        disp = rec['disposition_proposed']
        if disp != 'NOT_OBLIGATION' and not rec['statements']:
            problems.append(f'{g}: {disp} without statements')
        if rec['floating_point'].split()[0].rstrip(':;,.') not in ('USED', 'NONE', 'NO_SCRIPTS'):
            problems.append(f'{g}: floating_point unclassified')
        binding = {'repository': REPO, 'commit': REF, 'lean_files': []}
        for p, f in sorted(lean.items()):
            data = blob_bytes(math_repo, p)
            blob = subprocess.run(['git', '-C', math_repo, 'rev-parse', f'{REF}:{p}'], capture_output=True, text=True).stdout.strip()
            if blob != f['git_blob']:
                problems.append(f'{g}: {p} blob differs from facts')
            binding['lean_files'].append({'path': p, 'git_blob': blob, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                                          'theorems': f['theorems'], 'identical_copies_elsewhere': f['identical_copies_elsewhere']})
        body = {'title': rec['title'], 'location': [f'{REPO}@{REF}:{g}'], 'disposition_proposed': disp,
                'layer_a_status': rec['layer_a_status'], 'exact_boundary': rec['exact_boundary'],
                'blocking_reason': rec['blocking_reason'], 'attempted': rec['attempted'], 'resumption_cost': rec['resumption_cost'],
                'owner': 'none', 'maintenance': 'zero', 'source_statements': rec['statements'],
                'placeholder_scan': rec['placeholder_scan'], 'vacuity_notes': rec['vacuity_notes'],
                'own_kernel_evidence': rec['own_kernel_evidence'], 'vendored_copies': rec['vendored_copies'],
                'source_hypotheses': rec['hypotheses'], 'source_dependencies': rec['dependencies'],
                'source_self_limits': rec['source_self_limits'], 'finite_checks': rec['finite_checks'],
                'floating_point': rec['floating_point'], 'lean_toolchain': fact['lean_toolchain'], 'lake_root': fact['lake_root'],
                'source_binding': binding, 'verification': {'ok': v['ok'], 'problems_fixed': v['problems']}}
        if g in HELD:
            body['held_by'] = HELD[g]
            held.append(body)
        elif disp == 'QUARANTINE_PENDING_KERNEL_REPLAY':
            quarantines.append(body)
        elif disp == 'CLOSED_CANDIDATE_OWN_RECEIPT':
            body['receipt_check'] = rc = receipt_check(math_repo, g)
            if rc['digests_unmatched'] or not rc['kernel_ok']:
                problems.append(f'{g}: closed candidate whose receipts do not bind the current bytes or show no passing kernel run')
            closed_candidates.append(body)
        else:
            not_obligations.append(body)
    if problems:
        print('\n'.join(problems)); print(f'{len(problems)} problem(s); nothing written')
        return 1
    for i, q in enumerate(quarantines, 1):
        q.update({'id': f'QP-C54-T5-{i:03d}', 'status': 'QUARANTINED', 'proposal_status': 'PROPOSED', 'proposal_kind': 'NEW_RECORD',
                  'isolation': ('These Lean modules host no CLOSED declaration and no indexed kernel receipt; no CLOSED root may import '
                                'them. A pinned kernel replay with a receipt can reopen this record deliberately and CLOSE literal '
                                'declarations.')})
    with open(output, 'x') as s:
        json.dump({'schema': 'claude54.quarantine-proposals.t5-lean.v1', 'math_ref': REF, 'records': quarantines,
                   'closed_candidates': closed_candidates, 'not_obligations': not_obligations, 'held_by_other_lane': held},
                  s, indent=1, ensure_ascii=False)
        s.write('\n')
    print(f'{len(quarantines)} quarantine proposals; {len(closed_candidates)} closed candidates; {len(not_obligations)} not obligations; '
          f'{len(held)} held by another lane')
    for c in closed_candidates:
        rc = c['receipt_check']
        print(f"closed candidate {c['location'][0].split(':')[-1]}: {rc['receipts']} receipts, {rc['digests_checked']} digests, "
              f"{len(rc['digests_unmatched'])} unmatched, kernel ok {[k['module'] for k in rc['kernel_ok']]}, failed {[k['module'] for k in rc['kernel_failed']]}")
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
