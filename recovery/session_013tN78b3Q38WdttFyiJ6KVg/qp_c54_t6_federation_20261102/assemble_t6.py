#!/usr/bin/env python3
"""Check tranche-6 items against federation-repository bytes and assemble catalog-schema proposals.

Usage: python3 -I -B assemble_t6.py MAIN_REPO MATH_REPO FACTS_DIR OBJECTS_JSON CANDIDATE_CATALOG CLOSED_JSON VERDICTS_JSON OUTPUT_JSON

FACTS_DIR holds fbatch<N>.json; OBJECTS_JSON is fed_objects.json (full, untruncated file lists, with clone and ref per object).

FACTS_DIR holds tbatch<N>.json (the objects given to the workflow). OBJECTS_JSON is objects.json (all 76 objects, including the
four resolved mechanically). VERDICTS_JSON is the workflow result {"batches": [{"batch", "kind", "verdicts": [{"object", "ok",
"problems", "items"}]}]}.

Fails (exit 1, nothing written) on: a missing, duplicate or unknown object; an item whose files are not a partition of the
object's files (a file covered by no item or by two); duplicate item ids within an object; an OBLIGATION without statements; a
statement whose file is not in its item or whose text differs from that file's lines at REF; an OBLIGATION whose floating_point does
not start with USED, NONE or NO_SCRIPTS; a REVIEW_OF or COPY_OR_EXPOSITION_OF item with no refers_to. Bindings come only from Git.
Each refers_to entry that is a repository path is resolved against main at REF and Math- at MATH_REF and, when it exists, against
the candidate catalog's dispositions; unresolved or undisposed referents are reported, not failed.
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

REF = 'd2b0216e15d545c47f25b6a9265dd3298971e9fe'  # main, for resolving refers_to only
MATH_REF = '3b11bff6479f3f40ac838fe1d8b87c7ed6814606'


def git_bytes(repo, commit, path):
    r = subprocess.run(['git', '-C', repo, 'cat-file', 'blob', f'{commit}:{path}'], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def git_exists(repo, commit, path):
    return subprocess.run(['git', '-C', repo, 'cat-file', '-e', f'{commit}:{path.rstrip("/")}'], capture_output=True).returncode == 0


def disposition_index(cand, closed_main_paths):
    idx = {}
    for key, rows in (('ADOPTED', cand['quarantines']), ('PROPOSED_QP_C54', cand['proposed_quarantines'] + cand['proposed_pinning_amendments'])):
        for r in rows:
            for loc in (r['location'] if isinstance(r['location'], list) else [r['location']]):
                if not isinstance(loc, str):
                    continue
                m = re.match(r'(d6g8k5htny-coder/(?:main|Math-))@[0-9a-f]{40}:([^#]+)', loc) or \
                    re.match(r'https://github.com/(d6g8k5htny-coder/(?:main|Math-))/blob/[0-9a-f]{40}/([^#]+)', loc)
                if m:
                    idx.setdefault((m.group(1).split('/')[1], m.group(2).rstrip('/')), []).append(f'{key}:{r["id"]}')
    for r in cand['main_census_independent_validation']:
        idx.setdefault(('main', r['path']), []).append(f'ADOPTED(main census):{r["id"]}')
    for n in cand['proposed_non_obligations']:
        idx.setdefault(('Math-', n['object'].rstrip('/')), []).append(f'NOT_OBLIGATION_QP_C54:{n["object"]}')
    for p in closed_main_paths:
        idx.setdefault(('main', p), []).append('CLOSED_SOURCE')
    return idx


def resolve(ref, main_repo, math_repo, idx):
    """Resolve a refers_to string to a repository path and its disposition, if it names one."""
    s = ref.strip()
    m = re.search(r'(Math-|main)[@:/ ]*(?:[0-9a-f]{7,40})?[:/ ]*((?:frontiers|coefficients|certificates|imports|reviews|companions|'
                  r'specifications|incoming|experiments|docs|formal|audits|cbz)/[^\s,;)\]`\'"#]+)', s)
    path = m.group(2) if m else None
    repo = m.group(1) if m else None
    if not path:
        m = re.search(r'((?:frontiers|coefficients|certificates|imports|reviews|companions|specifications|incoming|experiments|'
                      r'docs|formal|audits)/[^\s,;)\]`\'"#]+)', s)
        path = m.group(1) if m else None
    rec = re.findall(r'\b(Q-\d{4,5}|QP-C54-(?:T\d-)?\d{3}|OBL-MAIN-\d{2})\b', s)
    out = {'ref': ref, 'record_ids': rec}
    if not path:
        return out
    path = path.rstrip('.').rstrip('/')
    found = []
    for name, clone, commit in (('main', main_repo, REF), ('Math-', math_repo, MATH_REF)):
        if repo and repo != name:
            continue
        if git_exists(clone, commit, path):
            found.append(name)
    out['path'] = path
    out['exists_in'] = found
    disp = []
    for name in found:
        parts = path.split('/')
        for i in range(len(parts), 0, -1):
            disp += idx.get((name, '/'.join(parts[:i])), [])
        disp += [d for (r, p), ds in idx.items() if r == name and p.startswith(path + '/') for d in ds]
    out['dispositions'] = sorted(set(disp))
    return out


def main(main_repo, math_repo, facts_dir, objects_path, cand_path, closed_path, verdicts_path, output):
    facts = {}
    full = {(o['repository'], o['object']): o for o in json.load(open(objects_path))['objects']}
    batch_order = {}
    for f in sorted(glob.glob(os.path.join(facts_dir, 'fbatch*.json'))):
        b = json.load(open(f))
        batch_order[b['batch']] = []
        for o in b['objects']:
            key = f"{o['repository']}:{o['object']}"
            facts[key] = dict(o, files=full[(o['repository'], o['object'])]['files'])
            batch_order[b['batch']].append(key)
    all_objects = []
    cand = json.load(open(cand_path))
    closed_main = {e['path'] for e in json.load(open(closed_path))['entries'] if e['repository'].endswith('main')}
    result = json.load(open(verdicts_path))
    problems, seen = [], {}
    for b in result['batches']:
        # Verdicts name objects without their repository; resolve them within their batch, in batch-file order, so that
        # same-named objects of different repositories map to the batch file's order of appearance.
        queues = {}
        for key in batch_order.get(b['batch'], []):
            queues.setdefault(key.split(':', 1)[1], []).append(key)
        for v in (b['verdicts'] or []):
            name = v['object'].split(':', 1)[1] if v['object'] in facts else v['object']
            if v['object'] in facts and v['object'] in queues.get(name, []):
                key = v['object']; queues[name].remove(key)
            elif queues.get(name):
                key = queues[name].pop(0)
            else:
                problems.append(f'batch {b["batch"]}: unknown object {v["object"]}')
                continue
            if key in seen:
                problems.append(f'duplicate object {key}')
            seen[key] = v
    problems += [f'missing: {o}' for o in sorted(set(facts) - set(seen))]
    problems += [f'unknown: {o}' for o in sorted(set(seen) - set(facts))]
    idx = disposition_index(cand, closed_main)
    records, others, n = [], [], 0
    for obj in sorted(facts):
        if obj not in seen:
            continue
        v, fact = seen[obj], facts[obj]
        clone, oref, REPO = fact['clone'], fact['ref'], 'd6g8k5htny-coder/' + fact['repository']
        files = {f['path']: f for f in fact['files']}
        cover = {p: [] for p in files}
        ids = [it['item_id'] for it in v['items']]
        if len(ids) != len(set(ids)):
            problems.append(f'{obj}: duplicate item ids')
        for it in v['items']:
            if it['object'] not in (obj, obj.split(':', 1)[1]):
                problems.append(f'{obj}: item {it["item_id"]} names object {it["object"]}')
            mine = set()
            for spec in it['files']:
                hits = [p for p in files if p == spec or (spec.endswith('/') and p.startswith(spec))]
                if not hits:
                    problems.append(f'{obj}:{it["item_id"]}: file spec {spec} matches no object file')
                mine.update(hits)
            for p in mine:
                cover[p].append(it['item_id'])
            it['_files'] = sorted(mine)
        for p, c in cover.items():
            if len(c) != 1:
                problems.append(f'{obj}: {p} covered by {len(c)} items {c}')
        for it in v['items']:
            cls = it['classification']
            for st in it['statements']:
                if st['file'] not in it['_files']:
                    problems.append(f'{obj}:{it["item_id"]}: statement file {st["file"]} not in the item')
                    continue
                data = git_bytes(clone, oref, st['file'])
                if data is None:
                    problems.append(f'{obj}:{it["item_id"]}: cannot read {st["file"]}')
                    continue
                lines = data.decode('utf-8').split('\n')
                a, b = st['line_start'], st['line_end']
                if not (1 <= a <= b <= len(lines)) or '\n'.join(lines[a - 1:b]) != st['text'] or not st['text'].strip():
                    problems.append(f'{obj}:{it["item_id"]}: statement {st["file"]}:{a}-{b} is not verbatim')
            if cls == 'OBLIGATION':
                if not it['statements']:
                    problems.append(f'{obj}:{it["item_id"]}: OBLIGATION without statements')
                if it['floating_point'].split()[0].rstrip(':;,.') not in ('USED', 'NONE', 'NO_SCRIPTS'):
                    problems.append(f'{obj}:{it["item_id"]}: floating_point unclassified')
            if cls in ('REVIEW_OF', 'COPY_OR_EXPOSITION_OF') and not it['refers_to']:
                problems.append(f'{obj}:{it["item_id"]}: {cls} without refers_to')
            binding = {'repository': REPO, 'commit': oref, 'files': []}
            for p in it['_files']:
                data = git_bytes(clone, oref, p)
                blob = subprocess.run(['git', '-C', clone, 'rev-parse', f'{oref}:{p}'], capture_output=True, text=True).stdout.strip()
                if blob != files[p]['git_blob']:
                    problems.append(f'{obj}: {p} blob differs from facts')
                binding['files'].append({'path': p, 'git_blob': blob, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            resolved = [resolve(r, main_repo, math_repo, idx) for r in it['refers_to']]
            verification = {'ok': v['ok'], 'problems_fixed': v['problems']}
            if cls != 'OBLIGATION':
                others.append({'object': obj, 'repository': REPO, 'item_id': it['item_id'], 'classification': cls, 'title': it['title'],
                               'reason': it['classification_reason'], 'refers_to': resolved, 'source_binding': binding,
                               'verification': verification})
                continue
            n += 1
            oname = obj.split(':', 1)[1]
            loc = [f'{REPO}@{oref}:{oname}'] if set(it['_files']) == set(files) and oname != '.' else [f'{REPO}@{oref}:{p}' for p in it['_files']]
            records.append({
                'id': f'QP-C54-T6-{n:03d}', 'title': it['title'], 'status': 'QUARANTINED', 'proposal_status': 'PROPOSED',
                'proposal_kind': 'NEW_RECORD', 'location': loc, 'layer_a_status': it['layer_a_status'],
                'exact_boundary': it['exact_boundary'], 'blocking_reason': it['blocking_reason'], 'attempted': it['attempted'],
                'resumption_cost': it['resumption_cost'],
                'isolation': ('The item contains no Lean source at this ref, so nothing in the CLOSED domain can import it. This record '
                              'excludes mathematical promotion from this region; the composer must adopt it before it becomes a '
                              'terminal quarantine.') if not any(p.endswith('.lean') for p in it['_files']) else
                             ('The item contains Lean source that hosts no CLOSED declaration; it must not be imported into a CLOSED '
                              'root. The composer must adopt this record before it becomes a terminal quarantine.'),
                'owner': 'none', 'maintenance': 'zero', 'object': obj, 'item_id': it['item_id'],
                'classification_reason': it['classification_reason'], 'source_statements': it['statements'],
                'source_hypotheses': it['hypotheses'], 'source_dependencies': it['dependencies'],
                'source_self_limits': it['source_self_limits'], 'finite_checks': it['finite_checks'],
                'floating_point': it['floating_point'], 'refers_to': resolved,
                'census_artifacts': list(fact['census_artifacts']), 'source_binding': binding,
                'verification': verification})
    pre = [{'object': o['object'], 'resolution': o['pre'], 'lean_closed': o['lean_closed'], 'lean_files': o['lean_files'],
            'files': [{k: f[k] for k in ('path', 'git_blob', 'copy_of', 'exact_location')} for f in o['files']]}
           for o in all_objects if o['pre'] != 'TO_CLASSIFY']
    # Byte-identical obligations in different repositories (custody replicas) are one obligation: keep the first record and
    # carry the others as identical copies, then renumber.
    merged, seen_blobs = [], {}
    for r in records:
        key = tuple(sorted(f['git_blob'] for f in r['source_binding']['files']))
        if key in seen_blobs:
            keep = seen_blobs[key]
            keep.setdefault('identical_copies', []).append({'location': r['location'], 'source_binding': r['source_binding'],
                                                             'title': r['title'], 'verification': r['verification']})
            keep['location'] = keep['location'] + r['location']
            continue
        seen_blobs[key] = r
        merged.append(r)
    for i, r in enumerate(merged, 1):
        r['id'] = f'QP-C54-T6-{i:03d}'
    records = merged
    if problems:
        print('\n'.join(problems))
        print(f'{len(problems)} problem(s); nothing written')
        return 1
    from collections import Counter
    with open(output, 'x') as stream:
        json.dump({'schema': 'claude54.quarantine-proposals.t6.v1', 'refs': {k.split(':', 1)[0]: f['ref'] for k, f in facts.items()},
                   'records': records, 'other_items': others, 'mechanically_resolved_objects': pre}, stream, indent=1, ensure_ascii=False)
        stream.write('\n')
    print(f'{len(records)} obligation records; other items {dict(Counter(o["classification"] for o in others))}; '
          f'{len(pre)} objects resolved mechanically')
    fp = Counter(r['floating_point'].split()[0].rstrip(':;,.') for r in records)
    print('floating point', dict(fp))
    undisposed = [(o['object'], o['item_id'], r['ref']) for o in others for r in o['refers_to']
                  if r.get('exists_in') and not r.get('dispositions')]
    unresolved = [(o['object'], o['item_id'], r['ref']) for o in others for r in o['refers_to']
                  if not r.get('exists_in') and not r.get('record_ids')]
    print(f'referents that exist but have no disposition: {len(undisposed)}; referents not resolved to a path or record: {len(unresolved)}')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 9:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
