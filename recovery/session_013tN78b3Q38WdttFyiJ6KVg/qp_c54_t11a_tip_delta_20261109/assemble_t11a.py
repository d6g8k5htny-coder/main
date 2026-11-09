#!/usr/bin/env python3
"""Check tranche-11a items (mathematical-type files that landed on a default-branch tip after the QP-C54 cuts and that no record
binds) against the bytes at each object's tip commit, and assemble catalog-schema rows.

Usage: python3 -I -B assemble_t11a.py CLONES_JSON FACTS_DIR CANDIDATE_CATALOG OUTPUT_JSON VERDICTS_JSON [VERDICTS_JSON...]

FACTS_DIR holds dbatch<N>.json. Each object names its repository (owner/name), the tip commit, the old cut, and its files with
full Git blobs. CLONES_JSON maps repository short names (e.g. "Math-", "main") to local clones. The checks are those of
assemble_t10.py: missing, duplicate or unknown objects; blob mismatch; partition; verbatim statements; floating point; REVIEW and
COPY targets; holders. Locations are file-level, <repository>@<tip>:<path>.
"""
import glob, hashlib, json, os, subprocess, sys
from collections import Counter



def git(repo, *a):
    return subprocess.run(['git', '-C', repo, *a], capture_output=True)


def refbind(fact):
    return {'repository': fact['repository'], 'commit': fact['commit'], 'ref': 'refs/heads/main (default-branch tip)',
            'old_cut': fact['old_cut']}


def where(fact):
    return f'the default-branch tip {fact["commit"]} of {fact["repository"]}'


def main(clones_path, facts_dir, cand_path, output, *verdict_paths):
    clones = json.load(open(clones_path))
    facts = {}
    for f in sorted(glob.glob(os.path.join(facts_dir, 'dbatch*.json'))):
        for o in json.load(open(f))['objects']:
            if o['object'] in facts:
                raise SystemExit(f'duplicate object in facts: {o["object"]}')
            facts[o['object']] = o
    cand = json.load(open(cand_path))
    adopted = {r['id'] for r in cand['quarantines']}
    proposed = {r['id'] for k in ('proposed_quarantines', 'proposed_quarantines_federation', 'proposed_pinning_amendments')
                for r in cand.get(k, [])}
    verdicts = [v for p in verdict_paths for b in json.load(open(p))['batches'] for v in (b['verdicts'] or [])]
    problems, seen = [], {}
    for v in verdicts:
        if v['object'] in seen:
            problems.append(f'duplicate object {v["object"]}')
        seen[v['object']] = v
    problems += [f'missing: {o}' for o in sorted(set(facts) - set(seen))]
    problems += [f'unknown: {o}' for o in sorted(set(seen) - set(facts))]
    def clone_of(fact):
        return clones[fact['repository'].split('/', 1)[1]]
    for o in facts.values():
        if git(clone_of(o), 'cat-file', '-e', f'{o["commit"]}^{{commit}}').returncode:
            problems.append(f'tip {o["commit"]} is not in {clone_of(o)}')
    records, others, mech, cross_item, n = [], [], [], [], 0
    cache = {}
    cur = {}

    def blob_at(commit, p):
        if (commit, p) not in cache:
            repo = cur['clone']
            r = git(repo, 'cat-file', 'blob', f'{commit}:{p}')
            b = git(repo, 'rev-parse', f'{commit}:{p}').stdout.decode().strip()
            cache[(commit, p)] = (None, None) if r.returncode else (r.stdout, b)
        return cache[(commit, p)]

    def bind(commit, p):
        data, blob = blob_at(commit, p)
        return {'path': p, 'git_blob': blob, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    for obj in sorted(facts):
        if obj not in seen:
            continue
        v, fact = seen[obj], facts[obj]
        commit = fact['commit']
        cur['clone'] = clone_of(fact)
        files = {f['path']: f for f in fact['files']}
        copies = {f['path']: f for f in fact.get('copies', [])}
        for p, f in list(files.items()) + list(copies.items()):
            data, blob = blob_at(commit, p)
            if data is None or not blob.startswith(f['git_blob']):
                problems.append(f'{obj}: {p} at {commit[:8]} does not match the facts blob {f["git_blob"]}')
        cover = {p: [] for p in files}
        ids = [it['item_id'] for it in v['items']]
        if len(ids) != len(set(ids)):
            problems.append(f'{obj}: duplicate item ids')
        for it in v['items']:
            if it['object'] != obj:
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
            it['_context'] = []
            for st in it['statements']:
                if st['file'] not in it['_files']:
                    if st['file'] in files or st['file'] in copies:
                        if st['file'] not in it['_context']:
                            it['_context'].append(st['file'])
                            cross_item.append(f'{obj}:{it["item_id"]} quotes {st["file"]} (another item\'s file or a resolved main copy)')
                    else:
                        problems.append(f'{obj}:{it["item_id"]}: statement file {st["file"]} not in the object')
                        continue
                data, _ = blob_at(commit, st['file'])
                lines = data.decode('utf-8').split('\n') if data is not None else []
                a, b = st['line_start'], st['line_end']
                if not (1 <= a <= b <= len(lines)) or '\n'.join(lines[a - 1:b]) != st['text'] or not st['text'].strip():
                    problems.append(f'{obj}:{it["item_id"]}: statement {st["file"]}:{a}-{b} is not verbatim at {commit[:8]}')
            held = list(it.get('claim_held_by') or [])
            for h in held:
                if h not in adopted and h not in proposed:
                    problems.append(f'{obj}:{it["item_id"]}: claim_held_by {h} is no adopted or proposed record')
            if cls == 'OBLIGATION':
                if not it['statements']:
                    problems.append(f'{obj}:{it["item_id"]}: OBLIGATION without statements')
                if it['floating_point'].split()[0].rstrip(':;,.') not in ('USED', 'NONE', 'NO_SCRIPTS'):
                    problems.append(f'{obj}:{it["item_id"]}: floating_point unclassified')
            elif held:
                problems.append(f'{obj}:{it["item_id"]}: claim_held_by on a {cls} item')
            if cls in ('REVIEW_OF', 'COPY_OR_EXPOSITION_OF') and not it['refers_to']:
                problems.append(f'{obj}:{it["item_id"]}: {cls} without refers_to')
            if any(blob_at(commit, p)[0] is None for p in it['_files'] + it['_context']):
                continue
            binding = dict(refbind(fact), files=[bind(commit, p) for p in it['_files']])
            if it['_context']:
                binding['context_files'] = [bind(commit, p) for p in it['_context']]
            verification = {'ok': v['ok'], 'problems_fixed': v['problems']}
            if cls != 'OBLIGATION':
                others.append({'object': obj, 'item_id': it['item_id'], 'classification': cls, 'title': it['title'],
                               'reason': it['classification_reason'], 'refers_to': it['refers_to'], 'source_binding': binding,
                               'verification': verification})
                continue
            n += 1
            held_adopted = [h for h in held if h in adopted]
            held_prop = [h for h in held if h not in adopted]
            kind = 'PINNING_AMENDMENT' if held_adopted and not held_prop else 'NEW_RECORD'
            lean = any(p.endswith('.lean') for p in it['_files'])
            rec = {
                'id': f'QP-C54-T11-{n:03d}', 'title': it['title'], 'status': 'QUARANTINED', 'proposal_status': 'PROPOSED',
                'proposal_kind': kind, 'target_records': held_adopted if kind == 'PINNING_AMENDMENT' else [],
                'location': [f'{fact["repository"]}@{commit}:{p}' for p in it['_files']],
                'layer_a_status': it['layer_a_status'], 'exact_boundary': it['exact_boundary'], 'blocking_reason': it['blocking_reason'],
                'attempted': it['attempted'], 'resumption_cost': it['resumption_cost'],
                'isolation': (f'These bytes landed at {where(fact)} after the accounting cut {fact["old_cut"]}; no CLOSED '
                              'declaration names them. '
                              + ('The item contains Lean source that hosts no CLOSED declaration; it must not be imported into a '
                                 'CLOSED root. ' if lean else 'The item contains no Lean source. ')
                              + 'The default branch retains these bytes. ' 
                              'The composer must adopt this record before it becomes a terminal quarantine.'),
                'archive_ref': None, 'owner': 'none', 'maintenance': 'zero', 'object': obj, 'item_id': it['item_id'],
                'classification_reason': it['classification_reason'], 'claim_held_by': held, 'source_statements': it['statements'],
                'source_hypotheses': it['hypotheses'], 'source_dependencies': it['dependencies'],
                'source_self_limits': it['source_self_limits'], 'finite_checks': it['finite_checks'],
                'floating_point': it['floating_point'], 'refers_to': it['refers_to'], 'source_binding': binding,
                'verification': verification}
            if held_prop:
                rec['claim_overlaps'] = [{'record': h, 'relation': 'SAME_CLAIM_AS_PROPOSAL' if h in proposed else 'HOLDS_CLAIM',
                                          'note': 'The verifiers found this record already holds the claim; the composer merges the two.'}
                                         for h in held]
            records.append(rec)
        if copies:
            mech.append({'object': obj, 'reason': 'byte copies of files on the Math- main tip; each copy\'s disposition is that of '
                         'the main file named in copy_of', 'source_binding': {
                             **refbind(fact), 'files': [dict(bind(commit, p), copy_of=copies[p]['copy_of']) for p in sorted(copies)]}})
    if problems:
        print('\n'.join(problems[:80]))
        print(f'{len(problems)} problem(s); nothing written')
        return 1
    with open(output, 'x') as stream:
        json.dump({'schema': 'claude54.quarantine-proposals.t11a.v1',
                   'tips': {o['repository']: o['commit'] for o in facts.values()},
                   'records': records, 'other_items': others, 'mechanical_copies': mech}, stream, indent=1, ensure_ascii=False)
        stream.write('\n')
    for c in cross_item:
        print('cross-item quote:', c)
    print(f'{len(records)} obligation records ({dict(Counter(r["proposal_kind"] for r in records))}); other items '
          f'{dict(Counter(o["classification"] for o in others))}')
    print('floating point', dict(Counter(r['floating_point'].split()[0].rstrip(':;,.') for r in records)))
    print('mechanical copies', sum(len(m['source_binding']['files']) for m in mech), 'in', len(mech), 'objects')
    print('records by ref', dict(Counter(r['source_binding']['ref'] for r in records)))
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 6:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
