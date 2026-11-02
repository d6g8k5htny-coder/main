#!/usr/bin/env python3
"""Check tranche-8 items (Math- residual directories) against Math- bytes and assemble catalog-schema rows.

Usage: python3 -I -B assemble_t8.py MATH_REPO FACTS_DIR CANDIDATE_CATALOG VERDICTS_JSON OUTPUT_JSON

FACTS_DIR holds mbatch<N>.json (the objects given to the workflow). CANDIDATE_CATALOG is the catalog the objects were censused
against (adopted rows in "quarantines", proposals in "proposed_*"). VERDICTS_JSON is the workflow result {"batches": [{"batch",
"kind", "verdicts": [{"object", "ok", "problems", "items"}]}]}. Bindings come only from Git at REF.

Fails (exit 1, nothing written) on: a missing, duplicate or unknown object; items that do not partition an object's files (a file
covered by no item or by two; a file spec matching nothing); duplicate item ids within an object; an OBLIGATION without
statements, with a statement whose file is not in the object or whose text differs from that file's lines at REF, or with an
unclassified floating-point field; a REVIEW_OF or COPY_OR_EXPOSITION_OF item without refers_to; a claim_held_by id that is no
adopted or proposed record. A statement may quote a file that another item of the same object holds; that file is then bound
under source_binding.context_files and reported.

Kinds: an OBLIGATION whose verified claim_held_by names only adopted records becomes a PINNING_AMENDMENT to them (it binds its
bytes into those records). One held by a QP-C54 proposal stays a NEW_RECORD and carries claim_overlaps naming that proposal, for
the composer to merge. Any other OBLIGATION is a NEW_RECORD.
"""
import glob, hashlib, json, os, subprocess, sys

REF = '3b11bff6479f3f40ac838fe1d8b87c7ed6814606'
REPO = 'd6g8k5htny-coder/Math-'


def git_bytes(repo, path):
    r = subprocess.run(['git', '-C', repo, 'cat-file', 'blob', f'{REF}:{path}'], capture_output=True)
    return None if r.returncode else r.stdout


def main(math_repo, facts_dir, cand_path, verdicts_path, output):
    facts = {}
    for f in sorted(glob.glob(os.path.join(facts_dir, 'mbatch*.json'))):
        for o in json.load(open(f))['objects']:
            facts[o['object']] = o
    cand = json.load(open(cand_path))
    adopted = {r['id'] for r in cand['quarantines']}
    proposed = {r['id'] for k in ('proposed_quarantines', 'proposed_quarantines_federation', 'proposed_pinning_amendments')
                for r in cand.get(k, [])}
    verdicts = [v for b in json.load(open(verdicts_path))['batches'] for v in (b['verdicts'] or [])]
    problems, seen = [], {}
    for v in verdicts:
        if v['object'] in seen:
            problems.append(f'duplicate object {v["object"]}')
        seen[v['object']] = v
    problems += [f'missing: {o}' for o in sorted(set(facts) - set(seen))]
    problems += [f'unknown: {o}' for o in sorted(set(seen) - set(facts))]
    records, others, cross_item, n = [], [], [], 0

    def bind(p):
        data = git_bytes(math_repo, p)
        blob = subprocess.run(['git', '-C', math_repo, 'rev-parse', f'{REF}:{p}'], capture_output=True, text=True).stdout.strip()
        return {'path': p, 'git_blob': blob, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    for obj in sorted(facts):
        if obj not in seen:
            continue
        v, fact = seen[obj], facts[obj]
        files = {f['path']: f for f in fact['files']}
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
                    if st['file'] in files:
                        if st['file'] not in it['_context']:
                            it['_context'].append(st['file'])
                            cross_item.append(f'{obj}:{it["item_id"]} quotes {st["file"]} (held by another item of the object)')
                    else:
                        problems.append(f'{obj}:{it["item_id"]}: statement file {st["file"]} not in the object')
                        continue
                data = git_bytes(math_repo, st['file'])
                lines = data.decode('utf-8').split('\n') if data is not None else []
                a, b = st['line_start'], st['line_end']
                if not (1 <= a <= b <= len(lines)) or '\n'.join(lines[a - 1:b]) != st['text'] or not st['text'].strip():
                    problems.append(f'{obj}:{it["item_id"]}: statement {st["file"]}:{a}-{b} is not verbatim')
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
            binding = {'repository': REPO, 'commit': REF, 'files': [bind(p) for p in it['_files']]}
            for p in files:
                if p in it['_files'] and binding['files'][it['_files'].index(p)]['git_blob'] != files[p]['git_blob']:
                    problems.append(f'{obj}: {p} blob differs from facts')
            if it['_context']:
                binding['context_files'] = [bind(p) for p in it['_context']]
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
            loc = [f'{REPO}@{REF}:{obj}'] if set(it['_files']) == set(files) else [f'{REPO}@{REF}:{p}' for p in it['_files']]
            rec = {
                'id': f'QP-C54-T8-{n:03d}', 'title': it['title'], 'status': 'QUARANTINED', 'proposal_status': 'PROPOSED',
                'proposal_kind': kind, 'target_records': held_adopted if kind == 'PINNING_AMENDMENT' else [], 'location': loc,
                'layer_a_status': it['layer_a_status'], 'exact_boundary': it['exact_boundary'], 'blocking_reason': it['blocking_reason'],
                'attempted': it['attempted'], 'resumption_cost': it['resumption_cost'],
                'isolation': ('The item contains no Lean source at this ref, so nothing in the CLOSED domain can import it. This record '
                              'excludes mathematical promotion from this region; the composer must adopt it before it becomes a '
                              'terminal quarantine.') if not any(p.endswith('.lean') for p in it['_files']) else
                             ('The item contains Lean source that hosts no CLOSED declaration; it must not be imported into a CLOSED '
                              'root. The composer must adopt this record before it becomes a terminal quarantine.'),
                'owner': 'none', 'maintenance': 'zero', 'object': obj, 'item_id': it['item_id'],
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
    if problems:
        print('\n'.join(problems[:80]))
        print(f'{len(problems)} problem(s); nothing written')
        return 1
    from collections import Counter
    with open(output, 'x') as stream:
        json.dump({'schema': 'claude54.quarantine-proposals.t8.v1', 'math_ref': REF, 'records': records, 'other_items': others},
                  stream, indent=1, ensure_ascii=False)
        stream.write('\n')
    for c in cross_item:
        print('cross-item quote:', c)
    print(f'{len(records)} obligation records ({dict(Counter(r["proposal_kind"] for r in records))}); other items '
          f'{dict(Counter(o["classification"] for o in others))}')
    print('floating point', dict(Counter(r['floating_point'].split()[0].rstrip(':;,.') for r in records)))
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
