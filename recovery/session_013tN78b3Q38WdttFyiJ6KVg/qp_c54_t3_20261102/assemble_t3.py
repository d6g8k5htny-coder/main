#!/usr/bin/env python3
"""Check tranche-3 review classifications and A3 audits against Math- bytes and assemble the output.

Usage: python3 -I -B assemble_t3.py MATH_REPO FACTS_DIR CANDIDATE_CATALOG CLOSED_JSON PINS_JSON RESULT_JSON OUTPUT_JSON

PINS_JSON is the pinning sweep (pins.json); its Math- paths count as disposed through their target records once adopted.

FACTS_DIR holds rb1.json..rb3.json (the 102 Math- reviews/ directories that no record names). RESULT_JSON is the workflow result
{"reviews": [{"batch", "verdicts": [{"object", "ok", "problems", "record"}]}], "a3": {"audits": [...], "cross": {"reviews": [...]}}}.

Fails (exit 1, nothing written) on: a missing, duplicate or unknown directory; a REVIEW_OF with no reviewed_objects; a
STANDALONE_CLAIM with no statements, with a statement whose file is not in the directory or whose text differs from that file's
lines at REF, or with an unclassified floating_point. Bindings come only from Git. Each reviewed object is resolved against Math- at
REF and the candidate catalog's dispositions; reviewed objects that exist but have no disposition are reported, not failed.
"""
import glob, hashlib, json, os, re, subprocess, sys

REF = '19a55750451b8016b50d1d20be6a18c021807ecb'
REPO = 'd6g8k5htny-coder/Math-'
ROOTS = r'(?:frontiers|coefficients|certificates|imports|reviews|companions|specifications|incoming|formal|docs)'


def git_bytes(repo, path):
    r = subprocess.run(['git', '-C', repo, 'cat-file', 'blob', f'{REF}:{path}'], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def exists(repo, path):
    return subprocess.run(['git', '-C', repo, 'cat-file', '-e', f'{REF}:{path}'], capture_output=True).returncode == 0


def index(cand):
    idx = {}
    for key, rows in (('ADOPTED', cand['quarantines']), ('PROPOSED_QP_C54', cand['proposed_quarantines'] + cand['proposed_pinning_amendments'])):
        for r in rows:
            for loc in (r['location'] if isinstance(r['location'], list) else [r['location']]):
                if isinstance(loc, str):
                    m = re.match(r'd6g8k5htny-coder/Math-@[0-9a-f]{40}:([^#]+)', loc) or \
                        re.match(r'https://github.com/d6g8k5htny-coder/Math-/blob/[0-9a-f]{40}/([^#]+)', loc)
                    if m:
                        idx.setdefault(m.group(1).rstrip('/'), set()).add(f'{key}:{r["id"]}')
    for n in cand['proposed_non_obligations']:
        idx.setdefault(n['object'].rstrip('/'), set()).add(f'NOT_OBLIGATION_QP_C54:{n["object"]}')
    for r in cand['non_obligation_support']:
        text = json.dumps(r, ensure_ascii=False)
        if r['id'].startswith('lean:math-formal-'):
            idx.setdefault('formal', set()).add(f'NON_OBLIGATION_SUPPORT:{r["id"]}')
        for m in re.findall(rf'({ROOTS}/[A-Za-z0-9_./\-]+)', text):
            idx.setdefault(m.rstrip('.,/'), set()).add(f'NON_OBLIGATION_SUPPORT:{r["id"]}')
    for p in CTX.get('pins', []):
        if p['repository'].endswith('Math-'):
            idx.setdefault(p['path'], set()).add(f'PINNING_SWEEP:{p["id"]}->{p["record"]}')
    return idx


CTX = {}


def resolve(ref, math_repo, idx):
    out = {'ref': ref, 'record_ids': re.findall(r'\b(Q-\d{4,5}|QP-C54-(?:T\d-)?\d{3}|K-[A-Z0-9-]+)\b', ref)}
    m = re.search(rf'({ROOTS}/[^\s,;)\]`\'"#]+)', ref)
    if not m:
        # Claim ids (e.g. OA-...-v1) and Lean declaration names: look them up in record text and in the CLOSED index.
        tok = ref.strip().strip('`')
        if re.fullmatch(r'[A-Za-z0-9_.\-]{8,}', tok):
            hits = sorted(rid for rid, text in CTX['texts'].items() if tok in text)[:6]
            decl = sorted(e['id'] for e in CTX['closed'] if e['declaration'] == tok or e['declaration'].endswith('.' + tok))[:6]
            if hits:
                out['text_mentions'] = hits
            if decl:
                out['closed_declarations'] = decl
        return out
    path = m.group(1).rstrip('.').rstrip('/')
    out['path'], out['exists'] = path, exists(math_repo, path)
    disp = set()
    if out['exists']:
        parts = path.split('/')
        for i in range(len(parts), 0, -1):
            disp |= idx.get('/'.join(parts[:i]), set())
        for p, d in idx.items():
            if p.startswith(path + '/'):
                disp |= d
        # Packet level: a disposed location in the same packet directory (two path components below the root).
        packet = '/'.join(parts[:2])
        if not disp:
            for p, d in idx.items():
                if p == packet or p.startswith(packet + '/'):
                    disp |= {x + ' (same packet)' for x in d}
    closed_under = sorted({e['id'] for e in CTX['closed'] if e['repository'].endswith('Math-') and
                           (e['path'] == path or e['path'].startswith(path + '/') or path.startswith(e['path'].rsplit('/', 1)[0] + '/'))})
    if closed_under and not disp:
        disp.add(f'CLOSED_PACKAGE:{len(closed_under)} declarations, e.g. {closed_under[0]}')
    if not disp:
        named = sorted(rid for rid, text in CTX['texts'].items() if path in text)[:6]
        if named:
            out['named_in_record_text_only'] = named
    out['dispositions'] = sorted(disp)
    return out


def main(math_repo, facts_dir, cand_path, closed_path, pins_path, result_path, output):
    facts = {}
    for f in sorted(glob.glob(os.path.join(facts_dir, 'rb[0-9].json'))):
        for o in json.load(open(f)):
            facts[o['object']] = o
    cand = json.load(open(cand_path))
    CTX['pins'] = json.load(open(pins_path))['pins']
    idx = index(cand)
    CTX['closed'] = json.load(open(closed_path))['entries']
    CTX['texts'] = {r['id']: json.dumps(r, ensure_ascii=False) for r in cand['quarantines'] + cand['proposed_quarantines']}
    result = json.load(open(result_path))
    problems, seen = [], {}
    for b in result['reviews']:
        for v in (b['verdicts'] or []):
            o = v['record']['object']
            if o in seen:
                problems.append(f'duplicate: {o}')
            seen[o] = v
    problems += [f'missing: {o}' for o in sorted(set(facts) - set(seen))]
    problems += [f'unknown: {o}' for o in sorted(set(seen) - set(facts))]
    records, reviews, not_obligations, n = [], [], [], 0
    for obj in sorted(facts):
        if obj not in seen:
            continue
        v, fact = seen[obj], facts[obj]
        rec = v['record']
        files = {f['path']: f for f in fact['files']}
        cls = rec['classification']
        for st in rec['statements']:
            if st['file'] not in files:
                problems.append(f'{obj}: statement file {st["file"]} not in the directory'); continue
            data = git_bytes(math_repo, st['file'])
            lines = data.decode('utf-8').split('\n')
            a, e = st['line_start'], st['line_end']
            if not (1 <= a <= e <= len(lines)) or '\n'.join(lines[a - 1:e]) != st['text'] or not st['text'].strip():
                problems.append(f'{obj}: statement {st["file"]}:{a}-{e} is not verbatim')
        binding = {'repository': REPO, 'commit': REF, 'files': []}
        for p, f in sorted(files.items()):
            data = git_bytes(math_repo, p)
            if data is None or hashlib.sha256(data).hexdigest() != f['sha256']:
                problems.append(f'{obj}: {p} bytes differ from facts'); continue
            binding['files'].append({'path': p, 'git_blob': f['git_blob'], 'bytes': len(data), 'sha256': f['sha256']})
        verification = {'ok': v['ok'], 'problems_fixed': v['problems']}
        if cls == 'REVIEW_OF':
            if not rec['reviewed_objects']:
                problems.append(f'{obj}: REVIEW_OF without reviewed_objects')
            reviews.append({'object': obj, 'title': rec['title'], 'reason': rec['classification_reason'],
                            'reviewed_objects': [resolve(r, math_repo, idx) for r in rec['reviewed_objects']],
                            'source_binding': binding, 'verification': verification})
        elif cls == 'NOT_OBLIGATION':
            not_obligations.append({'object': obj, 'title': rec['title'], 'reason': rec['classification_reason'],
                                    'source_binding': binding, 'verification': verification})
        else:
            if not rec['statements']:
                problems.append(f'{obj}: STANDALONE_CLAIM without statements')
            if rec['floating_point'].split()[0].rstrip(':;,.') not in ('USED', 'NONE', 'NO_SCRIPTS'):
                problems.append(f'{obj}: floating_point unclassified')
            n += 1
            records.append({
                'id': f'QP-C54-T3-{n:03d}', 'title': rec['title'], 'status': 'QUARANTINED', 'proposal_status': 'PROPOSED',
                'proposal_kind': 'NEW_RECORD', 'location': [f'{REPO}@{REF}:{obj}'], 'layer_a_status': rec['layer_a_status'],
                'exact_boundary': rec['exact_boundary'], 'blocking_reason': rec['blocking_reason'], 'attempted': rec['attempted'],
                'resumption_cost': rec['resumption_cost'],
                'isolation': ('The directory contains no Lean source at this ref, so nothing in the CLOSED domain can import it. This '
                              'record excludes mathematical promotion from this region; the composer must adopt it before it '
                              'becomes a terminal quarantine.'),
                'owner': 'none', 'maintenance': 'zero', 'classification_reason': rec['classification_reason'],
                'reviewed_objects': [resolve(r, math_repo, idx) for r in rec['reviewed_objects']],
                'source_statements': rec['statements'], 'source_hypotheses': rec['hypotheses'],
                'source_dependencies': rec['dependencies'], 'source_self_limits': rec['source_self_limits'],
                'finite_checks': rec['finite_checks'], 'floating_point': rec['floating_point'], 'source_binding': binding,
                'verification': verification})
    t3cls = {r['object']: 'T3_REVIEW_OF' for r in reviews}
    t3cls.update({r['object']: 'T3_NOT_OBLIGATION' for r in not_obligations})
    t3cls.update({r['location'][0].split(':', 1)[1]: f'T3_PROPOSED:{r["id"]}' for r in records})
    for group in (reviews, records):
        for r in group:
            for x in r['reviewed_objects']:
                if x.get('exists') and not x.get('dispositions'):
                    d = '/'.join(x['path'].split('/')[:2])
                    if d in t3cls:
                        x['dispositions'] = [t3cls[d]]
    a3 = result.get('a3') or {}
    cross = {r['record']: r for r in ((a3.get('cross') or {}).get('reviews') or [])}
    advice = [{'record': x['record'], 'auditor_verdict': x['verdict'], 'cross_verdict': cross.get(x['record'], {}).get('final_verdict'),
               'cross_agrees': cross.get(x['record'], {}).get('agrees'), 'audit': x, 'cross': cross.get(x['record'])}
              for x in (a3.get('audits') or [])]
    if problems:
        print('\n'.join(problems)); print(f'{len(problems)} problem(s); nothing written')
        return 1
    with open(output, 'x') as s:
        json.dump({'schema': 'claude54.quarantine-proposals.t3.v1', 'math_ref': REF, 'records': records, 'reviews': reviews,
                   'not_obligations': not_obligations, 'a3_advice': advice}, s, indent=1, ensure_ascii=False)
        s.write('\n')
    und = [(r['object'], x['ref']) for r in reviews for x in r['reviewed_objects']
           if x.get('exists') and not x.get('dispositions') and not x.get('named_in_record_text_only')]
    named_only = [(r['object'], x['ref']) for r in reviews for x in r['reviewed_objects'] if x.get('named_in_record_text_only')]
    unres = [(r['object'], x['ref']) for r in reviews for x in r['reviewed_objects']
             if not x.get('path') and not x.get('record_ids') and not x.get('text_mentions') and not x.get('closed_declarations')]
    print(f'{len(records)} standalone records; {len(reviews)} reviews; {len(not_obligations)} not obligations')
    print(f'reviewed objects existing without disposition: {len(und)} {sorted(set(x for _, x in und))}; named only in record text: {len(named_only)}; '
          f'not resolved to a path or id: {len(unres)}')
    print('A3:', [(a['record'], a['auditor_verdict'], a['cross_verdict']) for a in advice])
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 8:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
