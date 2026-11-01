#!/usr/bin/env python3
"""Check tranche-2 proposals against Math- bytes and assemble catalog-schema records.

Usage: python3 -I -B assemble_t2.py MATH_REPO FACTS_JSON VERDICTS_JSON OUTPUT_JSON

VERDICTS_JSON is {"batches": [{"batch": n, "verdicts": [{"object", "ok", "problems", "record"}]}]}.
Fails (exit 1, nothing written) on: a missing, duplicate or unknown object; a statement whose
file is not part of its object or whose text differs from that file's lines at REF; an
OBLIGATION or PINNING_AMENDMENT with no statements; a floating_point field not starting with
USED, NONE or NO_SCRIPTS; a PINNING_AMENDMENT whose title does not start with an existing
mentioned_by record id. Bindings come only from FACTS_JSON. For every bound file it also
records whether the bytes are identical at the catalog cut CUT.
"""
import hashlib
import json
import subprocess
import sys

REF = '19a55750451b8016b50d1d20be6a18c021807ecb'
CUT = '8151f592918a845bf19eb8eafaf8444540a32734'
REPO = 'd6g8k5htny-coder/Math-'


def blob_at(repo, commit, path):
    r = subprocess.run(['git', '-C', repo, 'rev-parse', f'{commit}:{path}'], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def main(math_repo, facts_path, verdicts_path, output):
    facts = {f['object']: f for f in json.load(open(facts_path))}
    result = json.load(open(verdicts_path))
    verdicts = [v for b in result['batches'] for v in (b['verdicts'] or [])]
    problems, seen = [], {}
    for v in verdicts:
        o = v['record']['object']
        if o in seen:
            problems.append(f'duplicate: {o}')
        seen[o] = v
    problems += [f'missing: {o}' for o in sorted(set(facts) - set(seen))]
    problems += [f'unknown: {o}' for o in sorted(set(seen) - set(facts))]
    records, not_obligations, n = [], [], 0
    for obj in sorted(facts):
        if obj not in seen:
            continue
        v, fact = seen[obj], facts[obj]
        rec = v['record']
        files = {f['path']: f for f in fact['files']}
        cls = rec['classification']
        for st in rec['statements']:
            if st['file'] not in files:
                problems.append(f'{obj}: statement file {st["file"]} is not in the object')
                continue
            data = subprocess.run(['git', '-C', math_repo, 'show', f'{REF}:{st["file"]}'], capture_output=True, check=True).stdout
            if hashlib.sha256(data).hexdigest() != files[st['file']]['sha256']:
                problems.append(f'{obj}: {st["file"]} bytes differ from facts')
            lines = data.decode('utf-8').split('\n')
            a, b = st['line_start'], st['line_end']
            if not (1 <= a <= b <= len(lines)):
                problems.append(f'{obj}: invalid range {st["file"]}:{a}-{b}')
            elif '\n'.join(lines[a - 1:b]) != st['text'] or not st['text'].strip():
                problems.append(f'{obj}: statement {st["file"]}:{a}-{b} is not verbatim')
        if cls in ('OBLIGATION', 'PINNING_AMENDMENT') and not rec['statements']:
            problems.append(f'{obj}: {cls} without statements')
        if cls != 'NOT_OBLIGATION' and rec['floating_point'].split()[0].rstrip(':;,.') not in ('USED', 'NONE', 'NO_SCRIPTS'):
            problems.append(f'{obj}: floating_point unclassified')
        if cls == 'PINNING_AMENDMENT' and not any(rec['title'].startswith(rid) for rid in fact['mentioned_by']):
            problems.append(f'{obj}: pinning amendment title must start with one of {fact["mentioned_by"]}')
        binding = {'repository': REPO, 'commit': REF, 'files': [
            dict(f, identical_at_catalog_cut=blob_at(math_repo, CUT, f['path']) == f['git_blob']) for f in fact['files']],
            'files_truncated': fact['files_truncated']}
        if cls == 'NOT_OBLIGATION':
            not_obligations.append({'object': obj, 'reason': rec['classification_reason'], 'title': rec['title'],
                                    'source_binding': binding, 'verification': {'layerA_ok': v['ok'], 'problems_fixed': v['problems']}})
            continue
        n += 1
        records.append({
            'id': f'QP-C54-T2-{n:03d}', 'title': rec['title'], 'status': 'QUARANTINED', 'proposal_status': 'PROPOSED',
            'proposal_kind': 'PINNING_AMENDMENT' if cls == 'PINNING_AMENDMENT' else 'NEW_RECORD',
            'location': [f'{REPO}@{REF}:{obj}' if obj != '.' else f'{REPO}@{REF}:{fact["gap_files"][0]}'],
            'layer_a_status': rec['layer_a_status'], 'exact_boundary': rec['exact_boundary'],
            'blocking_reason': rec['blocking_reason'], 'attempted': rec['attempted'], 'resumption_cost': rec['resumption_cost'],
            'isolation': ('The object contains no Lean source at this ref, so nothing in the CLOSED domain can import it. This '
                          'record excludes mathematical promotion from this region; the composer must adopt it before it '
                          'becomes a terminal quarantine.'),
            'owner': 'none', 'maintenance': 'zero',
            'classification_reason': rec['classification_reason'], 'source_statements': rec['statements'],
            'source_hypotheses': rec['hypotheses'], 'source_dependencies': rec['dependencies'],
            'source_self_limits': rec['source_self_limits'], 'finite_checks': rec['finite_checks'],
            'floating_point': rec['floating_point'], 'mentioned_by': fact['mentioned_by'], 'source_binding': binding,
            'verification': {'layerA_ok': v['ok'], 'problems_fixed': v['problems']}})
    if problems:
        print('\n'.join(problems)); print(f'{len(problems)} problem(s); nothing written')
        return 1
    with open(output, 'x') as stream:
        json.dump({'schema': 'claude54.quarantine-proposals.t2.v1', 'math_ref': REF, 'catalog_cut': CUT,
                   'records': records, 'not_obligations': not_obligations}, stream, indent=1, ensure_ascii=False)
        stream.write('\n')
    print(f'{len(records)} records ({sum(r["proposal_kind"] == "PINNING_AMENDMENT" for r in records)} pinning amendments); '
          f'{len(not_obligations)} not obligations')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
