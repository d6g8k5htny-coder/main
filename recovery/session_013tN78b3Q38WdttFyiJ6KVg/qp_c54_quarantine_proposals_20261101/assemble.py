#!/usr/bin/env python3
"""Check drafted quarantine proposals against Math- bytes and assemble catalog-schema records.

Usage: python3 -I -B assemble.py MATH_REPO FACTS_JSON VERDICTS_JSON OUTPUT_JSON

VERDICTS_JSON is the workflow result ({"batches": [{"batch": n, "verdicts": [...]}]}).
Fails (exit 1, nothing written) if any packet is missing, duplicated or extra, if any
quoted statement differs from its PROOF.md lines, if a statement is empty or its line
range is invalid, or if floating_point does not start with USED, NONE or NO_SCRIPTS.
Source bindings come only from FACTS_JSON (precomputed from Git), never from drafts.
"""
import hashlib
import json
import subprocess
import sys

CUT = '8151f592918a845bf19eb8eafaf8444540a32734'
REPO = 'd6g8k5htny-coder/Math-'


def main(math_repo, facts_path, verdicts_path, output):
    facts = {f['packet']: f for f in json.load(open(facts_path))}
    result = json.load(open(verdicts_path))
    verdicts = [v for batch in result['batches'] for v in (batch['verdicts'] or [])]
    problems = []
    seen = {}
    for verdict in verdicts:
        packet = verdict['record']['packet']
        if packet in seen:
            problems.append(f'duplicate record: {packet}')
        seen[packet] = verdict
    for packet in sorted(set(facts) - set(seen)):
        problems.append(f'missing record: {packet}')
    for packet in sorted(set(seen) - set(facts)):
        problems.append(f'unknown packet: {packet}')
    records = []
    for number, packet in enumerate(sorted(facts), 1):
        if packet not in seen:
            continue
        verdict, fact = seen[packet], facts[packet]
        record = verdict['record']
        proof = subprocess.run(['git', '-C', math_repo, 'show', f'{CUT}:{fact["proof"]}'],
                               capture_output=True, check=True).stdout
        if hashlib.sha256(proof).hexdigest() != fact['proof_sha256']:
            problems.append(f'{packet}: PROOF.md bytes differ from facts')
        lines = proof.decode('utf-8').split('\n')
        if not record['statements']:
            problems.append(f'{packet}: no statements')
        for statement in record['statements']:
            start, end = statement['line_start'], statement['line_end']
            if not (1 <= start <= end <= len(lines)):
                problems.append(f'{packet}: invalid line range {start}-{end}')
                continue
            if '\n'.join(lines[start - 1:end]) != statement['text']:
                problems.append(f'{packet}: statement at {start}-{end} is not verbatim')
            if not statement['text'].strip():
                problems.append(f'{packet}: empty statement at {start}-{end}')
        if record['floating_point'].split()[0].rstrip(':;,.') not in ('USED', 'NONE', 'NO_SCRIPTS'):
            problems.append(f'{packet}: floating_point lacks USED/NONE/NO_SCRIPTS')
        records.append({
            'id': f'QP-C54-{number:03d}',
            'title': record['title'],
            'status': 'QUARANTINED',
            'proposal_status': 'PROPOSED',
            'location': [f'{REPO}@{CUT}:{packet}'],
            'layer_a_status': record['layer_a_status'],
            'exact_boundary': record['exact_boundary'],
            'blocking_reason': record['blocking_reason'],
            'attempted': record['attempted'],
            'resumption_cost': record['resumption_cost'],
            'isolation': ('The packet contains no Lean source at this cut (0 .lean files), so nothing in the CLOSED '
                          'domain (Math-/formal/ResearchFormalCoreR1 and main/formal/UniversalLaw at their admitted '
                          'cuts) can import it. This record excludes mathematical promotion from this region; source '
                          'navigation and diagnostic uses remain outside that admission. The composer must adopt it '
                          'before it becomes a terminal quarantine.'),
            'owner': 'none',
            'maintenance': 'zero',
            'source_statements': record['statements'],
            'source_hypotheses': record['hypotheses'],
            'source_dependencies': record['dependencies'],
            'source_self_limits': record['source_self_limits'],
            'finite_checks': record['finite_checks'],
            'floating_point': record['floating_point'],
            'source_binding': {'repository': REPO, 'commit': CUT, 'proof': fact['proof'],
                               'proof_git_blob': fact['proof_blob'], 'proof_sha256': fact['proof_sha256'],
                               'proof_bytes': fact['proof_bytes'], 'files': fact['files']},
            'verification': {'verifier_ok_first_pass': verdict['ok'],
                             'verifier_problems_fixed': verdict['problems']},
        })
    if problems:
        print('\n'.join(problems))
        print(f'{len(problems)} problem(s); nothing written')
        return 1
    report = {'schema': 'claude54.quarantine-proposals.v1', 'math_cut': CUT,
              'catalog_target': 'main#695 cbz/quarantine/catalog.json quarantines (schema of existing records)',
              'scope': 'PROPOSED records only; no CLOSED promotion; scientific effect NONE',
              'records': records}
    with open(output, 'x') as stream:
        json.dump(report, stream, indent=1, ensure_ascii=False)
        stream.write('\n')
    print(f'{len(records)} records written; first-pass ok {sum(r["verification"]["verifier_ok_first_pass"] for r in records)}')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
