#!/usr/bin/env python3
"""Verify contribution data identity, never scientific closure completeness."""
from pathlib import Path
import copy, datetime as dt, hashlib, json, re
ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'cbz/evidence/website-foundation-20261029'

def require(condition, message='contribution identity invariant failed'):
    if not condition:
        raise AssertionError(message)

def load(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def digest(j):
    return hashlib.sha256(json.dumps(j, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def validate(index, hashes=True):
    records = index['records']
    require(len(records) == 207 and len({r['id'] for r in records}) == 207, 'contribution identity invariant failed')
    require(index['coordination']['all_main_obligations_complete'] is False, 'contribution identity invariant failed')
    byid = {r['id']: r for r in records}
    require({s: sum((r['status'] == s for r in records)) for s in ['CLOSED', 'QUARANTINED']} == {'CLOSED': 31, 'QUARANTINED': 176}, 'contribution identity invariant failed')
    dataset = load(OUT / 'sources/legacy-v20/engine/dataset.json')
    graph = load(OUT / 'sources/Math-/frontiers/downstream_gate_20260925/GRAPH.json')
    main = load(OUT / 'sources/lean-foundation/MAIN_PROOF_INDEX.json')
    for p in dataset['records']:
        require(byid['V20:' + p['id']]['original_record'] == p, 'contribution identity invariant failed')
    for ident, p in graph['nodes'].items():
        require(byid['MATH_GRAPH:' + ident]['original_record'] == p, 'contribution identity invariant failed')
    require(index['legacy_obligations_exact'] == dataset['obligations'] and index['legacy_imports_exact'] == dataset['imports'], 'contribution identity invariant failed')
    require(index['original_math_graph_edges'] == graph['edges'], 'contribution identity invariant failed')
    for p in main['proofs']:
        require(byid['MAIN:' + p['declaration']]['original_record'] == p, 'contribution identity invariant failed')
    for r in records:
        require(digest(r['original_record']) == r['source_record_sha256'], r['id'])
        require(r['source_hashes'] and 'exact_boundary' in r and r['scientific_closure_limits'], 'contribution identity invariant failed')
        require(isinstance(r['dependencies'], list), 'contribution identity invariant failed')
        if r['status'] == 'CLOSED':
            require(r['axioms'] == [] and r['kernel_evidence'] and (r['category'] == 'literal_main_arithmetic'), 'contribution identity invariant failed')
            require(not r['dependencies'], 'closed imports cannot be inferred from Q records')
            require(r['exact_boundary']['statement'] == r['original_record']['statement'], 'contribution identity invariant failed')
            require(r['exact_boundary']['nonvacuity'] == r['original_record']['nonvacuity'], 'contribution identity invariant failed')
        else:
            require(r['owner'] == 'none' and r['reason'] and r['attempts'] and r['isolation'], 'contribution identity invariant failed')
        if hashes:
            for b in r['source_hashes'] + r.get('kernel_evidence', []):
                require(sha(ROOT / b['path']) == b['sha256'], b)
    return True

def main():
    index = load(OUT / 'proof-index.json')
    validate(index)
    controls = []

    def corrupt_record(j):
        j['records'][31]['original_record'].pop('scope', None)

    def overclaim(j):
        j['coordination']['all_main_obligations_complete'] = True

    def promote_q(j):
        j['records'][31]['status'] = 'CLOSED'

    def remove_witness(j):
        j['records'][0]['exact_boundary'].pop('nonvacuity')
    for name, mutation in [('deleted_original_scope', corrupt_record), ('all_main_overclaim', overclaim), ('quarantine_promoted', promote_q), ('missing_nonvacuity', remove_witness)]:
        j = copy.deepcopy(index)
        mutation(j)
        try:
            validate(j, False)
        except (AssertionError, KeyError):
            controls.append({'name': name, 'detected': True})
        else:
            raise AssertionError('missed ' + name)
    manifest = load(OUT / 'source-manifest.json')
    for b in manifest['files']:
        require(sha(ROOT / b['path']) == b['sha256'], 'contribution identity invariant failed')
    for p in OUT.glob('*.proposed.md'):
        markers = re.findall('<!-- cbz-gates: (.+?) -->', p.read_text())
        require(len(markers) == 1, p)
        require(json.loads(markers[0]) == {'G' + str(i): 'INCOMPLETE' for i in range(12)}, 'contribution identity invariant failed')
    require(len(list(OUT.glob('*.proposed.md'))) == 7, 'contribution identity invariant failed')
    prior = load(OUT / 'archive/STATUS.pre-cbz-20261029.json')
    require(sha(OUT / 'archive/STATUS.pre-cbz-20261029.md') == prior['sha256'], 'contribution identity invariant failed')
    require(sha(ROOT / 'STATUS.md') == prior['sha256'], 'root STATUS changed during proposal')
    registry = load(OUT / 'tool-registry.json')
    for tool in registry['tools']:
        require((ROOT / tool['path']).is_file() and isinstance(tool['self_test'], list), 'contribution identity invariant failed')
        for b in tool['source_hashes']:
            require(sha(ROOT / b['path']) == b['sha256'], 'contribution identity invariant failed')
    result = {'verified_utc': dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z'), 'result': 'PASS_CONTRIBUTION_DATA_CHECKS_ONLY', 'records': 207, 'closed_literal_arithmetic': 31, 'quarantined': 176, 'original_legacy_records_verified': 104, 'original_graph_records_verified': 49, 'source_files_verified': len(manifest['files']), 'root_status_unchanged': True, 'controls': controls, 'scope': 'Record identity, exact fields, byte hashes, finite metadata/isolation checks and proposal gate markers; no scientific/kernel rerun or exhaustive Main obligation closure.'}
    print(json.dumps(result))
if __name__ == '__main__':
    main()
