"""Reproduce one source-identity connection; no network or proof execution.

The fixed artifact was retrieved from its actual trusted run. Byte/log validation
does not authenticate arbitrary caller-supplied evidence or semantic alignment.
"""
from __future__ import annotations
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import tempfile
import zipfile

if __package__:
    from .formal_gate_check import source_check
    from .required_formal_check import bind_receipt, strict_json
else:
    from formal_gate_check import source_check
    from required_formal_check import bind_receipt, strict_json

CLAIM = 'math.side24-coefficient'
TARGET = 'UniversalLaw.Side24.pairing_sum_six'
COMMIT = '3b39c42d6edcc98ae165270be346f0e31c57f81d'
MANIFEST = '8dfcfe4e9296bca47eede517f37b546ea420ec6b51944b7c16164a906b40e090'
ARTIFACT = '80ac88cf770afbadd89f4801a4692d6b6540a9269dfb916c249d491c2b41f737'
RECEIPT = '1ae8341c86a043ba917ec1fe68bf7e66ec528cd94b15170528487009f090c7e0'
REVIEW = 'ce0540dd8ac49970fd7ddb92fdb470725e2fb7ee0a69be27a6e955c8b8632852'
REVIEW_MD = '78c21c96d851dfd6a4fbcf68edb091d3aac21b757885e615403d3fe6337e1692'
EVIDENCE_DIR = 'docs/site/evidence/side24-main-3b39'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build_connection(root, snapshot, proof, artifact):
    root = Path(root)
    m, manifest_digest, _ = source_check(root)
    require(manifest_digest == MANIFEST, 'formal manifest changed')
    target = next(t for t in m['targets'] if t['name'] == TARGET)
    source = next(s for s in m['sources'] if s['id'] == target['source'])
    require(len(proof) == source['bytes'] and digest(proof) == source['sha256'],
            'proof bytes differ from the manifest source')
    require(source['sha256'] == snapshot['graph']['data']['nodes'].get(CLAIM, {}).get('fingerprint'),
            'graph record fingerprint differs from proof')
    graph_raw = (root/'docs/site/dependency-source/GRAPH.json').read_bytes()
    require(digest(graph_raw) == snapshot['graph']['identity']['sha256']
            and blob(graph_raw) == snapshot['graph']['identity']['blob'],
            'graph record source identity differs')
    graph = strict_json(graph_raw.decode())
    require(graph['nodes'].get(CLAIM) == snapshot['graph']['data']['nodes'].get(CLAIM)
            and CLAIM in snapshot['graph']['data']['nodes'], 'graph record changed')
    node = graph['nodes'][CLAIM]
    candidates = [s for s in snapshot['sources']
                  if s['repository'] == source['repository'] and s['path'] == node['source']]
    require(len(candidates) == 1, 'proof inventory missing or ambiguous')
    inventory = candidates[0]
    require(inventory['path'] == source['path'] and inventory['bytes'] == len(proof)
            and inventory['blob'] == blob(proof), 'proof inventory differs from original bytes')
    require(digest(artifact) == ARTIFACT, 'original execution artifact identity differs')
    with zipfile.ZipFile(BytesIO(artifact)) as archive, tempfile.TemporaryDirectory() as temp:
        require(len(archive.namelist()) == len(set(archive.namelist())), 'duplicate artifact member')
        require(archive.testzip() is None, 'artifact CRC failure')
        receipt_raw = archive.read('receipt/receipt.json')
        receipt = strict_json(receipt_raw.decode())
        folder = Path(temp)
        (folder/'receipt.json').write_bytes(receipt_raw)
        for name in receipt['logs']:
            require(re.fullmatch(r'[A-Za-z0-9_-]+\.log', name), 'unsafe receipt log')
            (folder/name).write_bytes(archive.read('receipt/'+name))
        env = {'GITHUB_REPOSITORY': 'd6g8k5htny-coder/main', 'GITHUB_SHA': COMMIT,
               'GITHUB_RUN_ID': '40246431034', 'GITHUB_RUN_ATTEMPT': '1'}
        binding = bind_receipt(folder/'receipt.json', root/'formal/manifest.json', env, COMMIT)
        require(binding == strict_json(archive.read('receipt/required-check-binding.json').decode()),
                'original required-check binding differs')
        require(binding['receipt_sha256'] == RECEIPT, 'original receipt identity differs')
        require(archive.read('commit.txt').decode().strip() == COMMIT, 'artifact source revision differs')
        require(set(receipt['axioms']) == {t['name'] for t in m['targets']},
                'receipt declaration coverage differs')
        types = (folder/'elaborated-types.log').read_text()
    lean_path = 'formal/'+target['module']
    lean = (root/lean_path).read_text()
    statement = re.search(r'^theorem pairing_sum_six\s*:[\s\S]*?(?=:=)', lean, re.M)
    require(statement is not None and statement.group().strip()
            == 'theorem pairing_sum_six : pairingSum 6 = 76', 'closed declaration changed')
    elaborated = re.search(re.escape(TARGET)+r' :\n[\s\S]*?(?=^UniversalLaw\.Side24\.|\Z)', types, re.M)
    require(elaborated is not None, 'checked elaborated declaration missing')
    review_raw = (root/EVIDENCE_DIR/'review.json').read_bytes()
    review_md = (root/EVIDENCE_DIR/'review.md').read_bytes()
    require(digest(review_raw) == REVIEW and digest(review_md) == REVIEW_MD,
            'review evidence differs from submitted immutable bytes')
    review = strict_json(review_raw.decode())
    require(review['manifest_sha256'] == manifest_digest
            and review['scope_sha256'] == m['files']['formal/SCOPE.md']
            and set(review['targets']) == {t['name'] for t in m['targets']}
            and len(review['targets']) == len(m['targets'])
            and review['author'] == m['author'] and review['disposition'] == 'REJECTED'
            and review['evidence']['sha256'] == REVIEW_MD, 'review source or disposition differs')
    row = next(line for line in review_md.decode().splitlines()
               if '| '+chr(96)+TARGET+chr(96)+' |' in line)
    selected_verdict = re.search(r'\*\*([^*]+)\*\*', row).group(1)
    manifest_raw = (root/'formal/manifest.json').read_bytes()
    return {
        'schema_version': 1,
        'claim': {'id': CLAIM, 'record': node},
        'document': {**inventory, 'sha256': digest(proof), 'manifest_source': source},
        'formal': {'repository': 'd6g8k5htny-coder/main', 'commit': COMMIT,
                   'package': m['package'], 'manifest_path': 'formal/manifest.json',
                   'manifest_sha256': MANIFEST, 'manifest_blob': blob(manifest_raw),
                   'manifest_bytes': len(manifest_raw), 'scope_sha256': m['files']['formal/SCOPE.md'],
                   'formalization_status': m['formalization_status'],
                   'alignment_status': m['alignment_status']},
        'declaration': {**target, 'module': lean_path,
                        'line': lean[:statement.start()].count('\n')+1,
                        'source_statement': statement.group().strip(),
                        'elaborated_type': elaborated.group().rstrip(),
                        'hypothesis_binders': []},
        'execution': {**binding, 'artifact_id': 12618189537, 'artifact_sha256': ARTIFACT,
                      'lean_version': receipt['lean_version'], 'formalization_status': receipt['formalization_status'],
                      'target_axioms': receipt['axioms'][TARGET],
                      'types_sha256': receipt['logs']['elaborated-types.log']},
        'alignment': {'record': review, 'selected_target_verdict': selected_verdict,
                      'record_source': {'repository': 'd6g8k5htny-coder/main',
                                        'commit': 'a228f4410141549462260ba5ea426b7d08c358c1',
                                        'path': 'audits/formal_alignment/20261026_grok_bot_agent_3.json',
                                        'sha256': REVIEW}},
        'scientific_effect': 'NONE',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    assets = root/EVIDENCE_DIR
    result = build_connection(root, strict_json(args.snapshot.read_text()),
                              (root/'formal/sources/side24_v1/PROOF.md').read_bytes(),
                              (assets/'verified-execution.zip').read_bytes())
    rendered = ('// Reproduced by tools/atlas_formal_evidence_build.py; source identity is not claim coverage.\n'
                'export const EVIDENCE = '+json.dumps(result, ensure_ascii=False, indent=2)+';\n')
    output = root/'docs/site/atlas-formal-evidence-data.mjs'
    if args.check:
        require(output.read_text() == rendered, 'generated evidence differs')
    else:
        output.write_text(rendered)
    print(json.dumps({'source_connection': CLAIM, 'declaration': TARGET,
                      'receipt_sha256': RECEIPT, 'submitted_alignment': result['alignment']['record']['disposition'],
                      'scientific_effect': 'NONE'}))


if __name__ == '__main__':
    main()
