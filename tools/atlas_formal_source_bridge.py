"""Reproduce one explicit source bridge to an unchanged historical artifact.

This reads frozen historical control text as data, never imports or executes it.
Original execution and alignment stay at their original manifest and revision.
No network, Lean execution, arbitrary manifest tolerance or scientific promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

if __package__:
    from .atlas_formal_evidence_build import build_connection, digest, blob, require, historical_source_identity
    from .formal_gate_check import safe_path
    from .required_formal_check import strict_json
else:
    from atlas_formal_evidence_build import build_connection, digest, blob, require, historical_source_identity
    from formal_gate_check import safe_path
    from required_formal_check import strict_json

CURRENT_MANIFEST='1a09020f0706abd506824f54fca76df9daa533749350b9ab6b818e0bbe9ed744'
CURRENT_BLOB='8228e896c728d95ab94391899320e9d44ab7cdfc'
CURRENT_SOURCE='c1678a556ab487954abbdc2456686bb64ddc7114'
CORRECTED_MANIFEST='38eefbfdd84658cd0f6e18bd7d714458a7522b1021f73b1075e7972a63b9bcb0'
CORRECTED_BLOB='a22f06a3a7b9ab451e9e344d7b2345d2e2ab0553'
CORRECTED_SOURCE='a30e336982ff4a9400182961ae1f8a241ceff927'
OLD_BRIDGE='cdb075de1dd63b4313135c166078b71d827d04c882222accc415d8ddf97fa414'
OLD_MANIFEST='8dfcfe4e9296bca47eede517f37b546ea420ec6b51944b7c16164a906b40e090'
OLD_EVIDENCE='441babc5f605492b361822c26d59ebee4126e45b90695915195bf951c3c943ef'
OLD_PROJECTION='7b36e2376f4010444108e81f316293ef3b3834d767e0900f68d684c760af2db1'
LEGACY='tests/fixtures/atlas_formal_build_3b39'
LEGACY_DATA={
    'formal/README.md':'README.md.txt',
    'formal/REVIEW_LANE.md':'REVIEW_LANE.md.txt',
    'tools/formal_gate_check.py':'formal_gate_check.py.txt',
    'tests/test_formal_gate.py':'test_formal_gate.py.txt',
}
CONTROLS={'tools/formal_gate_check.py','tests/test_formal_gate.py'}
DOCUMENTATION={'formal/README.md','formal/REVIEW_LANE.md'}
ANNOTATED_TARGETS={'UniversalLaw.Side24.d3_below_d2','UniversalLaw.Side24.endpoints_in_unit_tenth'}
EVIDENCE_DIR='docs/site/evidence/side24-main-3b39'
RECORD='evidence/side24-main-3b39/source-bridge-8228.json'
MODULE='atlas-formal-source-bridge-data.mjs'


def read(root,relative):
    return safe_path(root,relative).read_bytes()


def evidence_data(raw):
    require(digest(raw)==OLD_EVIDENCE,'historical evidence identity differs')
    return strict_json(raw.decode().split('export const EVIDENCE = ',1)[1].removesuffix(';\n'))


def build_bridge(root,snapshot):
    root=Path(root).resolve()
    raw=read(root,'formal/manifest.json')
    corrected=digest(raw)==CORRECTED_MANIFEST
    manifest_sha,manifest_blob,manifest_bytes,source_revision=(
        (CORRECTED_MANIFEST,CORRECTED_BLOB,18104,CORRECTED_SOURCE) if corrected else
        (CURRENT_MANIFEST,CURRENT_BLOB,17521,CURRENT_SOURCE))
    require(digest(raw)==manifest_sha and blob(raw)==manifest_blob and len(raw)==manifest_bytes,
            'unrecognized projection manifest; no bridge inferred')
    current,_,_=historical_source_identity(root, manifest_sha)
    manifests=[s for s in snapshot['sources'] if s['repository']=='d6g8k5htny-coder/main'
               and s['path']=='formal/manifest.json']
    require(len(manifests)==1 and manifests[0]['blob']==manifest_blob and manifests[0]['bytes']==manifest_bytes,
            'projection manifest inventory missing, changed or ambiguous')

    old_raw=read(root,LEGACY+'/manifest.json')
    require(digest(old_raw)==OLD_MANIFEST,'historical input identity differs: manifest')
    old=strict_json(old_raw.decode())
    require(set(old['files'])==set(current['files']),'formal file inventory changed')
    changed={path:{'before_sha256':old['files'][path],'after_sha256':current['files'][path]}
             for path in old['files'] if old['files'][path]!=current['files'][path]}
    allowed_changes=CONTROLS|DOCUMENTATION if corrected else CONTROLS
    require(set(changed)==allowed_changes,'unexpected formal source or control delta')
    normalized={**old,'files':current['files']}
    annotations={}
    if corrected:
        normalized['targets']=[dict(target) for target in old['targets']]
        for before,after in zip(normalized['targets'],current['targets']):
            if before['name'] in ANNOTATED_TARGETS:
                annotations[before['name']]={key:{'before':before[key],'after':after[key]}
                                             for key in ['informal_anchor','does_not_claim']}
                for key in ['informal_anchor','does_not_claim']:before[key]=after[key]
        require(set(annotations)==ANNOTATED_TARGETS,'unexpected target annotation delta')
    require(normalized==current,'non-file manifest metadata changed')

    evidence_raw=read(root,'docs/site/atlas-formal-evidence-data.mjs')
    evidence=evidence_data(evidence_raw)
    projection_raw=read(root,'tests/fixtures/atlas_formal_projection.json')
    require(digest(projection_raw)==OLD_PROJECTION,'historical input identity differs: projection')
    original_projection=strict_json(projection_raw.decode())
    identical={}
    with tempfile.TemporaryDirectory() as folder:
        historical=Path(folder)
        for path,expected in old['files'].items():
            value=read(root,LEGACY+'/'+LEGACY_DATA[path]) if path in LEGACY_DATA else read(root,path)
            require(digest(value)==expected,'historical input identity differs: '+path)
            destination=historical/path;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(value)
            if path not in allowed_changes:
                require(value==read(root,path),'formal source bytes changed: '+path)
                identical[path]={'bytes':len(value),'sha256':expected,'blob':blob(value)}
        (historical/'formal/manifest.json').write_bytes(old_raw)
        for path in ['docs/site/dependency-source/GRAPH.json',EVIDENCE_DIR+'/review.json',EVIDENCE_DIR+'/review.md']:
            destination=historical/path;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(read(root,path))
        # Current validator reads these old files only as literal hash-bound data.
        historical_source_identity(historical)
        proof=read(root,'formal/sources/side24_v1/PROOF.md')
        artifact=read(root,EVIDENCE_DIR+'/verified-execution.zip')
        original=build_connection(historical,original_projection,proof,artifact)
        require(original==evidence,'historical evidence identity did not reproduce exactly')
        projected=build_connection(historical,snapshot,proof,artifact)
        require(projected['claim']==original['claim'],'current graph record differs from pinned claim')

    result={
        'schema_version':1,
        'kind':'explicit source-identity bridge to a historical arithmetic artifact',
        'claim_id':original['claim']['id'],
        'projection_manifest':{'repository':'d6g8k5htny-coder/main','commit':source_revision,
                               'path':'formal/manifest.json','blob':manifest_blob,
                               'bytes':len(raw),'sha256':manifest_sha},
        'historical_artifact':{'commit':original['formal']['commit'],
                               'manifest_sha256':OLD_MANIFEST,'evidence_sha256':OLD_EVIDENCE,
                               'receipt_sha256':original['execution']['receipt_sha256']},
        'input_graph_identity':snapshot['graph']['identity'],
        'input_document_identity':projected['document'],
        'historical_evidence_reproduced_exactly':True,
        'identical_formal_files':identical,
        'changed_control_files':{path:change for path,change in changed.items() if path in CONTROLS},
        'source_validators':'Existing source_check on both projections; original build_connection and receipt/log binding.',
        'historical_inputs':{'source_revision':'336b6f8927f37e2a8d727c6bea845f874c52d8f5',
                             'directory':LEGACY,'manifest_sha256':OLD_MANIFEST,
                             'projection_sha256':OLD_PROJECTION,
                             'handling':'Literal data only; no historical code imported or executed.'},
        'execution_boundary':'Only original3b39/run40246431034/attempt1 execution. No current execution inferred.',
        'alignment_boundary':'Original manifest alignment PENDING; submitted31-target review REJECTED(AMEND). No alignment transferred to the current manifest.',
        'coverage_boundary':'Closed Nat arithmetic pairingSum6=76 only; no derivative bound, coefficient enclosure or parent theorem.',
        'scientific_effect':'NONE',
    }
    if corrected:
        result['changed_documentation_files']={path:change for path,change in changed.items() if path in DOCUMENTATION}
        result['changed_target_annotations']=annotations
    return result


def bridge_path(record):
    return 'evidence/side24-main-3b39/source-bridge-'+('a22f' if record['projection_manifest']['blob']==CORRECTED_BLOB else '8228')+'.json'


def whitelist_entry(record,detailed):
    entry={key:record[key] for key in ['schema_version','claim_id','projection_manifest','historical_artifact','scientific_effect']}
    entry.update(identical_formal_file_count=len(record['identical_formal_files']),
                 changed_control_file_count=len(record['changed_control_files']),
                 record_path=bridge_path(record),record_sha256=digest(detailed.encode()))
    if 'changed_documentation_files' in record:
        entry['changed_documentation_file_count']=len(record['changed_documentation_files'])
        entry['changed_target_annotation_count']=len(record['changed_target_annotations'])
    return entry


def render_bridge(record,previous=None):
    detailed=json.dumps(record,ensure_ascii=False,indent=2)+'\n'
    entries=[whitelist_entry(record,detailed)]
    if previous is not None:
        require(digest(previous)==OLD_BRIDGE,'historical bridge identity differs')
        historical=strict_json(previous.decode())
        entries.insert(0,whitelist_entry(historical,previous.decode()))
    module=('// Generated by tools/atlas_formal_source_bridge.py; historical execution and alignment remain unchanged.\n'
            'export const SOURCE_BRIDGES = '+json.dumps(entries,ensure_ascii=False,indent=2)+';\n')
    return detailed,module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    record=build_bridge(root,strict_json(args.snapshot.read_text()))
    previous=read(root,'docs/site/'+RECORD) if record['projection_manifest']['blob']==CORRECTED_BLOB else None
    detailed,module=render_bridge(record,previous)
    for relative,value in [(bridge_path(record),detailed),(MODULE,module)]:
        target=root/'docs/site'/relative
        expected=value.encode('utf-8')
        if args.check:
            require(target.is_file() and target.read_bytes()==expected,'generated bridge differs: '+relative)
        else:
            if relative!=MODULE and target.exists():
                require(target.read_bytes()==expected,'immutable bridge record differs: '+relative)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(expected)
    print(json.dumps({'source_bridge':record['claim_id'],'current_manifest':record['projection_manifest']['blob'],
                      'historical_artifact':record['historical_artifact']['commit'],
                      'identical_formal_files':len(record['identical_formal_files']),
                      'changed_controls':2,'scientific_effect':'NONE'}))


if __name__=='__main__':main()
