"""Coverage/data-identity negative controls; no scientific authority is implied."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
from tools.drive_coverage import check


def sha(data):return hashlib.sha256(data).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    base=tmp_path/'bundle';(base/'packs').mkdir(parents=True)
    data=b'inert source bytes\n';digest=sha(data)
    raw=io.BytesIO()
    with zipfile.ZipFile(raw,'w') as z:z.writestr(digest,data)
    packed=raw.getvalue();pack='packs/'+sha(packed)+'.zip';(base/pack).write_bytes(packed)
    source={'id':'file-1','title':'source.py','path':'research/source.py','mimeType':'text/x-python','bytes':len(data),'sha256':digest,'context':'RESEARCH_SOURCE_CHECK_STATUS'}
    inv=(json.dumps(source)+'\n').encode();(tmp_path/'inventory.jsonl').write_bytes(inv)
    scope={'schema_version':1,'inventory':{'path':'inventory.jsonl','sha256':sha(inv)},'additional_sources':[]}
    objects={'schema_version':1,'chunk_size':1048576,'objects':{digest:{'bytes':len(data),'chunks':[digest]}},'chunks':{digest:{'bytes':len(data),'pack':pack}}}
    row={'id':'file-1','context':source['context'],'status':'EXACT_SOURCE_BYTES','bytes':len(data),'sha256':digest,'storage':{'object':digest}}
    for name,value in [('scope.json',scope),('objects.json',objects)]: (base/name).write_text(json.dumps(value))
    (base/'coverage.jsonl').write_text(json.dumps(row)+'\n')
    return tmp_path,base,source,row,objects


def test_packed_bytes_and_scope_are_verified(bundle):
    root,base,source,row,objects=bundle
    result=check(root,'bundle')
    assert result['eligible_sources_accounted_for']
    assert result['counts']=={'EXACT_SOURCE_BYTES':1}


@pytest.mark.parametrize('mutation',['missing','duplicate','native_as_exact','source_mismatch','traversal','symlink','pack_corruption','chunk_order','context','nonstored_payload'])
def test_negative_controls_refuse_false_coverage(bundle,mutation):
    root,base,source,row,objects=bundle
    if mutation=='missing':(base/'coverage.jsonl').write_text('')
    elif mutation=='duplicate':(base/'coverage.jsonl').write_text((json.dumps(row)+'\n')*2)
    elif mutation=='pack_corruption':
        path=base/next(iter(objects['chunks'].values()))['pack'];path.write_bytes(path.read_bytes()+b'x')
    elif mutation=='chunk_order':
        next(iter(objects['objects'].values()))['chunks']*=2
        (base/'objects.json').write_text(json.dumps(objects))
    elif mutation=='context':row['context']='APPROVED_PROOF'
    elif mutation=='nonstored_payload':row.update(status='HELD_SOURCE_DRIFT',reason='drift')
    elif mutation in ('native_as_exact','source_mismatch'):
        source['mimeType']='application/vnd.google-apps.document' if mutation=='native_as_exact' else source['mimeType']
        if mutation=='source_mismatch':source['sha256']='0'*64
        data=(json.dumps(source)+'\n').encode();(root/'inventory.jsonl').write_bytes(data)
        scope=json.loads((base/'scope.json').read_text());scope['inventory']['sha256']=sha(data);(base/'scope.json').write_text(json.dumps(scope))
    elif mutation=='traversal':row['storage']={'file':'../outside'}
    elif mutation=='symlink':
        (root/'link').symlink_to(root/'inventory.jsonl');row['storage']={'file':'link'}
    if mutation not in ('missing','duplicate','pack_corruption','chunk_order','native_as_exact','source_mismatch'):
        (base/'coverage.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises((ValueError,KeyError,OSError)):check(root,'bundle')


def test_explicit_missing_is_reported_as_unresolved_never_verified(bundle):
    root,base,source,row,objects=bundle
    row={'id':row['id'],'context':row['context'],'status':'MISSING','reason':'not fetched'}
    (base/'coverage.jsonl').write_text(json.dumps(row)+'\n')
    result=check(root,'bundle')
    assert not result['eligible_sources_accounted_for'] and result['unresolved']==1
    assert result['counts']=={'MISSING':1}


def test_cli_extracts_original_bytes_and_refuses_overwrite(bundle):
    root,base,source,row,objects=bundle
    script=Path(__file__).resolve().parents[1]/'tools/drive_coverage.py'
    dest=root/'restored.py'
    args=[sys.executable,str(script),'--root',str(root),'--bundle','bundle','--extract-id','file-1','--output',str(dest)]
    first=subprocess.run(args,capture_output=True,text=True)
    assert first.returncode==0,first.stderr
    assert sha(dest.read_bytes())==source['sha256']
    second=subprocess.run(args,capture_output=True,text=True)
    assert second.returncode==1 and 'refusing to replace' in second.stderr


def test_duplicate_json_key_is_rejected(bundle):
    root,base,*_=bundle
    (base/'scope.json').write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError,match='duplicate JSON key'):check(root,'bundle')
