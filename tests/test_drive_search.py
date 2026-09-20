"""Keyword/hash retrieval: identities, archive occurrences, mentions and holds."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from tools import drive_index as di
from tools import drive_search as ds

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def records():
    return ds.load_current(di.load())


@pytest.mark.parametrize('bad',('',None,'abc','g'*64,'a'*65,'abcd ef12','../thing'))
def test_invalid_hash_rejected(bad):
    with pytest.raises(ValueError):ds.sha_matches([],bad)


def test_ambiguous_prefix_keeps_all_occurrences():
    rows=[{'record_key':str(i),'hashes':[{'sha256':s}]} for i,s in enumerate(('12345678'+'a'*56,'12345678'+'b'*56,'12345678'+'a'*56))]
    out=ds.sha_matches(rows,'12345678')
    assert out['ambiguous'] and len(out['matches'])==3 and len(out['distinct_digests'])==2
    out=ds.sha_matches(rows,'12345678'+'A'*56)
    assert not out['ambiguous'] and len(out['matches'])==2


def test_current_scope_and_post_cutoff_deliveries(records):
    assert len(records)==4934 and len({r['id'] for r in records})==4934
    assert sum(r['source_role']=='POST_CUTOFF_DELIVERY' for r in records)==6
    assert any(r['id']=='1Ektz5a0wJIHQXq5ao96tGX_2G5M4EeBp' for r in records)
    reg=next(r for r in records if r['id']=='1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no')
    assert reg['sha256']=='d6f4c7a693ad8788c28578c0a322960a09fe1d37c64d205bdd7e1a0ac9586194'
    assert reg['hashes'][0]['kind']=='NATIVE_EXPORT_SHA256' and not reg['revision_pinned']


def test_file_hit_never_suppresses_archive_member(records):
    h='bd3074fd900fc80bebdd430e34ae953f0a78685fff672758e7b927cc1e6d9421'
    out=ds.sha_matches(records+ds.archive_records(),h)
    assert not out['ambiguous']
    assert any(r['id']=='14FIaDPHfbTTGQWzxZi8xKAzlKub5Iud8' for r in out['matches'])
    assert any(r['id']=='1TJr1awRjB0D9niq1Jyi6XGnJKuvW9Wmx' and r['record_type']=='ARCHIVE_MEMBER' for r in out['matches'])
    assert all(r['match_kind']=='HASH_IDENTITY' for r in out['matches'])


def test_duplicate_archive_occurrences_preserved():
    rows=ds.archive_records()
    assert len(rows)==11649 and len({r['record_key'] for r in rows})==len(rows)
    out=ds.sha_matches(rows,'d9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8')
    assert len(out['matches'])==4
    assert len({r['id'] for r in out['matches']})>=3
    assert all(r['carrier_sha256']!=r['sha256'] for r in out['matches'])


def test_hash_mentions_do_not_become_identities(records):
    h='ac89f60b8206bfe011e6c2bc653e7acb39bc83a55c2a2c8fd1e17fd70c6c0383'
    identity=ds.sha_matches(records,h)
    mentions=ds.text_matches(records,h,hash_mention=True)
    assert any(r['id']=='1LtnvNd0vAW-y3pzbyHLjgtF7Uw5sTph5' for r in identity['matches'])
    assert mentions and all(r['match_kind']=='HASH_MENTION' for r in mentions)
    assert any(r['sha256']!=h for r in mentions)


def test_substantive_text_search_and_excluded_scopes(records):
    hits=ds.text_matches(records,'Cholesky')
    assert hits and all(r['content_eligible'] and ds.content_allowed(r) for r in hits)
    report=json.loads((ROOT/ds.SEARCH/'text-coverage.json').read_text())
    assert report['text_records']==3140 and not report['archive_member_text_indexed']
    assert report['counts']['SCOPE_EXCLUDED']==1012


def test_keyword_discovers_python_even_if_live_drive_search_misses(records):
    hits=ds.keyword_matches(records,'rnu_ds3.py')
    assert any(r['id']=='14FIaDPHfbTTGQWzxZi8xKAzlKub5Iud8' for r in hits)
    old=next(r for r in records if r['id']=='1v7JAh_FbXHYMaL7W6DNwEoLnHPwQ5U21')
    assert not old['content_eligible'] and old['source_role']=='METADATA_ONLY_EXCLUDED'


def test_cli_hash_validation_is_not_match_everything():
    run=subprocess.run([sys.executable,str(ROOT/'tools/drive_index.py'),'sha',''],capture_output=True,text=True)
    assert run.returncode==2 and not run.stdout and '8..64 hexadecimal' in run.stderr


def test_direct_sha_helper_also_preserves_archive_hits(records,capsys):
    assert di.cmd_sha(records,'bd3074fd900fc80b')==0
    lines=[json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert {x['record_type'] for x in lines}=={'FILE','ARCHIVE_MEMBER'}


def test_source_map_defaults_resolve_at_call_time(tmp_path,monkeypatch):
    p=tmp_path/'inventory.jsonl';p.write_text(json.dumps({'id':'fixture','path':'x'})+'\n')
    monkeypatch.setattr(di,'INVENTORY',str(p));monkeypatch.setattr(di,'DELTAS',str(tmp_path/'no-deltas'))
    assert di.load()==[{'id':'fixture','path':'x'}]


@pytest.mark.parametrize('mutation',('compressed_bytes','source_digest','scope'))
def test_disposable_cache_rejects_tampering_and_stale_scope(tmp_path,monkeypatch,mutation):
    text='Gaussian\u2028Cholesky\nsource'; data=text.encode()
    class Reader:
        def __init__(self,*args):pass
        def read(self,row):return data
    monkeypatch.setattr(ds,'Reader',Reader)
    monkeypatch.setattr(ds,'cache_path',lambda root:tmp_path/'cache.gz')
    (tmp_path/ds.SEARCH).mkdir(parents=True)
    e={'id':'x','record_key':'file:x','title':'x.txt','mimeType':'text/plain',
       'content_eligible':True,'sha256':ds.digest(data),'bytes':len(data),'custody':{}}
    ds.build_text_cache([e],tmp_path)
    assert len(ds.text_matches([e],'Cholesky',tmp_path))==1
    if mutation=='compressed_bytes':
        import gzip
        (tmp_path/'cache.gz').write_bytes(gzip.compress(b'changed'))
    elif mutation=='source_digest':e['sha256']='f'*64
    else:e['content_eligible']=False
    with pytest.raises(ValueError,match='cache'):
        ds.text_matches([e],'Cholesky',tmp_path)
