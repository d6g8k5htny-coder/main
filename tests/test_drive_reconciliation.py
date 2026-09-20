"""Metadata unknowns, missing membership and native export loss controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools.drive_reconcile import DEFAULT, metadata_value, reconcile, check
from tools.native_export_check import compare

ROOT=Path(__file__).resolve().parents[1]


def folders(parent_marker=True):
    item={'id':'file','title':'example','mime_type':'text/plain','size':'12','modified_time':'yesterday'}
    if parent_marker is not ...:item['parent_ids']=parent_marker
    return [{'id':'folder','complete':True,'count':1,'items':[item]}]


@pytest.mark.parametrize('parent', [None, [], ..., ['different-metadata-parent']])
def test_unknown_or_empty_parent_metadata_is_not_an_observed_move(parent):
    result=reconcile(folders(['folder']),folders(parent))
    assert result['membership_changes']==[]
    assert result['changed']==[]
    assert result['not_observed']==[]


def test_presence_is_not_collapsed():
    assert metadata_value({},'x')==('MISSING',None)
    assert metadata_value({'x':None},'x')==('NULL',None)
    assert metadata_value({'x':[]},'x')==('OBSERVED',[])


def test_removal_is_not_silently_discarded_or_called_deletion():
    new=folders();new[0].update(count=0,items=[])
    result=reconcile(folders(),new)
    assert result['not_observed']==['file']
    assert 'deleted' not in result


@pytest.mark.parametrize('mutation', ['incomplete','duplicate-folder','duplicate-item','count','universe'])
def test_incomplete_discovery_refuses(mutation):
    new=folders()
    if mutation=='incomplete':new[0]['complete']=False
    elif mutation=='duplicate-folder':new.append(deepcopy(new[0]))
    elif mutation=='duplicate-item':new[0]['items']*=2;new[0]['count']=2
    elif mutation=='count':new[0]['count']=1000
    else:new[0]['id']='elsewhere'
    with pytest.raises(ValueError):reconcile(folders(),new)


def test_timestamp_change_requires_current_payload():
    new=folders();new[0]['items'][0]['modified_time']='today'
    assert reconcile(folders(),new)['changed']==[{'id':'file','fields':['modified_time']}]


def native_fixture():
    n=[{'id':'sheet','kind':'sheet','observed_at':'2026-09-20','requested_ranges':["'Long title'!A1:B20"],
        'sheets':[{'properties':{'sheetId':1,'index':0,'title':'Long title','gridProperties':{'rowCount':20,'columnCount':2}},
                   'formulas':[{'row':2,'column':1,'formula':'=SUM(A3:A)'}]}]}]
    e=[{'id':'sheet','kind':'sheet','container_valid':True,'sheets':[{'title':'Long title','formulas':[{'cell':'A2','formula':'=SUM(A3:A20)','attributes':{}}]}]}]
    return n,e


def test_native_formula_rewrite_is_preserved_not_assumed_equivalent():
    n,e=native_fixture();result=compare(n,e)
    assert result['counts']['transformed_nonshared_formulas']==1
    assert result['nonshared_formula_transformation_examples'][0]['native']=='=SUM(A3:A)'


@pytest.mark.parametrize('mutation',['missing-tab','missing-formula','native-formula-dropped','partial-read','bad-container','unresolved-shared','duplicate-id'])
def test_native_export_loss_controls(mutation):
    n,e=native_fixture()
    if mutation=='missing-tab':e[0]['sheets']=[]
    elif mutation=='missing-formula':e[0]['sheets'][0]['formulas']=[]
    elif mutation=='native-formula-dropped':n[0]['sheets'][0]['formulas']=[]
    elif mutation=='partial-read':n[0]['requested_ranges']=["'Long title'!A1:A2"]
    elif mutation=='bad-container':e[0]['container_valid']=False
    elif mutation=='unresolved-shared':e[0]['sheets'][0]['formulas'][0].update(formula='=',attributes={'t':'shared','si':'1'})
    else:n.append(deepcopy(n[0]))
    with pytest.raises(ValueError):compare(n,e)


def test_multitab_document_refuses_first_tab_only_assumption():
    n=[{'id':'doc','kind':'doc','observed_at':'now','revision_id':'r','tabs':[{'tabId':'first'},{'tabId':'second'}]}]
    e=[{'id':'doc','kind':'doc','container_valid':True,'document_body':True}]
    with pytest.raises(ValueError):compare(n,e)


def test_forged_rn_report_fails_cli(tmp_path):
    p=ROOT/'research/rn/candidates/affine_moments_20260920.json'
    data=json.loads(p.read_text());data['field_certified']=True
    forged=tmp_path/'forged.json';forged.write_text(json.dumps(data))
    r=subprocess.run([sys.executable,str(ROOT/'tools/rn_moment_report.py'),'--check',str(forged)],capture_output=True,text=True)
    assert r.returncode!=0 and 'mismatch' in r.stderr


def test_numerical_overlap_ignores_changed_case_labels():
    from tools.rn_moment_report import fixture,numerical_identity,numerical_overlaps
    assert fixture(0).fingerprint()!=fixture(105).fingerprint()
    assert numerical_identity(fixture(0))==numerical_identity(fixture(105))
    overlaps=numerical_overlaps(range(4),range(100,116))
    assert [(r['evaluation_case'],r['development_case']) for r in overlaps]==[(105,0),(106,1),(107,2),(108,3)]


@pytest.mark.parametrize('mutation',['none','missing-payload','digest','counts','baseline-pin','global-overclaim','changed-cutoff'])
def test_reconciliation_cli_payload_controls(tmp_path,mutation):
    base=tmp_path/'base';base.mkdir()
    (base/'coverage.jsonl').write_text(json.dumps({'id':'file','status':'RAW_SOURCE_SNAPSHOT'})+'\n')
    for name in ('scope.json','objects.json','_MANIFEST.jsonl'):(base/name).write_text('{}\n')
    old=folders();new=deepcopy(old);new[0]['observed_at']='2026-09-20T00:00:00Z'
    new[0]['items'].append({'id':'added','title':'new'});new[0]['count']=2
    bundle=tmp_path/'delta';bundle.mkdir();(bundle/'data').write_bytes(b'new bytes')
    payload={'id':'added','path':'delta/data','status':'RAW_SOURCE_SNAPSHOT','bytes':9,'sha256':hashlib.sha256(b'new bytes').hexdigest()}
    spec={'schema_version':1,'baseline_coverage':'base/coverage.jsonl',
          'baseline_pins':[{'path':'base/'+p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in base.iterdir()],
          'comparison':reconcile(old,new),'payloads':[payload],'current_counts':{'RAW_SOURCE_SNAPSHOT':2},
          'scoped_objects':2,'not_in_folder_membership':[],'whole_account_complete':False,
          'historical_revisions_complete':False,'membership_cutoff':new[0]['observed_at']}
    if mutation=='missing-payload':spec['payloads']=[]
    elif mutation=='digest':payload['sha256']='0'*64
    elif mutation=='counts':spec['current_counts']['RAW_SOURCE_SNAPSHOT']=1
    elif mutation=='baseline-pin':spec['baseline_pins'].pop()
    elif mutation=='global-overclaim':spec['whole_account_complete']=True
    elif mutation=='changed-cutoff':spec['membership_cutoff']='2099-01-01'
    for name,data in [('reconciliation.json',spec),('previous-membership.json',old),('current-membership.json',new)]:
        (bundle/name).write_text(json.dumps(data))
    result=subprocess.run([sys.executable,str(ROOT/'tools/drive_reconcile.py'),'--root',str(tmp_path),'--bundle','delta'],capture_output=True,text=True)
    assert (result.returncode==0)==(mutation=='none'),result.stderr
