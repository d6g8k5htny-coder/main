import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import * as m from '../docs/site/current-model.mjs';
const commit='a'.repeat(40),main='c'.repeat(40),digest='b'.repeat(64),pin='math.d5-component.punctured-pin-proof';
const snapshot=nodes=>({schema_version:1,observed_at:'2026-10-13T12:00:00Z',scientific_effect:'NONE',repositories:[{repository:'d6g8k5htny-coder/main',commit:main,tree:main},{repository:'d6g8k5htny-coder/Math-',commit,tree:commit}],sources:[{repository:'d6g8k5htny-coder/Math-',commit,path:'proof/PROOF.md',blob:commit,bytes:20}],graph:{identity:{repository:'d6g8k5htny-coder/Math-',commit,path:'frontiers/x/GRAPH.json',blob:commit,bytes:20,sha256:digest},data:{nodes,edges:[]}},equations:[],references:[]});
const KEYS=['node','snapshot','freshness','warning','state','review','operational','continuation','continuation_text'];
const unrecorded=view=>view.operational.every(row=>row.text===m.RESUME_NOT_RECORDED)&&view.operational.map(row=>row.field).join()==='next_action,missing_input,owner_evidence,retained_stop';
// A finished, closed or accepted review and lookalike metadata never become task state.
const closed=()=>snapshot({
  done:{kind:'candidate',classification:'PROVED_REVIEWED',controlling:false,review_disposition:'ACCEPT',review_issue:67,review_issue_state:'closed',review_source:'https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206',source:'proof/PROOF.md',fingerprint:digest},
  lookalike:{kind:'candidate',classification:'OPEN_ACTIVE',next_action:'merge now',missing_input:'none',owner_evidence:'Claude S owns this',retained_stop:'none',owner:'Claude S',status:'complete',task:'https://github.com/d6g8k5htny-coder/main/pull/1'},
  bare:{kind:'axiom'},
  twin:{kind:'candidate',classification:'OPEN_ACTIVE',source:'proof/PROOF.md',fingerprint:digest},
  [pin]:{kind:'reading_rule_component',classification:'PROVED_REVIEWED',source:'proof/PROOF.md',fingerprint:digest},
});
test('resume projection labels exact snapshot identity and separates review records from operational state',()=>{
  const view=m.resumeProjection(closed(),'done');assert.deepEqual(Object.keys(view),KEYS);
  assert.deepEqual(view.snapshot,{observed_at:'2026-10-13T12:00:00Z',graph_repository:'d6g8k5htny-coder/Math-',graph_commit:commit,graph_path:'frontiers/x/GRAPH.json',graph_sha256:digest,main_commit:main});
  assert.equal(view.freshness,null);assert.equal(view.warning,m.RESUME_FRESHNESS_WARNING);assert.match(view.warning,/before any ownership- or head-dependent action/);
  assert.deepEqual(view.state,[{field:'classification',text:'PROVED_REVIEWED'},{field:'controlling',text:'false'}]);
  assert.equal(view.review.label,m.RESUME_REVIEW_LABEL);assert.match(view.review.label,/not the current task, an owner claim or a completion record/);
  assert.deepEqual(view.review.fields.map(row=>row.field),['review_disposition','review_issue','review_issue_state','review_source']);
  assert.ok(unrecorded(view));assert.equal(view.continuation,null);assert.equal(view.continuation_text,m.RESUME_NO_TASK);
});
test('closed issues and accepted dispositions do not invent completion, ownership or a task link',()=>{
  const view=m.resumeProjection(closed(),'done');const issue=view.review.fields.find(row=>row.field==='review_issue');
  assert.deepEqual(issue,{field:'review_issue',text:'67',url:null,note:m.RESUME_ISSUE_NOTE});
  assert.equal(view.review.fields.find(row=>row.field==='review_issue_state').url,null);
  assert.ok(unrecorded(view));assert.equal(view.continuation,null);assert.match(view.continuation_text,/review issue is not used as a task record/);
  assert.doesNotMatch(JSON.stringify(view),/"(?:complete|completed|owner|assignee|done)"\s*:/);
});
test('lookalike operational metadata in the graph is never promoted to next action, owner or stop',()=>{
  const view=m.resumeProjection(closed(),'lookalike');assert.ok(unrecorded(view));assert.deepEqual(view.review.fields,[]);
  assert.equal(view.continuation,null);assert.doesNotMatch(JSON.stringify({operational:view.operational,continuation:view.continuation}),/merge now|Claude S|complete|pull\/1/);
});
test('missing fields stay explicitly not recorded',()=>{
  const view=m.resumeProjection(closed(),'bare');
  assert.deepEqual(view.state,[{field:'classification',text:m.RESUME_NOT_RECORDED},{field:'controlling',text:m.RESUME_NOT_RECORDED}]);
  assert.deepEqual(view.review.fields,[]);assert.ok(unrecorded(view));
  const nulls=snapshot({n:{kind:'candidate',classification:null,controlling:null,review_disposition:null}});const row=m.resumeProjection(nulls,'n');
  assert.equal(row.state[0].text,m.RESUME_NOT_RECORDED);assert.deepEqual(row.review.fields,[{field:'review_disposition',text:'null',url:null,note:null}]);
  const noMain=closed();noMain.repositories=noMain.repositories.filter(r=>!r.repository.endsWith('/main'));assert.equal(m.resumeProjection(noMain,'done').snapshot.main_commit,null);
});
test('stale or disagreeing clocks change only the freshness state and keep every warning',()=>{
  const s=closed(),observed=Date.parse(s.observed_at),base=m.resumeProjection(s,pin);
  for(const [offset,state] of [[3600000,'current'],[3*3600000,'stale'],[-3600000,'clock disagreement']]){
    const view=m.resumeProjection(s,pin,observed+offset);assert.equal(view.freshness,state);assert.deepEqual({...view,freshness:null},base);
    assert.equal(view.warning,m.RESUME_FRESHNESS_WARNING);assert.ok(unrecorded(view));assert.match(view.continuation.meaning,/not scientific acceptance, source-binding admission, a registration or an owner claim/);
  }
  for(const now of [undefined,NaN,'2026-10-13T13:00:00Z'])assert.equal(m.resumeProjection(s,pin,now).freshness,null);
});
test('shared source hashes do not transfer review fields or the explicit task pointer',()=>{
  const s=closed();assert.deepEqual(m.resumeProjection(s,'twin').review.fields,[]);assert.equal(m.resumeProjection(s,'twin').continuation,null);
  const task=m.resumeProjection(s,pin).continuation;assert.deepEqual(task,{...m.RESUME_TASK_RECORDS[pin]});assert.deepEqual(m.resumeProjection(s,pin).review.fields,[]);
  assert.ok(unrecorded(m.resumeProjection(s,pin)));
});
test('only exact public review URLs link; issue numbers and other strings stay inert',()=>{
  const urls=['https://github.com/d6g8k5htny-coder/main/issues/67','https://github.com/d6g8k5htny-coder/Math-/pull/111#issuecomment-1'];
  const refused=['http://github.com/d6g8k5htny-coder/main/issues/67','https://github.com/other/main/issues/67','https://github.com/d6g8k5htny-coder/main/issues/67?x=1','https://github.com/d6g8k5htny-coder/main/issues/067','javascript:alert(1)','https://github.com/d6g8k5htny-coder/main/issues/67\n','reviews/x/REVIEW.md','https://github.com/d6g8k5htny-coder/main/issues/67#a b'];
  const nodes=Object.fromEntries([...urls,...refused].map((value,i)=>['n'+i,{kind:'candidate',review_source:value}]));const s=snapshot(nodes);
  urls.forEach((value,i)=>assert.equal(m.resumeProjection(s,'n'+i).review.fields[0].url,value));
  refused.forEach((value,i)=>assert.equal(m.resumeProjection(s,'n'+(i+urls.length)).review.fields[0].url,null));
  const issue=m.resumeProjection(snapshot({x:{kind:'candidate',review_issue:'https://github.com/d6g8k5htny-coder/main/issues/63'}}),'x').review.fields[0];assert.equal(issue.note,null);assert.equal(issue.url,'https://github.com/d6g8k5htny-coder/main/issues/63');
});
test('unknown and prototype identifiers have no resume projection',()=>{
  for(const id of ['missing','constructor','toString','__proto__',{toString:()=> 'done'}])assert.equal(m.resumeProjection(closed(),id),null);
  assert.equal(m.resumeProjection(null,'done'),null);
});
test('explicit task records are frozen operational pointers to public records',()=>{
  assert.ok(Object.isFrozen(m.RESUME_TASK_RECORDS)&&Object.isFrozen(m.RESUME_OPERATIONAL));
  for(const [id,record] of Object.entries(m.RESUME_TASK_RECORDS)){assert.ok(Object.isFrozen(record));assert.match(id,/^[A-Za-z0-9][A-Za-z0-9_.-]*$/);
    assert.deepEqual(Object.keys(record),['url','label','meaning','identified_on','identified_in']);
    assert.match(record.url,/^https:\/\/github\.com\/d6g8k5htny-coder\/(?:main|Math-)\/(?:issues|pull)\/[1-9][0-9]*#issuecomment-[1-9][0-9]*$/);
    assert.match(record.identified_in,/^https:\/\/github\.com\/d6g8k5htny-coder\/main\/(?:issues|pull)\/[1-9][0-9]*#issuecomment-[1-9][0-9]*$/);
    assert.match(record.label,/^Operational (?:request|continuation): /);assert.match(record.meaning,/operational pointer, not scientific acceptance/);assert.match(record.identified_on,/^\d{4}-\d{2}-\d{2}$/);}
  const view=m.resumeProjection(closed(),pin);view.continuation.url='x';assert.notEqual(m.RESUME_TASK_RECORDS[pin].url,'x');
});
test('static node pages render the identical projection for every real node and the fixtures',async()=>{
  const real=JSON.parse(await readFile(new URL('../docs/site/current.json',import.meta.url),'utf8'));
  const cases=[[real,Object.keys(real.graph.data.nodes)],[closed(),['done','lookalike','bare','twin',pin,'missing']]];
  const tools=fileURLToPath(new URL('../tools/',import.meta.url));
  const script='import json,sys\nsys.path.insert(0,sys.argv[1])\nfrom current_site_history import resume_projection\nprint(json.dumps([[resume_projection(data,i) for i in ids] for data,ids in json.load(sys.stdin)]))';
  const python=JSON.parse(execFileSync('python3',['-B','-S','-c',script,tools],{input:JSON.stringify(cases),encoding:'utf8'}));
  assert.deepEqual(python,cases.map(([data,ids])=>ids.map(id=>m.resumeProjection(data,id))));
  assert.equal(python[0].filter(view=>view.continuation).length,Object.keys(m.RESUME_TASK_RECORDS).filter(id=>m.hasNode(real.graph.data,id)).length);
});
