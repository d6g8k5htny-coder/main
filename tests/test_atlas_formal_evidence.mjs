import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {formalConnection} from '../docs/site/atlas-formal-evidence-model.mjs';
import {EVIDENCE} from '../docs/site/atlas-formal-evidence-data.mjs';

const fixture=()=>JSON.parse(fs.readFileSync(new URL('./fixtures/atlas_formal_projection.json',import.meta.url)));
const id='math.side24-coefficient';
test('Exact admitted claim/source identity exposes only the pinned arithmetic companion',()=>{
  const data=fixture(),result=formalConnection(data,id);
  assert.ok(result,'The verified source connection is absent');
  assert.equal(result.declaration.name,'UniversalLaw.Side24.pairing_sum_six');
  assert.equal(result.current_source.commit,data.graph.identity.commit);
  assert.equal(result.execution.checked_commit,result.formal.commit);
  assert.equal(result.alignment.record.disposition,'REJECTED');
});
test('Source-identical documents at another recorded revision retain their actual current pin',()=>{
  const data=fixture(),pin='a'.repeat(40);
  data.graph.identity.commit=pin;
  data.sources.filter(s=>s.repository.endsWith('Math-')).forEach(s=>s.commit=pin);
  assert.equal(formalConnection(data,id)?.current_source.commit,pin);
});
for(const [name,mutate] of [
  ['changed scope',d=>d.graph.data.nodes[id].scope='The full coefficient theorem'],
  ['extra graph field',d=>d.graph.data.nodes[id].formal_proof=true],
  ['changed source blob',d=>d.sources.find(s=>s.path===EVIDENCE.document.path).blob='0'.repeat(40)],
  ['changed source size',d=>d.sources.find(s=>s.path===EVIDENCE.document.path).bytes++],
  ['changed fingerprint',d=>d.graph.data.nodes[id].fingerprint='0'.repeat(64)],
  ['missing source',d=>d.sources=d.sources.filter(s=>s.path!==EVIDENCE.document.path)],
  ['changed formal manifest',d=>d.sources.find(s=>s.repository.endsWith('/main')&&s.path==='formal/manifest.json').blob='0'.repeat(40)],
]){
  test(name+' leaves the mapping unavailable',()=>{const data=fixture();mutate(data);assert.equal(formalConnection(data,id),null);});
}
test('An equal-looking different node ID is never inferred to be covered',()=>{
  const data=fixture();data.graph.data.nodes['math.lookalike']=structuredClone(data.graph.data.nodes[id]);
  assert.equal(formalConnection(data,'math.lookalike'),null);
});
test('Other actual claims keep missing mappings explicit',()=>{
  assert.equal(formalConnection(fixture(),'math.uniform-matrix-cap-lifetime'),null);
});
test('The actual older committed observation has a different manifest and remains unmapped',()=>{
  const old=JSON.parse(fs.readFileSync(new URL('../docs/site/current.json',import.meta.url)));
  assert.equal(formalConnection(old,id),null);
});
test('An execution for another revision cannot attach to the artifact',()=>{
  const e=structuredClone(EVIDENCE);e.execution.checked_commit='0'.repeat(40);
  assert.equal(formalConnection(fixture(),id,e),null);
});
test('A fabricated accepted alignment cannot replace the pinned adverse record',()=>{
  const e=structuredClone(EVIDENCE);e.alignment.record.disposition='ACCEPTED';
  assert.equal(formalConnection(fixture(),id,e),null);
});
