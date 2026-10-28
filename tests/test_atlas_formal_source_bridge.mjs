import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {formalConnection} from '../docs/site/atlas-formal-evidence-model.mjs';
import {EVIDENCE} from '../docs/site/atlas-formal-evidence-data.mjs';

const id='math.side24-coefficient';
const fixture=()=>{
  const data=JSON.parse(fs.readFileSync(new URL('./fixtures/atlas_formal_projection.json',import.meta.url)));
  const manifest=data.sources.find(s=>s.repository==='d6g8k5htny-coder/main'&&s.path==='formal/manifest.json');
  Object.assign(manifest,{commit:'93869215a82da60f014093e2ac28503865ed0aa3',
    blob:'8228e896c728d95ab94391899320e9d44ab7cdfc',bytes:17521});
  return data;
};

test('Explicit current-manifest bridge exposes the historical artifact without rebinding its execution or review',()=>{
  const data=fixture(),connection=formalConnection(data,id);
  assert.ok(connection,'Current source must have an explicitly verified historical-artifact connection');
  assert.equal(connection.source_bridge.projection_manifest.blob,'8228e896c728d95ab94391899320e9d44ab7cdfc');
  assert.equal(connection.formal.commit,'3b39c42d6edcc98ae165270be346f0e31c57f81d');
  assert.equal(connection.execution.run_id,'40246431034');
  assert.equal(connection.alignment.record.disposition,'REJECTED');
  assert.equal(connection.alignment.record.manifest_sha256,'8dfcfe4e9296bca47eede517f37b546ea420ec6b51944b7c16164a906b40e090');
  assert.equal(connection.current_source.commit,data.graph.identity.commit);
});

for(const [name,mutate] of [
  ['an unknown manifest',d=>d.sources.find(s=>s.repository.endsWith('/main')&&s.path==='formal/manifest.json').blob='f'.repeat(40)],
  ['a wrong manifest size',d=>d.sources.find(s=>s.repository.endsWith('/main')&&s.path==='formal/manifest.json').bytes++],
  ['duplicate formal inventory',d=>d.sources.push({...d.sources.find(s=>s.repository.endsWith('/main')&&s.path==='formal/manifest.json')})],
  ['changed claim scope',d=>d.graph.data.nodes[id].scope='The full coefficient theorem'],
  ['an extra claim coverage field',d=>d.graph.data.nodes[id].kernel_coverage=true],
])test(name+' cannot use the explicit bridge',()=>{
  const data=fixture();mutate(data);assert.equal(formalConnection(data,id),null);
});

for(const [name,mutate] of [
  ['another declaration',e=>e.declaration.name='UniversalLaw.Side24.entire_coefficient'],
  ['a forged receipt with the same commit',e=>e.execution.receipt_sha256='0'.repeat(64)],
  ['a fabricated accepted review',e=>e.alignment.selected_target_verdict='ACCEPTED'],
])test(name+' cannot replace the pinned evidence',()=>{
  const evidence=structuredClone(EVIDENCE);mutate(evidence);
  assert.equal(formalConnection(fixture(),id,evidence),null);
});

test('An equally named different claim and prototype key remain unmapped',()=>{
  const data=fixture();data.graph.data.nodes['math.lookalike']=structuredClone(data.graph.data.nodes[id]);
  assert.equal(formalConnection(data,'math.lookalike'),null);
  assert.equal(formalConnection(data,'__proto__'),null);
});

test('The separately pinned corrected manifest links only to historical execution, never to a transferred alignment',()=>{
  const data=fixture();
  Object.assign(data.sources.find(s=>s.repository.endsWith('/main')&&s.path==='formal/manifest.json'),
    {commit:'55961773d31a7b2b9906d8be73a26ad44c364dd6',blob:'a22f06a3a7b9ab451e9e344d7b2345d2e2ab0553',bytes:18104});
  const connection=formalConnection(data,id);
  assert.ok(connection,'The corrected current manifest needs its own explicit source bridge');
  assert.equal(connection.source_bridge.identical_formal_file_count,13);
  assert.equal(connection.source_bridge.changed_documentation_file_count,2);
  assert.equal(connection.source_bridge.changed_target_annotation_count,2);
  assert.equal(connection.execution.checked_commit,'3b39c42d6edcc98ae165270be346f0e31c57f81d');
  assert.equal(connection.alignment.record.disposition,'REJECTED');
  assert.equal(connection.alignment.record.manifest_sha256,'8dfcfe4e9296bca47eede517f37b546ea420ec6b51944b7c16164a906b40e090');
});

for(const [profile,blob,bytes] of [
  ['original',EVIDENCE.formal.manifest_blob,EVIDENCE.formal.manifest_bytes],
  ['8228','8228e896c728d95ab94391899320e9d44ab7cdfc',17521],
  ['a22f','a22f06a3a7b9ab451e9e344d7b2345d2e2ab0553',18104],
])for(const conflicting of [false,true])test(profile+' refuses '+(conflicting?'conflicting':'identical')+' duplicate proof-source entries',()=>{
  const data=fixture();
  Object.assign(data.sources.find(s=>s.repository===EVIDENCE.formal.repository&&s.path===EVIDENCE.formal.manifest_path),{blob,bytes});
  assert.ok(formalConnection(data,id),'The unambiguous positive control must map');
  const proof=data.sources.find(s=>s.repository===EVIDENCE.document.repository&&s.path===EVIDENCE.document.path);
  data.sources.push({...proof,...(conflicting?{blob:'f'.repeat(40)}:{})});
  assert.equal(formalConnection(data,id),null);
});
