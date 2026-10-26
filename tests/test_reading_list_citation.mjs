import test from 'node:test';
import assert from 'node:assert/strict';
import {readReferenceState,buildReference} from '../docs/site/source-reference.mjs';

let helper={};
try {helper=await import('../docs/site/reading-list-citation.mjs');}
catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;}
const {readingListCitationHref}=helper;
const commit='0123456789abcdef0123456789abcdef01234567';
const source=(extra={})=>({kind:'source',collection:'current',repository:'d6g8k5htny-coder/Math-',commit,path:'proofs/PROOF.md',savedAt:'2026-10-26T00:00:00Z',...extra});

test('saved source hands exact fields to the same-site reference builder',()=>{
  assert.equal(typeof readingListCitationHref,'function');
  const href=readingListCitationHref(source({sha256:'b'.repeat(64),blob:'c'.repeat(40),bytes:42}),'https://example.test/main/site/workspace.html?source-q=old&repo=evil#reading-list');
  assert.equal(href,'https://example.test/main/site/cite.html?repo=Math-&commit=0123456789abcdef0123456789abcdef01234567&path=proofs%2FPROOF.md&sha256='+('b'.repeat(64))+'#reference-builder');
  const state=readReferenceState(new URL(href).search);
  assert.deepEqual(state,{repository:'Math-',commit,path:'proofs/PROOF.md',sha256:'b'.repeat(64)});
  const json=JSON.parse(buildReference(state.repository,state.commit,state.path,state.sha256).json);
  assert.deepEqual(json,{schema:'universal-law/source-reference/v1',repository:'d6g8k5htny-coder/Math-',commit,path:'proofs/PROOF.md',sha256:'b'.repeat(64),url:'https://github.com/d6g8k5htny-coder/Math-/blob/'+commit+'/proofs/PROOF.md',verification:'not_performed'});
  assert.ok(!href.includes('blob=')&&!href.includes('bytes=')&&!href.includes('savedAt='));
});

test('Git blob alone never becomes a file SHA-256 or inferred author/review field',()=>{
  assert.equal(typeof readingListCitationHref,'function');
  for(const extra of [{},{blob:'a'.repeat(40)}]) {
    const href=readingListCitationHref(source({...extra,repository:'d6g8k5htny-coder/main'}),'https://example.test/site/');
    const state=readReferenceState(new URL(href).search);
    assert.deepEqual(state,{repository:'main',commit,path:'proofs/PROOF.md',sha256:''});
    const data=JSON.parse(buildReference(state.repository,state.commit,state.path,state.sha256).json);
    assert.equal(data.sha256,null);assert.equal(data.verification,'not_performed');
    assert.equal(Object.hasOwn(data,'author'),false);assert.equal(Object.hasOwn(data,'review'),false);
  }
});

test('valid punctuation and Unicode roundtrip without changing the literal Git path',()=>{
  assert.equal(typeof readingListCitationHref,'function');
  const path="proofs/Δ + a&b#c?d{}!'().md",entry=source({path}),before=JSON.stringify(entry);
  const href=readingListCitationHref(entry,'https://example.test/prefix/site/workspace.html?junk=1#old');
  const url=new URL(href);assert.equal(url.pathname,'/prefix/site/cite.html');assert.equal(url.hash,'#reference-builder');
  assert.deepEqual([...url.searchParams.keys()],['repo','commit','path']);
  assert.equal(url.searchParams.get('path'),path);assert.equal(readReferenceState(url.search).path,path);
  assert.equal(JSON.stringify(entry),before);
});

test('valid reading-list paths outside the Cite contract return explicit unavailable signal',()=>{
  assert.equal(typeof readingListCitationHref,'function');
  for(const path of ['proofs/literal%20.md','proofs/%2e%2e.md','proofs/\u202eproof.md','proofs/\u2066proof.md','a'.repeat(501)]) {
    const entry=source({path}),before=JSON.stringify(entry);
    assert.equal(readingListCitationHref(entry,'https://example.test/site/workspace.html'),null);
    assert.equal(JSON.stringify(entry),before);
  }
  assert.ok(readingListCitationHref(source({path:'a'.repeat(500)}),'https://example.test/site/workspace.html'));
});

test('queries and malformed or overprivileged imported records cannot become references',()=>{
  assert.equal(typeof readingListCitationHref,'function');
  const query={kind:'search',collection:'current',query:'PROOF.md',repository:'',path:'',savedAt:'2026-10-26T00:00:00Z'};
  for(const entry of [query,null,{},source({repository:'other/Math-'}),source({commit:'main'}),source({path:'../PROOF.md'}),source({path:'a\ud800'}),source({sha256:'a'.repeat(40)}),source({url:'https://evil.test'}),source({certified:true}),source({savedAt:'2026-02-30T00:00:00Z'})])assert.throws(()=>readingListCitationHref(entry,'https://example.test/site/workspace.html'),{name:'TypeError'});
  assert.throws(()=>readingListCitationHref(source(),'not a URL'),{name:'TypeError'});
});
