import test from 'node:test';
import assert from 'node:assert/strict';
import {buildReference, wireReferenceForm} from '../docs/site/source-reference.mjs';
const sha = '354b697669b59d43563f1b972203fa9ea6d3e6b7';
test('builds an immutable repository or file reference without attributing authorship', () => {
  assert.equal(buildReference('main',sha,'').url,`https://github.com/d6g8k5htny-coder/main/tree/${sha}`);
  const ref=buildReference('Math-',sha.toUpperCase(),'proofs/a space #1.md');
  assert.equal(ref.url,`https://github.com/d6g8k5htny-coder/Math-/blob/${sha}/proofs/a%20space%20%231.md`);
  assert.ok(ref.text.includes(sha)); assert.ok(ref.text.includes('proofs/a space #1.md'));
  assert.ok(!ref.text.includes('Roy, Dylan'));
});
test('rejects mutable refs, unsupported repositories and ambiguous or unsafe paths', () => {
  for(const commit of ['', 'main', sha.slice(0,7), sha+'0', ' '+sha, '\n'+sha, sha+'\n', sha+'\r']) assert.throws(()=>buildReference('main',commit,''));
  for(const repo of ['sandbox','trial','https://evil.test','../main','Main']) assert.throws(()=>buildReference(repo,sha,''));
  for(const path of ['../README.md','/README.md','a/../b','a/./b','a//b','a/','a\\b','%2e%2e/README.md','a%2Fb','a\u0000b','a\nb','a\u202Eb','a'.repeat(501)]) assert.throws(()=>buildReference('main',sha,path),path);
});
test('allows unicode file names and unambiguous encoded URL delimiters', () => {
  const ref=buildReference('query-',sha,'docs/α?β.md');
  assert.equal(ref.url,`https://github.com/d6g8k5htny-coder/query-/blob/${sha}/docs/%CE%B1%3F%CE%B2.md`);
});
test('real form wiring clears stale output and announces a corrected reference', () => {
  const listeners={};const e={
    form:{addEventListener:(event,fn)=>listeners[event]=fn},
    repository:{value:'main'},commit:{value:sha},path:{value:''},
    output:{textContent:''},link:{hidden:true,href:'',textContent:'',removeAttribute(key){delete this[key];}},
    status:{textContent:''}
  };
  wireReferenceForm(e);listeners.submit({preventDefault(){}});
  assert.equal(e.link.hidden,false);assert.ok(e.output.textContent.includes(sha));
  e.commit.value='main';listeners.input();
  assert.equal(e.output.textContent,'');assert.equal(e.link.hidden,true);assert.equal(e.link.href,undefined);
  listeners.submit({preventDefault(){}});assert.match(e.status.textContent,/40/);
  e.commit.value=sha;listeners.submit({preventDefault(){}});assert.equal(e.link.hidden,false);
  assert.match(e.status.textContent,/not checked/i);
});
