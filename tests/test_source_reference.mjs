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

import {readFile} from 'node:fs/promises';

const copyModule = await import('../docs/site/reproduce-copy.mjs').catch(()=>({}));
const copyCommands = 'git checkout --detach f430fdecbb1d8802d8af40419b701f12af87c039\npython3 -B -S coefficients/side24_v1/coefficient.py';
function copyControls() {
  let click;
  return {block:{textContent:copyCommands},status:{textContent:''},button:{disabled:true,addEventListener(name,handler){assert.equal(name,'click');click=handler;}},click:()=>click()};
}
test('copy sends exactly the displayed pinned commands and never executes them',async()=>{
  assert.equal(typeof copyModule.wireCommandCopy,'function');
  const c=copyControls();let copied;
  copyModule.wireCommandCopy({...c,clipboard:{async writeText(text){copied=text;}}});
  assert.equal(c.button.disabled,false);await c.click();assert.equal(copied,copyCommands);
  assert.match(c.status.textContent,/Copied/);assert.match(c.status.textContent,/not run/);
});
test('clipboard refusal provides manual fallback without success',async()=>{
  assert.equal(typeof copyModule.wireCommandCopy,'function');
  const c=copyControls();copyModule.wireCommandCopy({...c,clipboard:{async writeText(){throw Error('denied');}}});
  await c.click();assert.match(c.status.textContent,/Select/);assert.doesNotMatch(c.status.textContent,/Copied/);assert.equal(c.button.disabled,false);
});
test('unsupported clipboard leaves the command block readable and gives fallback',()=>{
  assert.equal(typeof copyModule.wireCommandCopy,'function');
  const c=copyControls();copyModule.wireCommandCopy({...c,clipboard:undefined});
  assert.equal(c.button.disabled,true);assert.equal(c.block.textContent,copyCommands);assert.match(c.status.textContent,/Select/);
});
test('reproduction controls have static fallback and an explicit live announcement',async()=>{
 const html=await readFile(new URL('../docs/site/reproduce.html',import.meta.url),'utf8');
 assert.match(html,/id="reproduction-commands"/);assert.match(html,/id="copy-commands"[^>]*disabled/);
 assert.match(html,/id="copy-commands-status"[^>]*role="status"/);
 assert.match(html,/script-src 'self'/);assert.match(html,/src="reproduce-copy.mjs/);
});
