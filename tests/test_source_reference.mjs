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

const refs = await import('../docs/site/source-reference.mjs');
const digest = 'AB'.repeat(32);
test('optional file digest is preserved as supplied metadata with explicit unverified JSON',()=>{
  const ref=buildReference('Math-',sha,'proofs/α #1.md',digest);
  assert.match(ref.text,new RegExp(digest.toLowerCase()));
  const record=JSON.parse(ref.json);
  assert.equal(record.repository,'d6g8k5htny-coder/Math-');
  assert.equal(record.commit,sha);assert.equal(record.path,'proofs/α #1.md');
  assert.equal(record.sha256,digest.toLowerCase());assert.equal(record.verification,'not_performed');
  assert.equal(record.url,ref.url);assert.equal(record.authors,undefined);
  assert.equal(JSON.parse(buildReference('main',sha).json).sha256,null);
});
test('rejects misleading or malformed digest identities',()=>{
  for(const value of [digest+'0',' main ', '0'.repeat(63), 'g'.repeat(64),digest+'\n',42]) {
    assert.throws(()=>buildReference('main',sha,'README.md',value));
  }
  assert.throws(()=>buildReference('main',sha,'',digest),/file path/i);
});
test('reference links roundtrip exact unicode identity and preserve unrelated URL fields',()=>{
  assert.equal(typeof refs.referenceStateURL,'function');
  assert.equal(typeof refs.readReferenceState,'function');
  const state={repository:'Math-',commit:sha,path:'proofs/α #1.md',sha256:digest};
  const url=new URL(refs.referenceStateURL('https://example.test/cite.html?lang=en#reference-builder',state));
  assert.equal(url.searchParams.get('lang'),'en');assert.equal(url.hash,'#reference-builder');
  assert.deepEqual(refs.readReferenceState(url.search),{...state,sha256:digest.toLowerCase()});
  assert.equal(refs.readReferenceState('?lang=en'),null);
});
test('malformed shared references are refused without partial source identity',()=>{
  assert.equal(typeof refs.readReferenceState,'function');
  for(const query of [`?repo=main&commit=${sha}&commit=${sha}`,`?repo=main&commit=main`,`?commit=${sha}`,`?repo=evil&commit=${sha}`,`?repo=main&commit=${sha}&path=%252e%252e/x`,`?repo=main&commit=${sha}&sha256=${digest}`]) {
    assert.throws(()=>refs.readReferenceState(query),query);
  }
});
function referenceControls(clipboard) {
  const node=(value='')=>({value,textContent:'',hidden:true,disabled:true,listeners:{},addEventListener(k,v){this.listeners[k]=v;},removeAttribute(k){delete this[k];}});
  const c={form:node(),repository:node('main'),commit:node(sha),path:node('README.md'),sha256:node(digest),output:node(),json:node(),link:node(),status:node(),actions:node(),copyText:node(),copyJSON:node(),share:node(),clipboard,
    location:{href:'https://example.test/cite.html#reference-builder',search:''},history:{pushState(_a,_b,url){c.location.href=url;c.location.search=new URL(url).search;}},events:node()};
  c.submit=()=>c.form.listeners.submit({preventDefault(){}});return c;
}
test('reference workflow copies selected representations and invalidates stale outputs',async()=>{
  let copied;const c=referenceControls({async writeText(text){copied=text;}});
  wireReferenceForm(c);c.submit();assert.equal(c.actions.hidden,false);
  await c.copyText.listeners.click();assert.equal(copied,c.output.textContent);
  await c.copyJSON.listeners.click();assert.equal(copied,c.json.textContent);
  assert.match(c.status.textContent,/not verified/i);assert.equal(c.share.hidden,false);
  c.path.value='other.md';c.form.listeners.input();assert.equal(c.actions.hidden,true);
  assert.equal(c.copyJSON.disabled,true);assert.equal(c.json.textContent,'');assert.equal(c.share.href,undefined);
});
test('shared reference restoration and Back reject invalid state without publishing old output',()=>{
  const c=referenceControls();c.location.search=`?repo=Math-&commit=${sha}&path=PROOF.md`;
  c.location.href='https://example.test/cite.html'+c.location.search;
  wireReferenceForm(c);assert.equal(c.repository.value,'Math-');assert.equal(c.path.value,'PROOF.md');
  assert.equal(c.copyText.disabled,true);assert.match(c.status.textContent,/manually/i);
  c.location.search='?repo=main&commit=main';c.events.listeners.popstate();
  assert.equal(c.output.textContent,'');assert.equal(c.actions.hidden,true);assert.match(c.status.textContent,/40/);
  c.location.search='';c.events.listeners.popstate();assert.equal(c.commit.value,'');
});
test('clipboard rejection and delayed completion do not claim a stale copy succeeded',async()=>{
  const c=referenceControls({async writeText(){throw Error('denied');}});
  wireReferenceForm(c);c.submit();await c.copyText.listeners.click();assert.match(c.status.textContent,/manually/i);assert.doesNotMatch(c.status.textContent,/Copied/);
  let finish;const d=referenceControls({writeText(){return new Promise(resolve=>{finish=resolve;});}});
  wireReferenceForm(d);d.submit();const pending=d.copyJSON.listeners.click();
  d.path.value='new.md';d.form.listeners.input();finish();await pending;
  assert.equal(d.actions.hidden,true);assert.equal(d.copyJSON.disabled,true);assert.doesNotMatch(d.status.textContent,/Copied/);
});
