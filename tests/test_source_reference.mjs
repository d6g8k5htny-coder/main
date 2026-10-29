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
const intervals=[{dimension:'2',lower:'0.07340691930603427103',upper:'0.07340691930603427104'},{dimension:'3',lower:'0.04177593184059834334',upper:'0.04177593184059834335'}];
const coefficientSource='https://github.com/d6g8k5htny-coder/Math-/blob/f430fdecbb1d8802d8af40419b701f12af87c039/coefficients/side24_v1/coefficient.py';
test('coefficient exports preserve exact narrow decimal enclosures and pinned identity',()=>{
 assert.equal(typeof copyModule.buildCoefficientExports,'function');
 const out=copyModule.buildCoefficientExports(intervals,coefficientSource);
 const data=JSON.parse(out.json);
 assert.deepEqual(data.intervals,intervals);assert.equal(data.source,coefficientSource);
 assert.equal(data.execution,'not_performed');assert.equal(data.source_state,'historical_pinned');
 for(const row of intervals){assert.ok(out.text.includes(row.lower));assert.ok(out.text.includes(row.upper));assert.ok(out.latex.includes(row.lower));assert.ok(out.latex.includes(row.upper));}
 assert.ok(out.text.includes('Historical'));
});
test('coefficient exports retain strict SIDE24 bounds and paste-ready display math',()=>{
 const out=copyModule.buildCoefficientExports(intervals,coefficientSource);
 for(const row of intervals){
  assert.ok(out.text.includes(`${row.lower} < c_${row.dimension},24 < ${row.upper}`));
  assert.ok(out.latex.includes(`\\[${row.lower} < c_{${row.dimension},24} < ${row.upper}\\]`));
 }
 assert.equal(JSON.parse(out.json).bounds,'strict');
 assert.equal(JSON.parse(out.json).coefficient,'c_{d,24}');
 assert.doesNotMatch(out.latex,/\\le/);assert.doesNotMatch(out.text,/<=/);
 assert.throws(()=>copyModule.buildCoefficientExports([{dimension:'2',lower:'0.1',upper:'0.10'}],coefficientSource));
});
test('coefficient exports refuse rounded numbers, reversed enclosures and mutable sources',()=>{
 assert.equal(typeof copyModule.buildCoefficientExports,'function');
 for(const rows of [[{...intervals[0],lower:Number(intervals[0].lower)}],[{...intervals[0],lower:intervals[0].upper,upper:intervals[0].lower}],[{...intervals[0],dimension:'2.5'}],[]])assert.throws(()=>copyModule.buildCoefficientExports(rows,coefficientSource));
 assert.throws(()=>copyModule.buildCoefficientExports(intervals,coefficientSource.replace('f430fdecbb1d8802d8af40419b701f12af87c039','main')));
});
test('interval copying locks all formats until the actual write finishes and handles denial',async()=>{
 assert.equal(typeof copyModule.wireCoefficientCopy,'function');
 const handlers={},buttons={};for(const format of ['text','json','latex'])buttons[format]={disabled:true,addEventListener(_,fn){handlers[format]=fn;}};
 const status={textContent:''};let resolve,writes=[];
 copyModule.wireCoefficientCopy({buttons,status,exports:copyModule.buildCoefficientExports(intervals,coefficientSource),clipboard:{writeText(text){writes.push(text);return new Promise(r=>resolve=r);}}});
 const first=handlers.text();assert.ok(Object.values(buttons).every(b=>b.disabled));await handlers.json();assert.equal(writes.length,1);resolve();await first;
 assert.ok(Object.values(buttons).every(b=>!b.disabled));assert.match(status.textContent,/historical|Historical/);
 copyModule.wireCoefficientCopy({buttons,status,exports:copyModule.buildCoefficientExports(intervals,coefficientSource),clipboard:{async writeText(){throw Error('denied');}}});await handlers.json();assert.match(status.textContent,/manually/);assert.doesNotMatch(status.textContent,/Copied/);
});
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
  const c={form:node(),repository:node('main'),commit:node(sha),path:node('README.md'),sha256:node(digest),output:node(),json:node(),bibtex:node(),link:node(),status:node(),actions:node(),copyText:node(),copyJSON:node(),copyBibTeX:node(),share:node(),clipboard,
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

test('rejects malformed percent and UTF-8 links rather than substituting source path bytes',()=>{
  for(const path of ['%FF','%C0%AF','%ED%A0%80','bad%','bad%2','%GG']) {
    assert.throws(()=>refs.readReferenceState(`?repo=main&commit=${sha}&path=${path}`),path);
  }
});
test('a pending clipboard operation prevents a newer reference from overtaking it',async()=>{
  const writes=[];let resolveA;
  const c=referenceControls({writeText(text){writes.push(text);return writes.length===1?new Promise(r=>{resolveA=r;}):Promise.resolve();}});
  wireReferenceForm(c);c.submit();const a=c.copyText.listeners.click();
  c.path.value='B.md';c.form.listeners.input();c.submit();
  assert.equal(c.copyJSON.disabled,true);
  await c.copyJSON.listeners.click();assert.equal(writes.length,1);
  resolveA();await a;assert.equal(c.copyJSON.disabled,false);
  assert.doesNotMatch(c.status.textContent,/Copied JSON/);
  await c.copyJSON.listeners.click();assert.equal(writes.length,2);
  assert.equal(JSON.parse(writes[1]).path,'B.md');assert.match(c.status.textContent,/Copied JSON/);
});

// Missing or unescaped source fields must fail these template contracts.
test('BibTeX is an honest source template with exact immutable identity and optional supplied digest',()=>{
 const ref=buildReference('Math-',sha,'proofs/α #1.md',digest);
 assert.match(ref.bibtex,/^@misc\{source_reference_template,/);
 assert.match(ref.bibtex,/title = \{Repository source reference\}/);
 assert.ok(ref.bibtex.includes(`Git commit ${sha}`));
 assert.ok(ref.bibtex.includes(`Supplied SHA-256 (not checked): ${digest.toLowerCase()}`));
 assert.ok(ref.bibtex.includes(`url = {https://github.com/d6g8k5htny-coder/Math-/blob/${sha}/proofs/%CE%B1%20%231.md}`));
 assert.match(ref.bibtex,/not verified/i);
 assert.doesNotMatch(ref.bibtex,/\b(author|year|doi|journal|publisher)\s*=/i);
 const snapshot=buildReference('main',sha).bibtex;
 assert.ok(snapshot.includes('d6g8k5htny-coder/main'));
 assert.doesNotMatch(snapshot,/File path:|SHA-256/);
});
test('BibTeX escapes accepted path specials as literal text without changing Unicode or JSON identity',()=>{
 const path='proofs/α_#${x}&~^.md';
 const ref=buildReference('main',sha,path);
 assert.ok(ref.bibtex.includes(String.raw`File path: proofs/α\_\#\$\textbraceleft{}x\textbraceright{}\&\textasciitilde{}\textasciicircum{}.md`));
 assert.equal(JSON.parse(ref.json).path,path);
 assert.equal(ref.url,`https://github.com/d6g8k5htny-coder/main/blob/${sha}/proofs/%CE%B1_%23%24%7Bx%7D%26~%5E.md`);
 assert.ok(ref.bibtex.includes(`url = {${ref.url}}`));
 for(const bad of ['proofs/\\input{evil}.md','proofs/%5cinput.md','proofs/\n@book{x}.md','proofs/\ud800.md']) assert.throws(()=>buildReference('main',sha,bad));
});
// Omitted BibTeX render/clear/copy handling or a format-specific lock breaks these workflows.
test('BibTeX renders, copies, clears on edits and restores with Back/Forward source identity',async()=>{
 let copied;const c=referenceControls({async writeText(value){copied=value;}});
 wireReferenceForm(c);c.submit();
 assert.match(c.bibtex.textContent,/^@misc/);assert.equal(c.copyBibTeX.disabled,false);
 await c.copyBibTeX.listeners.click();assert.equal(copied,c.bibtex.textContent);assert.match(c.status.textContent,/Copied BibTeX/);
 const oldSearch=c.location.search;c.path.value='next_α.md';c.form.listeners.input();
 assert.equal(c.bibtex.textContent,'');assert.equal(c.copyBibTeX.disabled,true);
 c.submit();const nextSearch=c.location.search;
 c.location.search=oldSearch;c.events.listeners.popstate();assert.match(c.bibtex.textContent,/File path: README.md/);
 c.location.search=nextSearch;c.events.listeners.popstate();assert.ok(c.bibtex.textContent.includes(String.raw`next\_α.md`));
 c.location.search='?repo=main&commit=main';c.events.listeners.popstate();
 assert.equal(c.bibtex.textContent,'');assert.equal(c.copyBibTeX.disabled,true);
});
test('BibTeX shares the copy lock with every format and stale completion cannot announce success',async()=>{
 const writes=[];let finish;const c=referenceControls({writeText(value){writes.push(value);return new Promise(r=>finish=r);}});
 wireReferenceForm(c);c.submit();const pending=c.copyBibTeX.listeners.click();
 assert.ok([c.copyText,c.copyJSON,c.copyBibTeX].every(b=>b.disabled));
 await c.copyText.listeners.click();await c.copyJSON.listeners.click();assert.equal(writes.length,1);
 c.path.value='later.md';c.form.listeners.input();c.submit();
 await c.copyBibTeX.listeners.click();assert.equal(writes.length,1);
 finish();await pending;assert.doesNotMatch(c.status.textContent,/Copied/);
 assert.ok(c.bibtex.textContent.includes('later.md'));assert.ok([c.copyText,c.copyJSON,c.copyBibTeX].every(b=>!b.disabled));
 const textPending=c.copyText.listeners.click();await c.copyBibTeX.listeners.click();assert.equal(writes.length,2);
 c.form.listeners.change();finish();await textPending;
 assert.equal(c.bibtex.textContent,'');assert.equal(c.copyBibTeX.disabled,true);assert.doesNotMatch(c.status.textContent,/Copied/);
});
test('denied and unavailable clipboard preserve selectable BibTeX and explain manual fallback',async()=>{
 for(const clipboard of [undefined,{async writeText(){throw Error('denied');}}]){
  const c=referenceControls(clipboard);wireReferenceForm(c);c.submit();
  if(clipboard) await c.copyBibTeX.listeners.click();else assert.equal(c.copyBibTeX.disabled,true);
  assert.match(c.bibtex.textContent,/^@misc/);assert.match(c.status.textContent,/BibTeX.*manually/i);assert.doesNotMatch(c.status.textContent,/Copied/);
 }
});

// Escaping braces with a backslash still changes BibTeX's database brace depth.
test('legal unmatched and nested path braces produce parser-safe literal text',async t=>{
 const {spawnSync}=await import('node:child_process');
 const {mkdtemp,writeFile,rm}=await import('node:fs/promises');
 const {tmpdir}=await import('node:os');
 const {join}=await import('node:path');
 const cases=[
  ['proofs/{.md',String.raw`proofs/\textbraceleft{}.md`,'proofs/%7B.md'],
  ['proofs/}.md',String.raw`proofs/\textbraceright{}.md`,'proofs/%7D.md'],
  ['proofs/{{α}}.md',String.raw`proofs/\textbraceleft{}\textbraceleft{}α\textbraceright{}\textbraceright{}.md`,'proofs/%7B%7B%CE%B1%7D%7D.md'],
  ['proofs/}#$_&~^.md',String.raw`proofs/\textbraceright{}\#\$\_\&\textasciitilde{}\textasciicircum{}.md`,'proofs/%7D%23%24_%26~%5E.md']
 ];
 const probe=spawnSync('bibtex',['--version'],{encoding:'utf8'});
 const directory=await mkdtemp(join(tmpdir(),'source-bibtex-'));
 try {
  for(const [path,literal,encoded] of cases){
   const ref=buildReference('main',sha,path);
   assert.equal(JSON.parse(ref.json).path,path);
   assert.equal(ref.url,`https://github.com/d6g8k5htny-coder/main/blob/${sha}/${encoded}`);
   assert.ok(ref.bibtex.includes(`url = {${ref.url}}`));
   if(probe.error?.code==='ENOENT'){assert.ok(ref.bibtex.includes(`File path: ${literal}.`));continue;}
   await writeFile(join(directory,'reference.bib'),ref.bibtex);
   await writeFile(join(directory,'reference.aux'),'\\relax\n\\citation{source_reference_template}\n\\bibstyle{plain}\n\\bibdata{reference}\n');
   const parsed=spawnSync('bibtex',['reference'],{cwd:directory,encoding:'utf8'});
   assert.equal(parsed.status,0,`${path}: ${parsed.stdout}\n${parsed.stderr}`);
   const bbl=await readFile(join(directory,'reference.bbl'),'utf8');
   assert.ok(bbl.includes(literal),`${path}: parser changed literal source path`);
  }
  if(probe.error?.code==='ENOENT') t.diagnostic('BibTeX binary unavailable; literal/identity assertions ran, real-parser integration omitted.');
 } finally {await rm(directory,{recursive:true,force:true});}
});
