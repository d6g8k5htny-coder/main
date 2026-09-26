import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
// Minimal DOM observation harness: exercises loading/search/selection/identity.
// This is not a browser layout or accessibility test.
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.attributes={};this.listeners={};this.value='';this.hidden=false;this.disabled=false;this._text='';}
  set textContent(t){this._text=String(t);this.children=[];} get textContent(){return this._text+this.children.map(x=>typeof x==='string'?x:x.textContent).join('');}
  append(...children){this.children.push(...children);}replaceChildren(...c){this.children=c;this._text='';}
  setAttribute(k,v){this.attributes[k]=v;}addEventListener(k,f){this.listeners[k]=f;}
}
const root=path.resolve(import.meta.dirname,'..');
const html=fs.readFileSync(path.join(root,'docs/site/index.html'),'utf8');
const nodes=new Map();
for(const match of html.matchAll(/<([a-z][a-z0-9-]*)\b[^>]*\bid="([^"]+)"[^>]*>/gi)) {
 assert.ok(!nodes.has(match[2]),`Duplicate HTML ID ${match[2]}`);
 nodes.set(match[2],new Element(match[1]));
}
// Missing HTML nodes stay missing, as they do in a browser.
globalThis.document={getElementById:id=>nodes.get(id)||null,createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),createTextNode:t=>({textContent:t})};
const required=id=>{assert.ok(nodes.has(id),`Missing real HTML element ${id}`);return nodes.get(id);};
required('dimension').value='2';
const fixture=process.env.SHOP_MATH_FIXTURE;
assert.ok(fixture,'Set SHOP_MATH_FIXTURE to exact public Math fixture directory');
const qfixture=process.env.SHOP_QUERY_FIXTURE;
assert.ok(qfixture,'Set SHOP_QUERY_FIXTURE to exact query fixture directory');
const config=JSON.parse(fs.readFileSync(path.join(root,'docs/site/config.json')));
const mapping=new Map([
 ['config.json',path.join(root,'docs/site/config.json')],['status.json',path.join(root,'docs/site/status.json')],
 [config.status.url,path.join(root,'STATUS.md')],
 [config.coefficient.url,path.join(fixture,config.coefficient.path)],
 [config.imports.url,path.join(fixture,config.imports.path)],
 [config.query.url,path.join(qfixture,config.query.path)],
 [config.inventory.url,path.join(root,'docs/public-math/sources.json')],
 ...config.inventory.pages.map(p=>[p.url,path.join(root,p.path)])
]);
let requested=[];
globalThis.fetch=async(url,options={})=>{requested.push(String(url));assert.equal(options.credentials,'omit');
 if(String(url).startsWith('https:'))assert.match(String(url),/^https:\/\/raw\.githubusercontent\.com\/d6g8k5htny-coder\/(?:main|Math-|query-)\/[0-9a-f]{40}\//,'Every remote read must use an immutable public commit');
 assert.ok(mapping.has(String(url)),`Unexpected request ${url}`);return new Response(fs.readFileSync(mapping.get(String(url))));};
await import('../docs/site/app.js');
assert.equal(document.getElementById('counts').children.length,3);
assert.match(document.getElementById('inventory-state').textContent,/2,138 matches/);
assert.equal(document.getElementById('catalog').children.length,50);
assert.equal(document.getElementById('lower').textContent,'0.07340691930603427103');
document.getElementById('dimension').value='3';document.getElementById('dimension').listeners.change();
assert.equal(document.getElementById('lower').textContent,'0.04177593184059834334');
document.getElementById('search').value='EC-014';document.getElementById('search').listeners.input();
assert.ok(document.getElementById('catalog').children.length>0);
assert.ok(document.getElementById('catalog').children.every(row=>row.textContent.toLowerCase().includes('ec-014')));
assert.ok(!requested.some(url=>/sandbox|Drive|google-drive/i.test(url)));
const disclaimer='This canvas explains the pinned source. It is not a proof and does not change status.';
function showsIdentity(element,pin,canvas=true) {
 for(const value of [pin.repository,pin.path,pin.commit,pin.sha256,...(canvas?[disclaimer]:[])])assert.ok(element.textContent.includes(value),`Visible source header missing ${value}`);
}
const descendants=element=>element.children.flatMap(child=>child instanceof Element?[child,...descendants(child)]:[]);
test('SIDE24 always identifies the engineering exhibit before the controls',()=>{
 const header=required('coefficient-source');
 showsIdentity(header,config.coefficient);
 assert.match(header.textContent,/D3 SIDE24/);
 assert.match(header.textContent,/SIDE24-COEFFICIENT-D23-20260924-v1/);
 assert.match(header.textContent,/engineering-only/);
 assert.ok(html.indexOf('id="coefficient-source"')<html.indexOf('class="explorer"'));
 assert.equal(required('upper').textContent,'0.04177593184059834335');
 assert.equal(required('moment').textContent,'29/6-sqrt(6)');
 showsIdentity(required('coefficient-identity'),config.coefficient,false);
});
test('count cards and every status row retain their own source context',()=>{
 const status=JSON.parse(fs.readFileSync(path.join(root,'docs/site/status.json')));
 for(const [i,card] of required('counts').children.entries()) {
  showsIdentity(card,status.source);
  assert.ok(card.textContent.includes('status-count-'+status.sections[i].key));
  assert.match(card.textContent,/engineering-only/);
 }
 const rows=descendants(required('status-rows')).filter(n=>n.className==='status-object');
 assert.equal(rows.length,10);
 const expected=status.sections.flatMap(section=>section.rows.map(row=>({section,row})));
 for(const [i,row] of rows.entries()) {
  showsIdentity(row,status.source);
  assert.match(row.textContent,/Object ID:/);
  const classes={accept:'ACCEPT-scoped',amend:'AMEND/open',engineering:'engineering-only'};
  assert.ok(row.textContent.includes('Class: '+classes[expected[i].section.key]));
  assert.equal(row.attributes['data-class'],classes[expected[i].section.key]);
  assert.ok(row.textContent.includes(expected[i].row[1].replace(/\*\*|`/g,'')));
 }
 assert.ok(descendants(required('status-rows')).some(n=>n.href==='https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841269490'));
});
test('repository and path filters combine with search and reset pagination',()=>{
 const repo=required('repository-filter'),pathFilter=required('path-filter'),search=required('search');
 search.value='';search.listeners.input();
 required('more').listeners.click();
 assert.equal(required('catalog').children.length,100);
 repo.value='Math-';repo.listeners.change();
 assert.equal(required('catalog').children.length,50);
 assert.ok(required('catalog').children.every(row=>row.children[0].children[0].textContent==='Math-'));
 required('more').listeners.click();
 assert.equal(required('catalog').children.length,54);
 pathFilter.value='coefficients/side24_v1/';pathFilter.listeners.input();
 assert.ok(required('catalog').children.length>0);
 assert.ok(required('catalog').children.every(row=>row.textContent.includes('coefficients/side24_v1/')));
 search.value='PROOF.md';search.listeners.input();
 assert.equal(required('catalog').children.length,1);
 assert.ok(required('catalog').children[0].children[0].children[1].href.includes('/blob/'));
 repo.value='main';repo.listeners.change();
 assert.equal(required('catalog').children.length,0);
 assert.match(required('inventory-state').textContent,/0 matches/);
 assert.equal(required('more').hidden,true);
 repo.value='';repo.listeners.change();pathFilter.value='';pathFilter.listeners.input();search.value='';search.listeners.input();
 assert.equal(required('catalog').children.length,50);
 assert.match(required('inventory-state').textContent,/2,138 matches/);
});
test('query view shows its pin and the local byte-check command without a live ref control',()=>{
 assert.equal(document.getElementById('check-tip'),null);
 assert.equal(document.getElementById('tip-result'),null);
 const query=required('query-identity');
 for(const value of [config.query.repository,config.query.path,config.query.commit,config.query.sha256,config.query.math_pin])assert.ok(query.textContent.includes(value));
 assert.ok(required('query-note').textContent.includes('python -B -S verify_portable_stubs.py --check-math-tip'));
 assert.ok(!requested.some(url=>url.startsWith('https://api.github.com/')));
});
console.log('Public shop base interactions passed: 2138 original rows, exact decimals, search and selection.');
