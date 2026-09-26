import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
// Minimal DOM observation harness: exercises loading/search/selection/currency.
// This is not a browser layout or accessibility test.
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.attributes={};this.listeners={};this.value='';this.hidden=false;this.disabled=false;this._text='';}
  set textContent(t){this._text=String(t);this.children=[];} get textContent(){return this._text+this.children.map(x=>typeof x==='string'?x:x.textContent).join('');}
  append(...children){this.children.push(...children);}replaceChildren(...c){this.children=c;this._text='';}
  setAttribute(k,v){this.attributes[k]=v;}addEventListener(k,f){this.listeners[k]=f;}
}
const nodes=new Map();
globalThis.document={getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id);},createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),createTextNode:t=>({textContent:t})};
document.getElementById('dimension').value='2';
const root=path.resolve(import.meta.dirname,'..');
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
 if(String(url).startsWith('https://api.github.com/'))return Response.json({ref:'refs/heads/main',object:{sha:config.query.math_pin}});
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
await document.getElementById('check-tip').listeners.click();
assert.match(document.getElementById('tip-result').textContent,/Commit pin current/);
assert.match(document.getElementById('tip-result').textContent,/Gate-byte equality was not tested/);
assert.ok(!requested.some(url=>/sandbox|Drive|google-drive/i.test(url)));
console.log('Public shop interaction harness passed: 2138 rows, exact decimals, search, selection, read-only pin currency.');
