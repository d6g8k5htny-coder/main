import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import * as model from '../docs/site/current-model.mjs';
const fixtures=JSON.parse(await readFile(new URL('./fixtures/current_inventory_acquisition_cases.json',import.meta.url),'utf8'));
for(const fixture of fixtures.cases)test('acquisition: '+fixture.name,async()=>{
  assert.equal(typeof model.projectCurrentAcquisitions,'function','current acquisition projection is missing');
  const before=JSON.stringify(fixture.snapshot);
  assert.deepEqual(await model.projectCurrentAcquisitions(fixture.snapshot),fixture.expected);
  assert.equal(JSON.stringify(fixture.snapshot),before,'projection mutated source evidence');
});

test('current row keeps excerpt identities separate and renders hostile source text inertly',async()=>{
  const renderer=await import('../docs/site/current.mjs');
  assert.equal(typeof renderer.renderCurrentSource,'function','actual dynamic inventory row renderer is missing');
  class Element {
    constructor(tag){this.tag=tag;this.children=[];this._text='';}
    set textContent(value){this._text=String(value);}
    get textContent(){return this._text+this.children.map(x=>typeof x==='string'?x:x.textContent).join('');}
    append(...children){this.children.push(...children);}
  }
  const saved=Object.getOwnPropertyDescriptor(globalThis,'document');
  globalThis.document={createElement:tag=>new Element(tag)};
  try{
    const f=fixtures.cases[0],row=renderer.renderCurrentSource(f.snapshot.sources[2],f.expected.rows[2]);
    assert.match(row.textContent,/Included excerpts \(2\)/);
    assert.match(row.textContent,/lines 2–3/);
    assert.match(row.textContent,/excerpt SHA-256/);
    const links=[];const visit=n=>{if(n.tag==='a')links.push(n);for(const child of n.children||[])if(typeof child!=='string')visit(child);};visit(row);
    assert.ok(links.some(a=>a.href.endsWith('/proof/PROOF.md#L2-L3')));
    const hostile=renderer.renderCurrentSource(f.snapshot.sources[3],f.expected.rows[3]);
    assert.match(hostile.textContent,/<b>λ<\/b>/);
    const graph=renderer.renderCurrentSource(f.snapshot.sources[0],f.expected.rows[0]);
    assert.match(graph.textContent,/producer-recorded complete-file SHA-256/);
    assert.match(graph.textContent,/dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd/);
    const refused=renderer.renderCurrentSource({...f.snapshot.sources[3],commit:'main'},f.expected.rows[3]);
    const refusedLinks=[];const walk=n=>{if(n.tag==='a')refusedLinks.push(n);for(const c of n.children||[])if(typeof c!=='string')walk(c);};walk(refused);
    assert.equal(refusedLinks.length,0,'unsafe source became an interactive link');
  }finally{if(saved)Object.defineProperty(globalThis,'document',saved);else delete globalThis.document;}
});
