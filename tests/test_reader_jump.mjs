import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {startReaderJump} from '../docs/site/reader-jump.mjs';
const root=new URL('../docs/site/',import.meta.url);
const pages=['index','explore','research','workspace','compare','cite','reproduce','dependencies','measure','museum','formal'];
test('current readers expose consistent native task links with real destinations',()=>{
  let expected;
  for(const page of pages){
    const html=fs.readFileSync(new URL(page+'.html',root),'utf8');
    assert.equal((html.match(/class="reader-jump"/g)||[]).length,1,page);
    const menu=html.match(/<details class="reader-jump">(.*?)<\/details>/s)?.[1];assert.ok(menu,page);
    const links=[...menu.matchAll(/href="([^"]+)"/g)].map(m=>m[1]);
    if(!expected)expected=links;else assert.deepEqual(links,expected,page);
    for(const href of links){
      const [name,id]=href.split('#');const target=new URL(name,root);assert.ok(fs.existsSync(target),href);
      if(id)assert.ok(fs.readFileSync(target,'utf8').includes(`id="${id}"`),href);
    }
    assert.ok(html.includes('reader-jump.css'),page);
    if(page==='formal')assert.ok(!/<script\b/i.test(html),'Formal stays script-free');
    else assert.ok(html.includes('reader-jump.mjs'),page);
  }
});
test('missing and already-enhanced menus receive no extra listeners',()=>{
  assert.equal(startReaderJump({querySelector:()=>null},{}),null);
  assert.equal(startReaderJump({querySelector:()=>({dataset:{enhanced:'true'}})},{}),null);
});
