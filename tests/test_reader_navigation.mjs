import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {legacyDestination} from '../docs/site/home.mjs';
import './test_workspace_navigation.mjs';
const site = new URL('../docs/site/', import.meta.url);
test('every former landing section routes to its preserved workspace section', () => {
  const workspace=fs.readFileSync(new URL('workspace.html',site),'utf8');
  for(const id of ['board','coefficient','inventory','contribute']) {
    assert.equal(legacyDestination('#'+id),'workspace.html#'+id);
    assert.ok(workspace.includes(`id="${id}"`));
  }
});
test('new home fragments and untrusted fragments never become redirect destinations', () => {
  for(const hash of ['', '#the-idea', '#discover', '#about', '#https://evil.test', '#//evil.test', '#inventory/../', '#INVENTORY']) assert.equal(legacyDestination(hash),null);
});
test('browser entry follows a bookmarked section and future hash changes', async () => {
  const replacements=[];let callback;
  globalThis.window={location:{hash:'#inventory',replace:v=>replacements.push(v)},addEventListener:(event,fn)=>{assert.equal(event,'hashchange');callback=fn;}};
  try {
    await import('../docs/site/home.mjs?browser-test');
    assert.deepEqual(replacements,['workspace.html#inventory']);
    window.location.hash='#about';callback();assert.equal(replacements.length,1);
    window.location.hash='#coefficient';callback();assert.equal(replacements[1],'workspace.html#coefficient');
  } finally { delete globalThis.window; }
});
test('the reader home does not preload the technical inventory or source viewer', () => {
  const home=fs.readFileSync(new URL('index.html',site),'utf8');
  assert.ok(!home.includes('src="app.js"'));
  assert.ok(!home.includes('Grab a task'));
  assert.ok(home.includes('href="workspace.html"'));
  assert.ok(fs.readFileSync(new URL('home.mjs',site),'utf8').includes('legacyDestination'));
});

test('verified source cards become reachable after asynchronous rendering', async () => {
  const {revealClaimFragment}=await import('../docs/site/museum.mjs');
  const scrolled=[],focused=[];
  const document={getElementById:id=>id==='d2-lifetime-remainder'?{scrollIntoView:()=>scrolled.push(id),focus:options=>focused.push([id,options])}:null};
  const allowed=['d2-lifetime-remainder','missing-card'];
  assert.equal(revealClaimFragment(document,'#d2-lifetime-remainder',allowed),true);
  for(const hash of ['#missing-card','#unknown','#%E0%A4%A','', '#https://evil.test']) assert.equal(revealClaimFragment(document,hash,allowed),false);
  assert.deepEqual(scrolled,['d2-lifetime-remainder']);
  assert.deepEqual(focused,[['d2-lifetime-remainder',{preventScroll:true}]]);
});
