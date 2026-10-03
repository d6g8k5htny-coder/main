import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const script=fileURLToPath(new URL('../tools/museum_conditionals.mjs',import.meta.url));
const fixture=fileURLToPath(new URL('./fixtures/cumulative_transfer_source.txt',import.meta.url));
const sourcePath='reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md';
const repository='d6g8k5htny-coder/Math-',commit='d6628da09384728992dcbe6e921cc28ba85aebb0';
const proof={repository,commit,path:sourcePath,bytes:3272,blob:'044ac5fdaf403a38e33983e31f0ad69f8e76d6d5',
  sha256:'83f653393dc6245980f6848e2bfc65ac9ad4c7d8304b028b88764356fd3825b6',
  url:`https://raw.githubusercontent.com/${repository}/${commit}/${sourcePath}`,
  html_url:`https://github.com/${repository}/blob/${commit}/${sourcePath}`};
function workspace(){
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'conditional-cli-'));
  const site=path.join(root,'docs/site');fs.mkdirSync(site,{recursive:true});
  return {root,site,cleanup:()=>fs.rmSync(root,{recursive:true,force:true})};
}
function putManifest(site,claims=[{id:'cumulative-transfer-correction',proof}]){
  const raw=Buffer.from(JSON.stringify({schema_version:1,scientific_status_authority:false,claims}));
  fs.writeFileSync(path.join(site,'museum.json'),raw);
  fs.writeFileSync(path.join(site,'config.json'),JSON.stringify({museum_json:{url:'museum.json',bytes:raw.length,sha256:createHash('sha256').update(raw).digest('hex')}}));
}
function run(root,...extra){
  assert.ok(fs.existsSync(script),'conditional CLI is absent');
  return spawnSync(process.execPath,[script,'--root',root,'--source',fixture,...extra],{encoding:'utf8',timeout:10000});
}
test('museum HTML has the route host, navigation and module without replacing existing cards',()=>{
  const htmlURL=new URL('../docs/site/museum.html',import.meta.url);
  assert.ok(fs.existsSync(htmlURL),'updated museum HTML is absent');
  const html=fs.readFileSync(htmlURL,'utf8');
  for(const text of ['id="conditional-route"','href="#conditionals"','id="claim-cards"','id="packet-cards"'])assert.ok(html.includes(text),text);
  for(const module of ['conditionals','museum'])assert.match(html,new RegExp(`src="${module}\\.mjs\\?site-release=[0-9a-f]{64}"`));
  assert.equal((html.match(/id="conditional-route"/g)||[]).length,1);
});
test('CLI projects the exact source and produces deterministic JSON',()=>{
  const w=workspace();try{
    putManifest(w.site);const p=run(w.root),again=run(w.root);
    assert.equal(p.status,0,p.stderr);assert.equal(again.stdout,p.stdout);
    const data=JSON.parse(p.stdout);assert.equal(data.nodes.length,7);
    assert.equal(data.routes[0].logic,'ALL');assert.equal(data.application_evaluation,'NOT_EVALUATED');
  }finally{w.cleanup();}
});
test('CLI rejects altered manifest bytes',()=>{
  const w=workspace();try{
    putManifest(w.site);fs.appendFileSync(path.join(w.site,'museum.json'),' ');
    const p=run(w.root);assert.notEqual(p.status,0);assert.match(p.stderr,/REFUSED.*manifest/i);
  }finally{w.cleanup();}
});
test('CLI refuses missing or repeated conditional claim',()=>{
  const w=workspace();try{
    for(const claims of [[],[{id:'cumulative-transfer-correction',proof},{id:'cumulative-transfer-correction',proof}]]){
      putManifest(w.site,claims);const p=run(w.root);assert.notEqual(p.status,0);assert.match(p.stderr,/REFUSED/);
    }
  }finally{w.cleanup();}
});
test('CLI rejects weakened projection in check mode',()=>{
  const w=workspace();try{
    putManifest(w.site);const p=run(w.root);assert.equal(p.status,0,p.stderr);
    const output=path.join(w.root,'projection.json');fs.writeFileSync(output,p.stdout);
    assert.equal(run(w.root,'--check',output).status,0);
    const data=JSON.parse(p.stdout);data.routes[0].logic='ANY';fs.writeFileSync(output,JSON.stringify(data));
    const bad=run(w.root,'--check',output);assert.notEqual(bad.status,0);assert.match(bad.stderr,/projection/);
  }finally{w.cleanup();}
});
test('CLI rejects nonlocal manifest and malformed hash types',()=>{
  const w=workspace();try{
    for(const change of [{url:'../other.json'},{sha256:123},{bytes:'3272'}]){
      putManifest(w.site);const f=path.join(w.site,'config.json'),c=JSON.parse(fs.readFileSync(f));
      Object.assign(c.museum_json,change);fs.writeFileSync(f,JSON.stringify(c));
      const p=run(w.root);assert.notEqual(p.status,0);assert.match(p.stderr,/REFUSED/);
    }
  }finally{w.cleanup();}
});
test('CLI refuses missing input and unknown options',()=>{
  const w=workspace();try{
    const p=run(w.root);assert.notEqual(p.status,0);assert.match(p.stderr,/REFUSED/);
    putManifest(w.site);assert.notEqual(run(w.root,'--accept','true').status,0);
  }finally{w.cleanup();}
});
test('CLI rejects source substitution',()=>{
  const w=workspace();try{
    putManifest(w.site,[{id:'cumulative-transfer-correction',proof:{...proof,commit:'main'}}]);
    const p=run(w.root);assert.notEqual(p.status,0);assert.match(p.stderr,/identity/);
  }finally{w.cleanup();}
});
