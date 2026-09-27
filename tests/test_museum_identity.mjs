// Regression for hash-correct bytes relabeled with an unaudited commit.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {createHash,webcrypto} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {projectVerified} from '../docs/site/conditionals.mjs';
const repository='d6g8k5htny-coder/Math-',commit='d6628da09384728992dcbe6e921cc28ba85aebb0';
const sourcePath='reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md';
const fixture=fileURLToPath(new URL('./fixtures/cumulative_transfer_source.txt',import.meta.url));
const raw=new Uint8Array(fs.readFileSync(fixture));
function pin(revision=commit){return {repository,commit:revision,path:sourcePath,bytes:3272,
  blob:'044ac5fdaf403a38e33983e31f0ad69f8e76d6d5',
  sha256:'83f653393dc6245980f6848e2bfc65ac9ad4c7d8304b028b88764356fd3825b6',
  url:`https://raw.githubusercontent.com/${repository}/${revision}/${sourcePath}`,
  html_url:`https://github.com/${repository}/blob/${revision}/${sourcePath}`};}
test('correct source bytes cannot acquire a different valid-looking commit',async()=>{
  await assert.rejects(projectVerified(raw,pin('1'.repeat(40)),webcrypto),/identity/);
});
test('exported projection preserves the complete audited source descriptor',async()=>{
  assert.deepEqual((await projectVerified(raw,pin(),webcrypto)).source,pin());
});
test('offline CLI refuses coherent hash-correct source relabeling',()=>{
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'conditional-identity-'));
  try{
    const site=path.join(root,'docs/site');fs.mkdirSync(site,{recursive:true});
    const manifest=Buffer.from(JSON.stringify({schema_version:1,scientific_status_authority:false,
      claims:[{id:'cumulative-transfer-correction',proof:pin('1'.repeat(40))}]}));
    fs.writeFileSync(path.join(site,'museum.json'),manifest);
    fs.writeFileSync(path.join(site,'config.json'),JSON.stringify({museum_json:{url:'museum.json',bytes:manifest.length,sha256:createHash('sha256').update(manifest).digest('hex')}}));
    const script=fileURLToPath(new URL('../tools/museum_conditionals.mjs',import.meta.url));
    const result=spawnSync(process.execPath,[script,'--root',root,'--source',fixture],{encoding:'utf8',timeout:10000});
    assert.notEqual(result.status,0);assert.match(result.stderr,/REFUSED:.*identity/);
  }finally{fs.rmSync(root,{recursive:true,force:true});}
});
