import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {verifiedBytes,verifiedJSON,validateStatus,validateCoefficient,safeSourceURL} from '../docs/site/core.mjs';
const bytes=new TextEncoder().encode('{"value":"0.04177593184059834334"}');
const pin={url:'status.json',bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')};
const fetcher=async()=>new Response(bytes);
test('valid source preserves exact strings',async()=>assert.equal((await verifiedJSON(pin,fetcher)).value,'0.04177593184059834334'));
test('digest mismatch refuses data',async()=>assert.rejects(()=>verifiedBytes({...pin,sha256:'0'.repeat(64)},fetcher),/SHA-256/));
test('byte count mismatch refuses data',async()=>assert.rejects(()=>verifiedBytes({...pin,bytes:pin.bytes+1},fetcher),/byte count/));
test('mutable branch URL refused before request',async()=>assert.rejects(()=>verifiedBytes({...pin,url:'https://raw.githubusercontent.com/d6g8k5htny-coder/main/main/STATUS.md'},()=>{throw Error('must not fetch');}),/pinned public/));
test('HTTP failure gives no data',async()=>assert.rejects(()=>verifiedBytes(pin,async()=>new Response('',{status:404})),/unavailable/));
const status=JSON.parse(fs.readFileSync(new URL('../docs/site/status.json',import.meta.url)));
test('status snapshot counts are selected rows',()=>assert.deepEqual(validateStatus(status).counts,{accept:4,amend:3,engineering:3}));
test('unknown category refused',()=>{const s=structuredClone(status);s.sections[0].key='proved';assert.throws(()=>validateStatus(s),/Unrecognized/);});
test('changed count refused',()=>{const s=structuredClone(status);s.counts.accept=5;assert.throws(()=>validateStatus(s),/count/);});
test('promotion authority refused',()=>assert.throws(()=>validateStatus({...status,scientific_status_authority:true}),/Invalid/));
const coefficient={object:'SIDE24-COEFFICIENT-D23-20260924-v1',scientific_acceptance:false,dimensions:{'2':{lower:'0.07340691930603427103',upper:'0.07340691930603427104'},'3':{lower:'0.04177593184059834334',upper:'0.04177593184059834335'}}};
test('coefficient endpoints remain distinct exactly',()=>{const d=validateCoefficient(coefficient).dimensions['2'];assert.notEqual(d.lower,d.upper);assert.equal(Number(d.lower),Number(d.upper));});
test('float endpoints refused',()=>{const d=structuredClone(coefficient);d.dimensions['2'].lower=Number(d.dimensions['2'].lower);assert.throws(()=>validateCoefficient(d));});
test('coefficient acceptance flag cannot be flipped',()=>assert.throws(()=>validateCoefficient({...coefficient,scientific_acceptance:true})));
const row={repository:'Math-',commit:'a'.repeat(40),path:'proofs/a proof.md',sha256:'b'.repeat(64)};
test('catalog points at immutable public paths',()=>assert.ok(safeSourceURL(row).endsWith('/proofs/a%20proof.md')));
test('catalog rejects private repo and traversal',()=>{assert.throws(()=>safeSourceURL({...row,repository:'sandbox'}));assert.throws(()=>safeSourceURL({...row,path:'../STATUS.md'}));});

test('correct bytes cannot be attributed to a different commit',async()=>{
 const full={...pin,repository:'d6g8k5htny-coder/main',commit:'a'.repeat(40),path:'STATUS.md',url:'https://raw.githubusercontent.com/d6g8k5htny-coder/main/'+ 'b'.repeat(40)+'/STATUS.md'};
 await assert.rejects(()=>verifiedBytes(full,fetcher),/Displayed identity/);
});
test('correct bytes cannot be attributed to a different path',async()=>{
 const full={...pin,repository:'d6g8k5htny-coder/main',commit:'a'.repeat(40),path:'STATUS.md',url:'https://raw.githubusercontent.com/d6g8k5htny-coder/main/'+ 'a'.repeat(40)+'/README.md'};
 await assert.rejects(()=>verifiedBytes(full,fetcher),/Displayed identity/);
});
