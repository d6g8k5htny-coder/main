import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash,webcrypto} from 'node:crypto';
import * as model from '../docs/site/current-model.mjs';

const pin='a'.repeat(40),hash='b'.repeat(64);
const snapshot=()=>({schema_version:1,scientific_effect:'NONE',observed_at:'2026-10-20T12:06:05Z',
 repositories:[{repository:'d6g8k5htny-coder/main',commit:pin,tree:pin},{repository:'d6g8k5htny-coder/Math-',commit:pin,tree:pin}],
 sources:[],references:[],equations:[],
 graph:{identity:{repository:'d6g8k5htny-coder/Math-',commit:pin,path:'frontiers/test/GRAPH.json',blob:pin,bytes:1,sha256:hash},
 data:{nodes:{a:{kind:'candidate',classification:'OPEN_ACTIVE'}},edges:[]}}});
const encode=x=>Buffer.from(JSON.stringify(x));
const identity=raw=>({sha256:createHash('sha256').update(raw).digest('hex'),bytes:raw.length});
const response=(raw,status=200)=>({ok:status>=200&&status<300,arrayBuffer:async()=>Uint8Array.from(raw).buffer});
const pair=()=>{const data=snapshot(),old=structuredClone(data);old.observed_at='2026-10-20T11:53:44Z';return{data,raw:encode(data),old:encode(old)};};
const run=(fetcher,id,options={})=>{assert.equal(typeof model.fetchVerifiedSnapshot,'function');return model.fetchVerifiedSnapshot(fetcher,id,{cryptoAPI:webcrypto,...options});};

test('coherent canonical response needs one request and preserves the advertised observation',async()=>{
 const {raw,data}=pair(),seen=[];
 const result=await run(async(url,opts)=>{seen.push({url,opts});return response(raw);},identity(raw));
 assert.deepEqual(result,data);assert.equal(seen.length,1);assert.equal(seen[0].url,'current.json');
 assert.equal(seen[0].opts.cache,'no-store');assert.equal(seen[0].opts.credentials,'omit');
});
test('mixed publication recovers once at the page digest while retaining the same abort signal',async()=>{
 const {raw,old,data}=pair(),id=identity(raw),seen=[],controller=new AbortController();
 const result=await run(async(url,opts)=>{seen.push({url,opts});return response(url==='current.json'?old:raw);},id,{signal:controller.signal});
 assert.deepEqual(result,data);assert.deepEqual(seen.map(x=>x.url),['current.json','current.json?observation='+id.sha256]);
 assert.ok(seen.every(x=>x.opts.cache==='no-store'&&x.opts.credentials==='omit'&&x.opts.signal===controller.signal));
});
test('persistent digest mismatch refuses after two requests and never admits the old observation',async()=>{
 const {raw,old}=pair();let calls=0;
 await assert.rejects(run(async()=>{calls++;return response(old);},identity(raw)),/digest mismatch/);
 assert.equal(calls,2);
});
test('byte-count mismatch may recover but the recovery bytes are checked as strictly',async()=>{
 const {raw}=pair();let calls=0;
 assert.equal((await run(async()=>response(++calls===1?Buffer.from('{}'):raw),identity(raw))).observed_at,'2026-10-20T12:06:05Z');
 assert.equal(calls,2);calls=0;
 await assert.rejects(run(async()=>{calls++;return response(Buffer.from('{}'));},identity(raw)),/byte count mismatch/);
 assert.equal(calls,2);
});
test('HTTP refusal is not retried as an integrity mismatch',async()=>{
 const {raw}=pair();let calls=0;
 await assert.rejects(run(async()=>{calls++;return response(raw,503);},identity(raw)),/unavailable/);
 assert.equal(calls,1);
});
test('network failure is not retried and does not replace source identity',async()=>{
 const {raw}=pair();let calls=0;
 await assert.rejects(run(async()=>{calls++;throw Error('offline');},identity(raw)),/offline/);
 assert.equal(calls,1);
});
test('invalid schema with a correct digest is refused without a recovery request',async()=>{
 const data=snapshot();data.scientific_effect='ACCEPTED';const raw=encode(data);let calls=0;
 await assert.rejects(run(async()=>{calls++;return response(raw);},identity(raw)),/Invalid current source snapshot/);
 assert.equal(calls,1);
});
test('fallback HTTP refusal remains unavailable',async()=>{
 const {raw,old}=pair();let calls=0;
 await assert.rejects(run(async()=>response(++calls===1?old:raw,calls===1?200:404),identity(raw)),/unavailable/);
 assert.equal(calls,2);
});
test('absence of digest verification cannot trigger an unverified fallback',async()=>{
 const {raw}=pair();let calls=0;
 await assert.rejects(run(async()=>{calls++;return response(raw);},identity(raw),{cryptoAPI:null}),/verification unavailable/);
 assert.equal(calls,1);
});
test('aborted acquisition makes no recovery request',async()=>{
 const {raw}=pair(),controller=new AbortController();controller.abort();let calls=0;
 await assert.rejects(run(async(_url,opts)=>{calls++;assert.equal(opts.signal,controller.signal);throw new DOMException('aborted','AbortError');},identity(raw),{signal:controller.signal}),/aborted/);
 assert.equal(calls,1);
});
test('invalid advertised identity is rejected before a request can construct an injected URL',async()=>{
 const {raw}=pair();
 for(const id of [{...identity(raw),sha256:'../../other?bad=1'},{...identity(raw),bytes:-1},{...identity(raw),bytes:'NaN'}]){
  let calls=0;await assert.rejects(run(async()=>{calls++;return response(raw);},id),/identity/);assert.equal(calls,0);
 }
});
