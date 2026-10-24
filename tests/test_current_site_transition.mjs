// Private test-first source proposal. Never run locally in the authoring unit.
// Production break caught: a publication advances current.json while an open
// document still advertises the previous digest; the reader must recover one
// coherent verified successor pair, with no admission under the old digest.
import test from 'node:test';
import assert from 'node:assert/strict';
import {webcrypto, createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {fetchVerifiedSnapshot, parseVerifiedSnapshot} from '../docs/site/current-model.mjs';

const fixtureURL = name => new URL(`./fixtures/current_site_transition_20261022/${name}`, import.meta.url);
const raw = async name => new Uint8Array(await readFile(fixtureURL(name)));
const [oldRaw,newRaw,oldCorrupt,newCorrupt,identityRaw,oldMetaRaw,newMetaRaw] = await Promise.all([
  raw('old-observation.json'),raw('new-observation.json'),raw('old-same-length-corrupt.json'),
  raw('new-same-length-corrupt.json'),raw('identities.json'),raw('old-meta.json'),raw('new-meta.json'),
]);
const decode = bytes => JSON.parse(new TextDecoder().decode(bytes));
const identities=decode(identityRaw), oldIdentity=identities.old, newIdentity=identities.new;
const oldData=decode(oldRaw),newData=decode(newRaw),oldMeta=decode(oldMetaRaw),newMeta=decode(newMetaRaw);
const oldKey=`current.json?observation=${oldIdentity.sha256}`;
const newKey=`current.json?observation=${newIdentity.sha256}`;
const asBytes=value=>new TextEncoder().encode(JSON.stringify(value));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const reply=(url,body,status=200)=>({url,body,status});
const transition=(metadata=newMetaRaw,body=newRaw)=>[
  reply('current.json',newRaw),reply(oldKey,newRaw),reply('current-meta.json',metadata),reply(newKey,body),
];

// Replace only the external transport. Production hashing, UTF-8 decoding,
// JSON parsing, snapshot validation, recovery decisions and abort propagation
// remain real. Native Response supplies the complete fetch response interface.
function transport(steps,{controller}={}) {
  const calls=[];
  const fetcher=async(url,options={})=>{
    calls.push({url:String(url),options});
    if(options.signal?.aborted){const e=new Error('Fixture request aborted');e.name='AbortError';throw e;}
    const step=steps[calls.length-1];
    if(!step||String(url)!==step.url)throw new Error(`Transport fixture rejected unexpected request ${String(url)}`);
    if(step.waitForAbort)return new Promise((resolve,reject)=>{
      const abort=()=>{const e=new Error('Fixture request timed out');e.name='AbortError';reject(e);};
      options.signal.addEventListener('abort',abort,{once:true});
      // The timeout occurs only after this request starts; no race with crypto.
      setTimeout(()=>controller.abort(),0);
    });
    return new Response(step.body,{status:step.status??200,headers:{'Content-Type':'application/json'}});
  };
  return {fetcher,calls};
}
async function outcome(fetcher,identity,signal) {
  try{return {value:await fetchVerifiedSnapshot(fetcher,identity,{cryptoAPI:webcrypto,signal}),error:null};}
  catch(error){return {value:null,error};}
}
function requests(calls,paths,signal) {
  assert.deepEqual(calls.map(call=>call.url),paths,'The bounded recovery must use the specified same-origin requests exactly once.');
  for(const {options}of calls){
    assert.equal(options.cache,'no-store');assert.equal(options.credentials,'omit');
    assert.equal(options.method??'GET','GET');assert.equal(options.signal,signal,'One caller abort budget spans all requests.');
  }
}
function success(result,expected) {
  assert.equal(result.error,null,`Expected a verified coherent observation, received ${result.error?.name}: ${result.error?.message}`);
  assert.deepEqual(result.value,expected);
}
function refusal(result) {assert.equal(result.value,null);assert.ok(result.error instanceof Error,'Unverified or incoherent bytes must refuse.');}

test('fixture positives are valid independently frozen old and new observations',async()=>{
  for(const [body,id,expected]of [[oldRaw,oldIdentity,oldData],[newRaw,newIdentity,newData]]){
    assert.equal(body.byteLength,id.bytes);assert.equal(sha(body),id.sha256);
    assert.deepEqual(await parseVerifiedSnapshot(body,id,webcrypto),expected);
  }
  assert.notEqual(oldRaw.byteLength,newRaw.byteLength);
  assert.equal(oldRaw.byteLength,oldCorrupt.byteLength);assert.notEqual(sha(oldRaw),sha(oldCorrupt));
});
for(const [name,body,id,expected]of [['old',oldRaw,oldIdentity,oldData],['new',newRaw,newIdentity,newData]]){
  test(`matching ${name} document and ${name} payload need exactly one verified request`,async()=>{
    const controller=new AbortController(),t=transport([reply('current.json',body)]);
    success(await outcome(t.fetcher,id,controller.signal),expected);requests(t.calls,['current.json'],controller.signal);
  });
}
test('the existing digest-keyed retry still recovers an alias lagging a new document',async()=>{
  const controller=new AbortController(),t=transport([reply('current.json',oldRaw),reply(newKey,newRaw)]);
  success(await outcome(t.fetcher,newIdentity,controller.signal),newData);
  requests(t.calls,['current.json',newKey],controller.signal);
});
test('old document plus newer payload automatically reacquires one verified successor pair',async()=>{
  const controller=new AbortController(),t=transport(transition());
  const result=await outcome(t.fetcher,oldIdentity,controller.signal);
  // On unchanged f946 this is an AssertionError after the real count refusal,
  // rather than an uncaught import/setup exception or a missing proposed API.
  success(result,newData);
  requests(t.calls,['current.json',oldKey,'current-meta.json',newKey],controller.signal);
});
test('same-length digest mismatch also recovers without admitting corrupt old bytes',async()=>{
  const controller=new AbortController(),t=transport([
    reply('current.json',oldCorrupt),reply(oldKey,oldCorrupt),reply('current-meta.json',newMetaRaw),reply(newKey,newRaw),
  ]);
  success(await outcome(t.fetcher,oldIdentity,controller.signal),newData);
  requests(t.calls,['current.json',oldKey,'current-meta.json',newKey],controller.signal);
});
test('published legacy three-field metadata can bind a successor without an archive assumption',async()=>{
  const controller=new AbortController(),t=transport(transition(asBytes(newIdentity)));
  success(await outcome(t.fetcher,oldIdentity,controller.signal),newData);
  requests(t.calls,['current.json',oldKey,'current-meta.json',newKey],controller.signal);
});

const invalidMetadata=[
  ['malformed JSON',new TextEncoder().encode('{broken')],
  ['missing digest',asBytes({bytes:newIdentity.bytes,observed_at:newIdentity.observed_at})],
  ['unknown field',asBytes({...newMeta,private_url:'must not be accepted'})],
  ['transient four-field schema',asBytes({...newIdentity,archive:'4'.repeat(64)})],
  ['nonhex digest',asBytes({...newMeta,sha256:'not-a-digest'})],
  ['string byte count',asBytes({...newMeta,bytes:String(newIdentity.bytes)})],
  ['negative byte count',asBytes({...newMeta,bytes:-1})],
  ['fractional byte count',asBytes({...newMeta,bytes:1.5})],
  ['payload above existing 16 MiB bound',asBytes({...newMeta,bytes:16777217})],
  ['invalid calendar timestamp',asBytes({...newMeta,observed_at:'2026-02-30T21:29:17Z'})],
  ['invalid archive identity',asBytes({...newMeta,archive:'wrong'})],
  ['invalid history identity',asBytes({...newMeta,history_manifest_sha256:'wrong'})],
  ['metadata above 16 KiB',new TextEncoder().encode(JSON.stringify(newMeta).padEnd(16385,' '))],
];
for(const [name,metadata]of invalidMetadata){
  test(`successor metadata refuses ${name} before requesting successor body`,async()=>{
    const controller=new AbortController(),t=transport(transition(metadata));
    refusal(await outcome(t.fetcher,oldIdentity,controller.signal));
    requests(t.calls,['current.json',oldKey,'current-meta.json'],controller.signal);
  });
}
test('metadata HTTP failure refuses without another body request or recovery cycle',async()=>{
  const controller=new AbortController(),steps=transition();steps[2]=reply('current-meta.json',newMetaRaw,503);
  const t=transport(steps);refusal(await outcome(t.fetcher,oldIdentity,controller.signal));
  requests(t.calls,['current.json',oldKey,'current-meta.json'],controller.signal);
});
test('verified successor body must have the metadata observation time',async()=>{
  const controller=new AbortController(),t=transport(transition(asBytes({...newMeta,observed_at:oldIdentity.observed_at})));
  refusal(await outcome(t.fetcher,oldIdentity,controller.signal));
  requests(t.calls,['current.json',oldKey,'current-meta.json',newKey],controller.signal);
});
for(const [name,body]of [['byte count',oldRaw],['digest',newCorrupt]]){
  test(`successor ${name} mismatch refuses after four requests with no loop`,async()=>{
    const controller=new AbortController(),t=transport(transition(newMetaRaw,body));
    refusal(await outcome(t.fetcher,oldIdentity,controller.signal));
    requests(t.calls,['current.json',oldKey,'current-meta.json',newKey],controller.signal);
  });
}
for(const [name,index,paths]of [
  ['metadata',2,['current.json',oldKey,'current-meta.json']],
  ['successor body',3,['current.json',oldKey,'current-meta.json',newKey]],
]){
  test(`one abort budget cancels a stalled ${name} request without another attempt`,{timeout:2000},async()=>{
    const controller=new AbortController(),steps=transition();steps[index]={url:paths.at(-1),waitForAbort:true};
    const t=transport(steps,{controller}),result=await outcome(t.fetcher,oldIdentity,controller.signal);
    refusal(result);assert.equal(result.error.name,'AbortError');requests(t.calls,paths,controller.signal);
  });
}
test('an already aborted caller is refused without metadata recovery',async()=>{
  const controller=new AbortController();controller.abort();const t=transport([reply('current.json',oldRaw)]);
  const result=await outcome(t.fetcher,oldIdentity,controller.signal);refusal(result);assert.equal(result.error.name,'AbortError');
  requests(t.calls,['current.json'],controller.signal);
});
test('ordinary payload HTTP failure is not treated as a publication identity transition',async()=>{
  const controller=new AbortController(),t=transport([reply('current.json',oldRaw,503)]);
  refusal(await outcome(t.fetcher,oldIdentity,controller.signal));requests(t.calls,['current.json'],controller.signal);
});
test('invalid HTML descriptor is refused before any request',async()=>{
  const controller=new AbortController(),t=transport([]);
  refusal(await outcome(t.fetcher,{sha256:'bad',bytes:oldIdentity.bytes},controller.signal));requests(t.calls,[],controller.signal);
});
test('hash-matching invalid snapshot JSON never earns successor recovery',async()=>{
  const body=new TextEncoder().encode('{broken'),id={sha256:sha(body),bytes:body.byteLength};
  const controller=new AbortController(),t=transport([reply('current.json',body)]);
  refusal(await outcome(t.fetcher,id,controller.signal));requests(t.calls,['current.json'],controller.signal);
});
