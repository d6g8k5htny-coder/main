import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {runInNewContext} from 'node:vm';

// Execute the production optional-status coordinator at its DOM/acquisition
// boundary. Real admission, native focus and camera geometry are browser-tested.
const source=await readFile(new URL('../docs/site/atlas.mjs',import.meta.url),'utf8');
const start=source.indexOf('  function patchOperationalResume(){'),end=source.indexOf('  query.value=state.query;',start);
assert.ok(start>=0&&end>start,'The production optional-status coordinator must be available');
const coordinator=source.slice(start,end);
const tick=async()=>{await Promise.resolve();await Promise.resolve();};
function fixture({controller=true}={}){
 const root=new EventTarget(),window=new EventTarget(),document={activeElement:null};
 const focus={},updates=[],pending=[];
 const resume={contains:node=>node===focus,replaceWith:record=>updates.push(record)};
 const context={root,window,document,selected:'a',graph:{nodes:{a:{},b:{}}},state:{error:null},data:{},nodeStatus:null,statusController:undefined,statusGeneration:0,statusActive:true,resumePending:false,Date,queueMicrotask,
  byId:()=>({querySelector:()=>resume}),hasNode:(graph,id)=>Object.hasOwn(graph.nodes,id),resumeSection:x=>x,resumeProjection:(data,id,now,records)=>({id,records}),
  loadNodeStatus(signal){return new Promise(resolve=>pending.push({signal,resolve}));}};
 if(controller)context.AbortController=AbortController;
 const api=runInNewContext(coordinator+'\n({request:requestNodeStatus,patch:patchOperationalResume,select:id=>selected=id,refuse:error=>state.error=error,records:()=>nodeStatus});',context);
 return {api,root,window,document,focus,updates,pending};
}
test('late optional data patches the currently selected node rather than its request-time selection',async()=>{
 const f=fixture();f.api.request();f.api.select('b');const records={b:{next_action:'actual selected record'}};f.pending[0].resolve(records);await tick();
 assert.equal(f.updates.length,1);assert.equal(f.updates[0].id,'b');assert.equal(f.updates[0].records,records);
});
test('an older optional response cannot overwrite the newer acquisition even when it ignores abort',async()=>{
 const f=fixture();f.api.request();f.api.request();assert.equal(f.pending[0].signal.aborted,true);
 const latest={b:{next_action:'new'}};f.pending[1].resolve(latest);await tick();f.pending[0].resolve({a:{next_action:'old'}});await tick();
 assert.equal(f.api.records(),latest);assert.equal(f.updates.length,1);
});
test('focused operational evidence is retained until focus leaves its section',async()=>{
 const f=fixture();f.document.activeElement=f.focus;f.api.request();f.pending[0].resolve({a:{next_action:'late'}});await tick();assert.equal(f.updates.length,0);
 f.document.activeElement=null;f.root.dispatchEvent(new Event('focusout'));await tick();assert.equal(f.updates.length,1);
});
test('a refused or no-longer-known selection receives no late substituted resume',async()=>{
 for(const state of ['refused','unknown']){const f=fixture();f.api.request();if(state==='refused')f.api.refuse('Snapshot refused');else f.api.select('missing');f.pending[0].resolve({a:{next_action:'late'}});await tick();assert.equal(f.updates.length,0);}
});
test('pagehide aborts the request and suppresses an uncooperative late response',async()=>{
 const f=fixture();f.api.request();f.window.dispatchEvent(new Event('pagehide'));assert.equal(f.pending[0].signal.aborted,true);f.pending[0].resolve({a:{next_action:'late'}});await tick();assert.equal(f.updates.length,0);
});
test('a persisted pageshow acquires current optional records once and preserves null fallback',async()=>{
 const f=fixture();f.api.request();f.window.dispatchEvent(new Event('pagehide'));const event=new Event('pageshow');Object.defineProperty(event,'persisted',{value:true});f.window.dispatchEvent(event);assert.equal(f.pending.length,2);
 f.pending[1].resolve(null);await tick();assert.equal(f.updates.length,0);assert.equal(f.api.records(),null);
});
test('missing cancellation platform does not prevent primary reading or late status coordination',async()=>{
 const f=fixture({controller:false});f.api.request();assert.equal(f.pending[0].signal,undefined);f.pending[0].resolve({a:{next_action:'available'}});await tick();assert.equal(f.updates.length,1);
});
