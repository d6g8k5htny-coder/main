import test from 'node:test';
import assert from 'node:assert/strict';
const module=await import('../docs/site/atlas-spatial-model.mjs').catch(()=>({}));
const graph={nodes:{a:{kind:'theorem'},b:{kind:'hypothesis'},c:{},d:{}},edges:[{from:'a',to:'b',relation:'requires',required:true},{from:'c',to:'a',relation:'context',required:false},{from:'b',to:'d',relation:'imports',required:true}]};
test('spatial projection keeps directed typed recorded edges and never mutates evidence',()=>{
 assert.equal(typeof module.spatialProjection,'function','spatialProjection must exist');
 const before=structuredClone(graph);const view=module.spatialProjection(graph,['a','b','c','missing'],'a');
 assert.deepEqual(view.ids,['a','b','c']);assert.deepEqual(view.edges,graph.edges.slice(0,2));assert.deepEqual(graph,before);
 assert.deepEqual(module.spatialProjection(graph,['a','b','c'],'a'),view);
 assert.ok(Object.values(view.positions).flat().every(Number.isFinite));assert.equal(new Set(Object.values(view.positions).map(JSON.stringify)).size,3);
});
test('dependency focus retains only selected dependent and its recorded direct dependencies',()=>{
 assert.equal(typeof module.spatialProjection,'function');
 const view=module.spatialProjection(graph,['a','b','c','d'],'a',true);assert.deepEqual(view.ids,['a','b']);assert.deepEqual(view.edges,[graph.edges[0]]);
});
test('empty/filtered selection and ceiling never invent source nodes',()=>{
 assert.equal(typeof module.spatialProjection,'function');
 assert.deepEqual(module.spatialProjection(graph,[],'a').ids,[]);
 assert.deepEqual(module.spatialProjection(graph,['b'],'a',true).ids,['b']);
 const large={nodes:Object.fromEntries(Array.from({length:200},(_,i)=>['n'+i,{}])),edges:[]};assert.equal(module.spatialProjection(large,Object.keys(large.nodes),'n0').ids.length,150);
});
