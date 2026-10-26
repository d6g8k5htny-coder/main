import test from 'node:test';
import assert from 'node:assert/strict';
const labels=await import('../docs/site/atlas-spatial-labels.mjs').catch(()=>({}));
const entry=(id,x,y,width=140,height=44,selected=false,z=0)=>({id,x,y,z,width,height,selected});
const intersects=(a,b,gap=0)=>a.left<b.left+b.width+gap&&a.left+a.width+gap>b.left&&a.top<b.top+b.height+gap&&a.top+a.height+gap>b.top;
function verify(result,entries,width,height){
  for(const box of result){
    assert.ok(box.left>=4&&box.top>=4);
    assert.ok(box.left+box.width<=width-4&&box.top+box.height<=height-4);
    const source=entries.find(e=>e.id===box.id);
    assert.equal(box.width,source.width);assert.equal(box.height,source.height);
    assert.ok(!intersects(box,{left:source.x-5,top:source.y-5,width:10,height:10}),'own node stays clear');
  }
  for(let i=0;i<result.length;i++)for(let j=i+1;j<result.length;j++)assert.ok(!intersects(result[i],result[j],4),'measured labels stay separate');
}
test('actual wrapped label sizes stay inside the viewport and do not overlap',()=>{
  assert.equal(typeof labels.placeSpatialLabels,'function','screen packing must exist');
  const input=[entry('long-claim',190,180,230,80,true),entry('a',160,160,180,44),entry('b',250,195,200,64),entry('c',220,250,120,44)];
  const result=labels.placeSpatialLabels(input,390,352);
  assert.ok(result.length>=2);assert.equal(result[0].id,'long-claim');verify(result,input,390,352);
});
test('selected claim takes priority over nearer coincident nodes without changing identifiers',()=>{
  assert.equal(typeof labels.placeSpatialLabels,'function');
  const id='math.d5-component.punctured-pin-continuum-check';
  const input=[entry('near',180,170,210,64,false,-.8),entry(id,180,170,230,80,true,.8)];
  const result=labels.placeSpatialLabels(input,390,352);
  assert.equal(result[0].id,id);verify(result,input,390,352);
});
test('labels leave a distinct unselected node clear for actual canvas picking',()=>{
  assert.equal(typeof labels.placeSpatialLabels,'function');
  const input=[entry('selected',195,180,230,80,true),entry('front-node',195,100,140,44,false,-.8)];
  const result=labels.placeSpatialLabels(input,390,352);
  assert.equal(result[0].id,'selected');verify(result,input,390,352);
  const target={left:190,top:95,width:10,height:10};
  assert.ok(result.every(box=>!intersects(box,target)),'An unselected node must remain uncovered');
});
test('camera-depth order and identifier ties are deterministic and inputs remain unchanged',()=>{
  assert.equal(typeof labels.placeSpatialLabels,'function');
  const input=[entry('z',150,140,110,44,false,.5),entry('b',320,200,120,60,false,-.4),entry('a',200,290,160,44,false,-.4)];
  const original=structuredClone(input);input.forEach(Object.freeze);Object.freeze(input);
  const result=labels.placeSpatialLabels(input,632,398);
  assert.deepEqual(result,labels.placeSpatialLabels([...input].reverse(),632,398));
  assert.equal(result[0].id,'a');assert.deepEqual(input,original);verify(result,input,632,398);
});
test('empty/tiny viewports and invalid or clipped points never create invalid rectangles',()=>{
  assert.equal(typeof labels.placeSpatialLabels,'function');
  const input=[entry('behind',100,100,100,44,false,2),entry('left',-1,100),entry('right',401,100),entry('bad',NaN,100),entry('huge',100,100,500,44)];
  assert.deepEqual(labels.placeSpatialLabels([],390,352),[]);
  assert.deepEqual(labels.placeSpatialLabels(input,390,352),[]);
  assert.deepEqual(labels.placeSpatialLabels([entry('a',0,0)],0,0),[]);
  assert.deepEqual(labels.placeSpatialLabels([entry('a',0,0)],NaN,352),[]);
});
test('edge anchors retain an in-bounds readable label when space permits',()=>{
  assert.equal(typeof labels.placeSpatialLabels,'function');
  for(const [x,y] of [[1,1],[389,1],[1,351],[389,351]]){
    const input=[entry('selected',x,y,230,80,true)];const result=labels.placeSpatialLabels(input,390,352);
    assert.equal(result.length,1);verify(result,input,390,352);
  }
});
test('a150-node dense projection keeps a selected ID readable without pretending every label fits',()=>{
  assert.equal(typeof labels.placeSpatialLabels,'function');
  const input=Array.from({length:150},(_,i)=>entry('n'+i,20+(i%15)*23,25+Math.floor(i/15)*30,160+(i%3)*25,44+(i%4)*16,i===149,i/150));
  const result=labels.placeSpatialLabels(input,390,352);
  assert.equal(result[0].id,'n149');assert.ok(result.length<150&&result.length>1);verify(result,input,390,352);
});
