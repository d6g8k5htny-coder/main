import test from 'node:test';
import assert from 'node:assert/strict';
import {relationshipGroups} from '../docs/site/atlas-relationships.mjs';

const graph={nodes:{parent:{},a:{},b:{},isolated:{}},edges:[
  {from:'parent',to:'a',required:true,relation:'reads_with_scope'},
  {from:'b',to:'parent',required:false,relation:'context_only'},
  {from:'parent',to:'a',required:false,relation:'alternative_reading'},
]};

test('dependency and dependent cards preserve parallel typed source edges in their actual direction',()=>{
  const before=JSON.stringify(graph);
  assert.deepEqual(relationshipGroups(graph,'parent'),[
    {id:'dependencies',label:'Depends on',items:[
      {nodeId:'a',from:'parent',to:'a',required:true,relation:'reads_with_scope'},
      {nodeId:'a',from:'parent',to:'a',required:false,relation:'alternative_reading'},
    ]},
    {id:'dependents',label:'Used by',items:[
      {nodeId:'b',from:'b',to:'parent',required:false,relation:'context_only'},
    ]},
  ]);
  assert.equal(JSON.stringify(graph),before,'Presentation must not alter recorded graph metadata');
});

test('an admitted isolated node has two explicit empty groups',()=>{
  assert.deepEqual(relationshipGroups(graph,'isolated'),[
    {id:'dependencies',label:'Depends on',items:[]},
    {id:'dependents',label:'Used by',items:[]},
  ]);
});

test('missing and prototype names cannot acquire inferred connections',()=>{
  for(const id of ['missing','__proto__','toString','constructor'])assert.deepEqual(relationshipGroups(graph,id),[]);
});
