import test from 'node:test';
import assert from 'node:assert/strict';
import {librarySearchHref,startLibrarySourceSearch} from '../docs/site/library-source-search.mjs';

test('search links preserve literal text and unrelated page state',()=>{
  const base='https://example.test/main/site/workspace.html?keep=literal%23note&source-q=old#reading-list';
  for(const query of ['Δ + a&b#c?d%20','<img src=x onerror=alert(1)>','  coefficients  ']){
    const url=new URL(librarySearchHref(base,query));
    assert.equal(url.origin,'https://example.test');
    assert.equal(url.pathname,'/main/site/workspace.html');
    assert.equal(url.searchParams.get('keep'),'literal#note');
    assert.equal(url.searchParams.get('source-q'),query);
    assert.equal(url.hash,'#library-current');
  }
});
test('empty queries remove only the current-source parameter',()=>{
  const url=new URL(librarySearchHref('http://localhost/site/workspace.html?source-q=old&catalog-q=other&keep=1',''));
  assert.equal(url.searchParams.has('source-q'),false);
  assert.equal(url.searchParams.get('catalog-q'),'other');
  assert.equal(url.searchParams.get('keep'),'1');
});
test('no initialization on another page or before snapshot admission',()=>{
  assert.equal(startLibrarySourceSearch({getElementById:()=>null},{}),null);
  const doc={getElementById:id=>id==='research-desk'?{}:null,querySelector:()=>({disabled:true})};
  assert.equal(startLibrarySourceSearch(doc,{}),null);
});
