import test from 'node:test';
import assert from 'node:assert/strict';
let discovery={};
try{discovery=await import('../docs/site/library-discovery.mjs');}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;}
const {libraryCollection,libraryDiscoveryHref,excerptSourceHref}=discovery;
const commit='a'.repeat(40);

test('existing and unknown collection bookmarks retain Files while explicit text survives',()=>{
  assert.equal(typeof libraryCollection,'function');
  for(const search of ['', '?source-q=gradients','?library-collection=constructor','?library-collection=FILES'])assert.equal(libraryCollection(search),'files');
  assert.equal(libraryCollection('?library-collection=text&source-q=gradients'),'text');
});
test('collection links preserve literal query and independent state without changing the target',()=>{
  assert.equal(typeof libraryDiscoveryHref,'function');
  const base='https://example.test/main/site/workspace.html?keep=a%26b&source-q=%CE%94%2B%23&library-collection=other#reading-list';
  const text=new URL(libraryDiscoveryHref(base,'text'));
  assert.equal(text.searchParams.get('source-q'),'Δ+#');assert.equal(text.searchParams.get('keep'),'a&b');
  assert.equal(text.searchParams.get('library-collection'),'text');assert.equal(text.hash,'#reading-list');
  const files=new URL(libraryDiscoveryHref(text,'files'));assert.equal(files.searchParams.has('library-collection'),false);
  assert.equal(files.pathname,'/main/site/workspace.html');assert.equal(files.searchParams.get('source-q'),'Δ+#');
});
test('an excerpt link retains exact source commit, literal filename and complete line range',()=>{
  assert.equal(typeof excerptSourceHref,'function');
  const e={repository:'d6g8k5htny-coder/Math-',commit,path:'proofs/a b.md',line_start:91,line_end:94};
  assert.equal(excerptSourceHref(e),'https://github.com/d6g8k5htny-coder/Math-/blob/'+commit+'/proofs/a%20b.md#L91-L94');
  assert.equal(excerptSourceHref({...e,line_end:91}),'https://github.com/d6g8k5htny-coder/Math-/blob/'+commit+'/proofs/a%20b.md#L91');
});
test('invalid or missing excerpt ranges cannot become invented source line identities',()=>{
  assert.equal(typeof excerptSourceHref,'function');
  const e={repository:'d6g8k5htny-coder/main',commit,path:'PROOF.md',line_start:2,line_end:4};
  for(const fields of [{line_start:0},{line_end:1},{line_start:'2'},{line_end:Infinity},{commit:'main'},{repository:'evil/../../x'},{path:'../outside'}])assert.equal(excerptSourceHref({...e,...fields}),null);
});
