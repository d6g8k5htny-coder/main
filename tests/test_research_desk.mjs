import test from 'node:test';
import assert from 'node:assert/strict';

// Missing exports produce ordinary assertion failures during the first RED run.
let store={};
try { store=await import('../docs/site/research-desk-store.mjs'); }
catch(error) { if(error.code!=='ERR_MODULE_NOT_FOUND')throw error; }
const {validateEntry,entryKey,parseReadingList,mergeEntries,sourceHref,searchHref,readingListJSON}=store;
const desk=await import('../docs/site/research-desk.mjs');

test('source-button identity matches legal filename URI spellings without conflating path segments',()=>{
  const url='https://github.com/d6g8k5htny-coder/main/blob/'+'a'.repeat(40)+'/docs/proof(1)!\'*.md';
  const escaped=url.replace('(', '%28').replace(')', '%29').replace('!', '%21').replace("'", '%27').replace('*','%2A');
  assert.equal(typeof desk.sourceLinkKey,'function');
  assert.equal(desk.sourceLinkKey(url),desk.sourceLinkKey(escaped));
  assert.notEqual(desk.sourceLinkKey(url+'/a%2Fb'),desk.sourceLinkKey(url+'/a/b'));
  assert.equal(desk.sourceLinkKey('javascript:alert(1)'),null);
  assert.equal(desk.sourceLinkKey(url+'?redirect=elsewhere'),null);
});
const schema='universal-law-reading-list/v1';
const commit='0123456789abcdef0123456789abcdef01234567';
const savedAt='2026-10-25T20:30:00Z';
const source=(extra={})=>({kind:'source',collection:'current',repository:'d6g8k5htny-coder/main',commit,path:'docs/PROOF.md',savedAt,...extra});
const search=(extra={})=>({kind:'search',collection:'catalog',query:'SIDE24',repository:'Math-',path:'coefficients/',savedAt,...extra});
const documentOf=entries=>JSON.stringify({schema,entries});

test('valid entries retain exact identities and normalize only search whitespace',()=>{
  assert.equal(typeof validateEntry,'function');
  const original=source({blob:'a'.repeat(40),sha256:'b'.repeat(64),bytes:0,observedAt:'2024-02-29T23:59:59.125Z'});
  assert.deepEqual(validateEntry(original),original);
  assert.notEqual(validateEntry(original),original);
  assert.deepEqual(validateEntry(search({query:'  a + b & c  ',path:'  docs/  '})),search({query:'a + b & c',path:'docs/'}));
  assert.equal(validateEntry(search({query:'',repository:'',path:''})).query,'');
});

test('entries require exact plain JSON data records with no inherited or executable fields',()=>{
  assert.equal(typeof validateEntry,'function');
  const accessor=source();Object.defineProperty(accessor,'bytes',{get(){throw new Error('getter ran');},enumerable:true});
  const hidden=source();Object.defineProperty(hidden,'url',{value:'https://evil.test'});
  for(const invalid of [null,[],new Date(),Object.create(source()),Object.assign(new (class Entry {})(),source()),accessor,hidden,{...source(),[Symbol('extra')]:true}]) {
    assert.throws(()=>validateEntry(invalid),{name:'TypeError'});
  }
  assert.deepEqual(validateEntry(Object.assign(Object.create(null),source())),source());
});

test('unexpected keys never become imported links, statuses or inferred metadata',()=>{
  assert.equal(typeof validateEntry,'function');
  for(const field of ['url','href','status','certified','title','constructor','__proto__']) {
    const invalid=JSON.parse(documentOf([{...source(),[field]:'https://evil.test'}])).entries[0];
    assert.throws(()=>validateEntry(invalid),{name:'TypeError'});
  }
  assert.throws(()=>validateEntry({...search(),commit}),{name:'TypeError'});
});

test('required properties and string types cannot be coerced or defaulted',()=>{
  assert.equal(typeof validateEntry,'function');
  for(const make of [source,search]) {
    for(const key of Object.keys(make())) {
      const missing=make();delete missing[key];assert.throws(()=>validateEntry(missing),{name:'TypeError'});
      for(const value of [null,undefined,0,false,[],{}])assert.throws(()=>validateEntry(make({[key]:value})),{name:'TypeError'});
    }
  }
  assert.throws(()=>validateEntry(source({kind:'url'})),{name:'TypeError'});
  assert.throws(()=>validateEntry(search({collection:'all'})),{name:'TypeError'});
});

test('source repositories and commits must name an allowed owner and exact immutable commit',()=>{
  assert.equal(typeof validateEntry,'function');
  for(const repository of ['main','Math-','d6g8k5htny-coder/query-','other/main','d6g8k5htny-coder/main@evil.test','https://github.com/d6g8k5htny-coder/main']) {
    assert.throws(()=>validateEntry(source({repository})),{name:'TypeError'});
  }
  for(const value of ['main','a'.repeat(39),'a'.repeat(41),'A'.repeat(40),'g'.repeat(40)])assert.throws(()=>validateEntry(source({commit:value})),{name:'TypeError'});
  assert.equal(validateEntry(source({repository:'d6g8k5htny-coder/Math-'})).repository,'d6g8k5htny-coder/Math-');
});

test('source paths reject absolute, dot, empty, backslash, URL and control segments',()=>{
  assert.equal(typeof validateEntry,'function');
  for(const path of ['', '/', '/docs/a', './a', '../a', 'docs/../a', 'docs/./a','docs//a','docs/','docs\\a','https://evil.test/a','javascript:alert(1)','a\nb','a\u007fb','a\u0085b','a\ud800']) {
    assert.throws(()=>validateEntry(source({path})),{name:'TypeError'},JSON.stringify(path));
  }
  for(const path of ['.github/CODEOWNERS','space dir/Δ & %20.md','docs/%2e%2e/%2F.md','docs/a?b#c.md'])assert.equal(validateEntry(source({path})).path,path);
});

test('dates reject calendar overflow, offsets and invalid timestamp types',()=>{
  assert.equal(typeof validateEntry,'function');
  for(const field of ['savedAt','observedAt']) {
    for(const value of [null,undefined,0,'2026-02-29T00:00:00Z','2024-02-30T00:00:00Z','2026-13-01T00:00:00Z','2026-10-25T24:00:00Z','2026-10-25T00:60:00Z','2026-10-25T00:00:60Z','2026-10-25','2026-10-25T00:00:00+00:00','2026-10-25T00:00:00z','2026-10-25T00:00:00.Z','2026-10-25T00:00:00Z\n'])assert.throws(()=>validateEntry(source({[field]:value})),{name:'TypeError'},String(value));
    for(const value of ['2024-02-29T00:00:00Z','2026-10-25T00:00:00.1Z','2026-10-25T00:00:00.12Z','2026-10-25T00:00:00.123Z'])assert.equal(validateEntry(source({[field]:value}))[field],value);
  }
});

test('Git blobs, SHA-256 and byte counts retain distinct validated types',()=>{
  assert.equal(typeof validateEntry,'function');
  for(const [field,invalid] of [['blob',['b'.repeat(64),'A'.repeat(40),null,undefined,3]],['sha256',['b'.repeat(40),'A'.repeat(64),null,undefined,3]],['bytes',[-1,1.5,NaN,Infinity,Number.MAX_SAFE_INTEGER+1,'42',null,undefined]]]) {
    for(const value of invalid)assert.throws(()=>validateEntry(source({[field]:value})),{name:'TypeError'});
  }
  assert.equal(validateEntry(source({bytes:Number.MAX_SAFE_INTEGER})).bytes,Number.MAX_SAFE_INTEGER);
});

test('search filters are bounded inert text with collection-specific repositories',()=>{
  assert.equal(typeof validateEntry,'function');
  for(const field of ['query','path']) {
    assert.equal(validateEntry(search({[field]:'a'.repeat(500)}))[field].length,500);
    for(const value of ['a'.repeat(501),'a\nb','a\tb','a\u0000b','a\u007fb','a\ud800'])assert.throws(()=>validateEntry(search({[field]:value})),{name:'TypeError'});
  }
  for(const repository of ['other','d6g8k5htny-coder/main','https://evil.test'])assert.throws(()=>validateEntry(search({repository})),{name:'TypeError'});
  assert.deepEqual(validateEntry(search({collection:'current',query:'all files',repository:'',path:''})),search({collection:'current',query:'all files',repository:'',path:''}));
  assert.throws(()=>validateEntry(search({collection:'current',repository:'main',path:''})),{name:'TypeError'});
  assert.throws(()=>validateEntry(search({collection:'current',repository:'',path:'docs/'})),{name:'TypeError'});
});

test('source links are reconstructed from exact pins with literal raw Git path encoding',()=>{
  assert.equal(typeof sourceHref,'function');
  assert.equal(sourceHref(source()),'https://github.com/d6g8k5htny-coder/main/blob/0123456789abcdef0123456789abcdef01234567/docs/PROOF.md');
  assert.equal(sourceHref(source({repository:'d6g8k5htny-coder/Math-',path:'proofs/Δ & %2e%2e/a?#.md'})),'https://github.com/d6g8k5htny-coder/Math-/blob/0123456789abcdef0123456789abcdef01234567/proofs/%CE%94%20%26%20%252e%252e/a%3F%23.md');
  assert.throws(()=>sourceHref(source({url:'https://evil.test'})),{name:'TypeError'});
  assert.throws(()=>sourceHref(search()),{name:'TypeError'});
});

test('current search links omit every catalog and unrelated parameter',()=>{
  assert.equal(typeof searchHref,'function');
  const href=searchHref(search({collection:'current',query:' Δ + & #%2F ',repository:'',path:''}),'https://example.test/main/site/workspace.html?q=old&path=bad&repository=main&from=else#board');
  assert.equal(href,'https://example.test/main/site/workspace.html?source-q=%CE%94+%2B+%26+%23%252F#library-current');
  assert.equal(new URL(href).searchParams.get('source-q'),'Δ + & #%2F');
});

test('catalog search links omit current parameters and roundtrip all catalog filters',()=>{
  assert.equal(typeof searchHref,'function');
  const href=searchHref(search({query:' a+b & c ',repository:'Math-',path:' Δ/%2F? '}),'https://example.test/main/site/research.html?source-q=old&unrelated=1#current-research');
  assert.equal(href,'https://example.test/main/site/workspace.html?q=a%2Bb+%26+c&repository=Math-&path=%CE%94%2F%252F%3F#inventory');
  const url=new URL(href);
  assert.deepEqual([...url.searchParams],[['q','a+b & c'],['repository','Math-'],['path','Δ/%2F?']]);
});

test('blank saved searches restore their collection all-files view',()=>{
  assert.equal(typeof searchHref,'function');
  for(const [collection,hash] of [['current','#library-current'],['catalog','#inventory']])assert.equal(searchHref(search({collection,query:'',repository:'',path:''}),'https://example.test/site/?x=y#old'),'https://example.test/site/workspace.html'+hash);
  assert.throws(()=>searchHref(source(),'https://example.test/site/workspace.html'),{name:'TypeError'});
});

test('entry identity separates source pins and normalized collection search contexts',()=>{
  assert.equal(typeof entryKey,'function');
  assert.equal(entryKey(source()),entryKey(source({collection:'catalog',savedAt:'2026-10-24T00:00:00Z',bytes:1})));
  for(const changes of [{repository:'d6g8k5htny-coder/Math-'},{commit:'a'.repeat(40)},{path:'docs/OTHER.md'}])assert.notEqual(entryKey(source()),entryKey(source(changes)));
  assert.equal(entryKey(search()),entryKey(search({query:' SIDE24 ',path:' coefficients/ '})));
  assert.notEqual(entryKey(search({repository:'',path:''})),entryKey(search({collection:'current',repository:'',path:''})));
  assert.notEqual(entryKey(search()),entryKey(search({query:'other'})));
  assert.notEqual(entryKey(search()),entryKey(search({repository:'main'})));
  assert.notEqual(entryKey(search()),entryKey(search({path:'other/'})));
  assert.throws(()=>entryKey({...source(),certified:true}),{name:'TypeError'});
});

test('list parsing rejects malformed JSON, schema drift and unexpected root fields',()=>{
  assert.equal(typeof parseReadingList,'function');
  for(const value of ['', '{', 'null', '[]','42','"hello"',JSON.stringify({schema:'v2',entries:[]}),JSON.stringify({entries:[]}),JSON.stringify({schema}),JSON.stringify({schema,entries:{}}),JSON.stringify({schema,entries:[],status:'accepted'}),JSON.stringify({schema,entries:[],url:'https://evil.test'})])assert.throws(()=>parseReadingList(value));
  for(const value of [null,undefined,{},[]])assert.throws(()=>parseReadingList(value),{name:'TypeError'});
  assert.deepEqual(parseReadingList(documentOf([source(),search()])),{schema,entries:[source(),search()]});
});

test('128 KiB input ceiling uses UTF-8 bytes before JSON parsing',()=>{
  assert.equal(typeof parseReadingList,'function');
  const empty=documentOf([]),atLimit=empty+' '.repeat(131072-Buffer.byteLength(empty));
  assert.deepEqual(parseReadingList(atLimit),{schema,entries:[]});
  assert.throws(()=>parseReadingList(atLimit+' '),{name:'RangeError'});
  assert.throws(()=>parseReadingList('Δ'.repeat(65537)),{name:'RangeError'});
  assert.throws(()=>parseReadingList(' '.repeat(131073)+'{'),{name:'RangeError'});
});

test('merge validates every entry before returning and leaves inputs unchanged',()=>{
  assert.equal(typeof mergeEntries,'function');
  const existing=[source()],incoming=[search()],before=JSON.stringify([existing,incoming]);
  const result=mergeEntries(existing,incoming);
  assert.deepEqual(result,[source(),search()]);assert.notEqual(result,existing);assert.notEqual(result[0],existing[0]);
  assert.equal(JSON.stringify([existing,incoming]),before);
  assert.throws(()=>mergeEntries(existing,[search(),source({url:'https://evil.test'})]),{name:'TypeError'});
  assert.equal(JSON.stringify([existing,incoming]),before);
  for(const invalid of [null,{},'entries',new Set()])assert.throws(()=>mergeEntries(invalid,[]),{name:'TypeError'});
});

test('deduplication keeps first context and enriches absent identity metadata',()=>{
  assert.equal(typeof mergeEntries,'function');
  const first=source({savedAt:'2026-10-24T00:00:00Z'});
  const later=source({collection:'catalog',blob:'b'.repeat(40),sha256:'c'.repeat(64),bytes:42,observedAt:'2026-10-25T00:00:00Z'});
  assert.deepEqual(mergeEntries([first],[later]),[{...first,blob:'b'.repeat(40),sha256:'c'.repeat(64),bytes:42}]);
  assert.deepEqual(mergeEntries([search()],[search({query:' SIDE24 ',path:' coefficients/ ',savedAt:'2026-10-24T00:00:00Z'})]),[search()]);
  assert.deepEqual(parseReadingList(documentOf([first,later])).entries,[{...first,blob:'b'.repeat(40),sha256:'c'.repeat(64),bytes:42}]);
});

test('conflicting byte, Git blob or SHA-256 identity metadata rejects the entire merge',()=>{
  assert.equal(typeof mergeEntries,'function');
  for(const [field,a,b] of [['bytes',0,1],['blob','a'.repeat(40),'b'.repeat(40)],['sha256','a'.repeat(64),'b'.repeat(64)]]) {
    const first=source({[field]:a}),conflict=source({[field]:b}),existing=[first],before=JSON.stringify(existing);
    assert.throws(()=>mergeEntries(existing,[search(),conflict]),/conflict/i);
    assert.equal(JSON.stringify(existing),before);
    assert.throws(()=>parseReadingList(documentOf([first,conflict])),/conflict/i);
    assert.throws(()=>readingListJSON([first,conflict]),/conflict/i);
  }
});

test('capacity counts unique entries and overflow is atomic',()=>{
  assert.equal(typeof mergeEntries,'function');
  const full=Array.from({length:50},(_,i)=>source({path:`docs/${i}.md`})),before=JSON.stringify(full);
  assert.equal(mergeEntries(full,[full[0],full[49]]).length,50);
  assert.throws(()=>mergeEntries(full,[search()]),{name:'RangeError'});assert.equal(JSON.stringify(full),before);
  assert.equal(parseReadingList(documentOf([...full,full[0]])).entries.length,50);
  assert.throws(()=>parseReadingList(documentOf([...full,search()])),{name:'RangeError'});
  assert.throws(()=>readingListJSON([...full,search()]),{name:'RangeError'});
});

test('exported JSON carries only strict schema data and roundtrips identity types',()=>{
  assert.equal(typeof readingListJSON,'function');
  const entries=[source({blob:'b'.repeat(40),sha256:'c'.repeat(64),bytes:42}),search({query:' Δ & <script> ',path:' docs/ '})];
  const output=readingListJSON(entries),parsed=JSON.parse(output);
  assert.deepEqual(Object.keys(parsed),['schema','entries']);
  assert.equal(parsed.schema,schema);
  assert.deepEqual(parsed.entries,[entries[0],search({query:'Δ & <script>',path:'docs/'})]);
  assert.deepEqual(parseReadingList(output),parsed);
  assert.throws(()=>readingListJSON([source({certified:true})]),{name:'TypeError'});
});

test('export rejects oversized identities instead of creating an unimportable file',()=>{
  const oversized=source({path:'docs/'+'漢'.repeat(45000)+'.md'});
  assert.throws(()=>readingListJSON([oversized]),{name:'RangeError'});
  assert.throws(()=>mergeEntries([],[oversized]),{name:'RangeError'});
});

test('combined UTF-8 export size rejects additions atomically even below entry capacity',()=>{
  const entries=Array.from({length:45},(_,i)=>search({query:'漢'.repeat(498)+String(i).padStart(2,'0'),repository:'',path:'漢'.repeat(500)}));
  const existing=entries.slice(0,40),incoming=entries.slice(40),before=JSON.stringify([existing,incoming]);
  const exportable=readingListJSON(existing);
  assert.ok(Buffer.byteLength(exportable)<=131072);
  assert.deepEqual(parseReadingList(exportable).entries,existing);
  assert.throws(()=>mergeEntries(existing,incoming),{name:'RangeError'});
  assert.equal(JSON.stringify([existing,incoming]),before);
  // Deduplicates add no bytes and remain legal at a high byte count.
  assert.deepEqual(mergeEntries(existing,[existing[0]]),existing);
});
