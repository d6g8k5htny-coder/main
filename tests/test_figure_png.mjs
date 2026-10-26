import test from 'node:test';
import assert from 'node:assert/strict';
import { deflateSync } from 'node:zlib';

let feature = {};
try { feature = await import('../docs/site/figure-png.mjs'); } catch {}
function api(name) { assert.equal(typeof feature[name], 'function', `${name} is available`); return feature[name]; }
const timestamp = '2026-10-26T00:30:00.000Z';
const metadata = {schema: 'universal-law/curvature-teaching-figure/v1', mode: 'teaching_model',
  params: {s: -2, R: 1}, generated_at: timestamp, source: {sha256: 'a'.repeat(64)},
  verification: 'not_performed', limits: 'Illustrative; no theorem computation.'};
const record = () => ({svg: '<svg>validated captioned content</svg>', metadata: structuredClone(metadata),
  citation: 'Universal Law. λ = −3; teaching figure; not mathematical result acceptance.', width: 600, height: 535});
function crc(bytes) {
  let c = 0xffffffff;
  for (const b of bytes) { c ^= b; for (let j = 0; j < 8; j++) c = (c >>> 1) ^ ((c & 1) ? 0xedb88320 : 0); }
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data = new Uint8Array()) {
  const b = Buffer.alloc(data.length + 12); b.writeUInt32BE(data.length); b.write(type, 4, 'ascii');
  b.set(data, 8); b.writeUInt32BE(crc(b.subarray(4, -4)), b.length - 4); return b;
}
function png(width = 1, height = 1) {
  const header = Buffer.alloc(13); header.writeUInt32BE(width); header.writeUInt32BE(height, 4);
  header[8] = 8; header[9] = 6;
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]), chunk('IHDR', header),
    chunk('IDAT', deflateSync(Buffer.from([0, 20, 100, 200, 255]))), chunk('IEND')]);
}
function chunks(bytes) {
  const b = Buffer.from(bytes); const found = []; let p = 8;
  while (p < b.length) {
    const n = b.readUInt32BE(p), type = b.toString('ascii', p + 4, p + 8);
    assert.equal(b.readUInt32BE(p + 8 + n), crc(b.subarray(p + 4, p + 8 + n)), `valid ${type} CRC`);
    found.push({type, data: b.subarray(p + 8, p + 8 + n)}); p += n + 12;
  }
  assert.equal(p, b.length); return found;
}
function textChunks(bytes) {
  const result = {};
  for (const {type, data} of chunks(bytes)) if (type === 'iTXt') {
    const split = data.indexOf(0), key = data.toString('latin1', 0, split);
    assert.deepEqual([...data.subarray(split + 1, split + 5)], [0,0,0,0]);
    result[key] = data.subarray(split + 5).toString('utf8');
  }
  return result;
}

test('PNG embeds exact structured figure metadata and UTF-8 citation without changing image chunks', () => {
  const input = png(), output = api('addFigurePNGMetadata')(input, record(), {width: 1, height: 1});
  const before = chunks(input), after = chunks(output);
  assert.deepEqual(after.filter(c => c.type !== 'iTXt'), before);
  assert.equal(after.at(-1).type, 'IEND');
  const text = textChunks(output);
  assert.deepEqual(JSON.parse(text['Universal Law figure']), metadata);
  assert.equal(text.Description, record().citation);
  assert.deepEqual(JSON.parse(text['Universal Law raster']), {schema: 'universal-law/teaching-png/v1',
    width: 1, height: 1, scale: 2, pixels: 'browser-rendered', mathematical_verification: 'not_performed'});
});

test('PNG refuses malformed signature, truncated or oversized chunks, corrupt CRC and trailing bytes', () => {
  const add = api('addFigurePNGMetadata');
  const badSignature = png(); badSignature[0] = 0;
  const badCRC = png(); badCRC[29] ^= 1;
  const badLength = png(); badLength.writeUInt32BE(0xffffffff, 8);
  for (const bytes of [badSignature, badCRC, badLength, png().subarray(0, -1), Buffer.concat([png(), Buffer.from([0])])]) {
    assert.throws(() => add(bytes, record(), {width:1,height:1}), /PNG|chunk|CRC/i);
  }
});

test('PNG refuses missing or misplaced required chunks, mismatched dimensions and preexisting text metadata', () => {
  const add = api('addFigurePNGMetadata'), cs = chunks(png()), signature = png().subarray(0, 8);
  const join = parts => Buffer.concat([signature, ...parts.map(c => chunk(c.type, c.data))]);
  for (const bytes of [join([cs[0],cs[2]]), join([cs[1],cs[0],cs[2]]),
    join([cs[0],cs[0],cs[1],cs[2]]), join([cs[0],{type:'tEXt',data:Buffer.from('Description\0other')},cs[1],cs[2]])]) {
    assert.throws(() => add(bytes, record(), {width:1,height:1}), /PNG|chunk|metadata/i);
  }
  assert.throws(() => add(png(), record(), {width:2,height:1}), /dimension/i);
});

test('PNG refuses unbounded or unsupported metadata, citations and target dimensions', () => {
  const add = api('addFigurePNGMetadata');
  for (const change of [r => {r.metadata.verification='passed'}, r => {r.metadata.mode='theorem'},
    r => {r.citation='x'.repeat(66000)}, r => {r.citation='bad\0text'}, r => {r.metadata.limits='x'.repeat(66000)}]) {
    const r=record();change(r);assert.throws(() => add(png(),r,{width:1,height:1}));
  }
  for (const size of [{width:0,height:1},{width:1,height:Infinity},{width:1.5,height:1},{width:5000,height:1}]) {
    assert.throws(() => add(png(),record(),size), /dimension/i);
  }
});

function harness({capture = () => record(), render, download, now = () => timestamp} = {}) {
  let resolve, reject; const rendered = [], downloaded = [], states = [];
  const controller = api('createFigurePNGController')({capture, now,
    render: render || ((r, {signal}) => { rendered.push({r,signal}); return new Promise((yes,no)=>{resolve=yes;reject=no;}); }),
    download: download || (blob => downloaded.push(blob)), onChange: state => states.push({...state})});
  controller.refresh(true);
  return {controller, rendered, downloaded, states, resolve: value => resolve(value), reject: e => reject(e)};
}
const result = () => new Blob([png()],{type:'image/png'});

test('controller downloads once only after asynchronous preparation and final unchanged recapture', async () => {
  const seen=[];const h=harness({capture: t=>{seen.push(t);return record();}});
  const work=h.controller.export();
  assert.equal(h.controller.state().pending,true);assert.equal(h.downloaded.length,0);
  h.resolve(result());assert.equal(await work,true);
  assert.deepEqual(seen,[timestamp,timestamp]);assert.equal(h.downloaded.length,1);
  assert.equal(h.controller.state().pending,false);assert.match(h.controller.state().status,/PNG download started/);
});

test('controller coalesces repeated activation while a preparation is pending', async () => {
  const h=harness();const first=h.controller.export();assert.equal(await h.controller.export(),false);
  assert.equal(h.rendered.length,1);h.resolve(result());assert.equal(await first,true);assert.equal(h.downloaded.length,1);
});

test('changed or tampered final SVG, metadata or citation refuses download', async () => {
  for(const field of ['svg','metadata','citation']) {
    let call=0;const h=harness({capture:()=>{const r=record();if(call++)r[field]=field==='metadata'?{...r.metadata,params:{s:2,R:1}}:'changed';return r;}});
    const work=h.controller.export();h.resolve(result());assert.equal(await work,false);
    assert.equal(h.downloaded.length,0);assert.match(h.controller.state().status,/changed|refused/i);
  }
});

test('control revision cancels work even after a change away and back', async () => {
  const h=harness();const work=h.controller.export();h.controller.refresh(true);h.controller.refresh(true);
  assert.equal(h.rendered[0].signal.aborted,true);h.resolve(result());assert.equal(await work,false);
  assert.equal(h.downloaded.length,0);assert.equal(h.controller.state().ready,true);
});

test('late failure from an invalidated operation cannot erase newer successful state', async () => {
  let firstReject;let calls=0;const h=harness({render:()=>++calls===1?new Promise((_,no)=>{firstReject=no}):Promise.resolve(result())});
  const old=h.controller.export();h.controller.refresh(true);assert.equal(await h.controller.export(),true);
  const status=h.controller.state().status;firstReject(new Error('late decode failure'));assert.equal(await old,false);
  assert.equal(h.controller.state().status,status);assert.equal(h.downloaded.length,1);
});

test('capture, preparation, null result and download errors stay explicit and allow retry', async () => {
  for(const stage of ['capture','render','null','download']) {
    let fail=true;const h=harness({capture:()=>{if(stage==='capture'&&fail)throw new Error('unsupported capture');return record()},
      render:async()=>{if(stage==='render'&&fail)throw new Error('decode failure');return stage==='null'&&fail?null:result()},
      download:()=>{if(stage==='download'&&fail)throw new Error('blocked download')}});
    assert.equal(await h.controller.export(),false);assert.equal(h.controller.state().pending,false);
    assert.match(h.controller.state().status,/PNG.*(?:could not|unavailable|refused)/i);
    fail=false;assert.equal(await h.controller.export(),true);
  }
});

test('unready controller cannot prepare or download a figure', async () => {
  const h=harness();h.controller.refresh(false);assert.equal(await h.controller.export(),false);
  assert.equal(h.rendered.length,0);assert.equal(h.downloaded.length,0);assert.equal(h.controller.state().ready,false);
});

test('PNG refuses an empty image-data stream even when chunk CRCs and dimensions are valid', () => {
  const cs = chunks(png());
  const empty = Buffer.concat([png().subarray(0,8), chunk('IHDR',cs[0].data), chunk('IDAT'), chunk('IEND')]);
  assert.throws(() => api('addFigurePNGMetadata')(empty,record(),{width:1,height:1}),/image.data/i);
});

function rasterEnvironment({context = true, imageFailure = false, hold = false, width = 600,
  nullBlob = false, encodeFailure = false} = {}) {
  const made = [], drawn = [];
  const document = {createElement(tag) {
    if (tag === 'canvas') {
      const node={tag,width:0,height:0,getContext:()=>context?{drawImage:(...args)=>drawn.push(args)}:null,
        toBlob(callback) {if(encodeFailure)throw new Error('encoder failed');queueMicrotask(()=>callback(nullBlob?null:new Blob([png(1200,1070)],{type:'image/png'})));}};
      made.push(node);return node;
    }
    const node={tag,naturalWidth:width,naturalHeight:535,removed:false,removeAttribute(name){assert.equal(name,'src');this.removed=true;}};
    Object.defineProperty(node,'src',{set(value) {
      assert.match(value,/^data:image\/svg\+xml;charset=utf-8,/);
      assert.equal(decodeURIComponent(value.split(',').slice(1).join(',')),record().svg);
      if(!hold)queueMicrotask(()=>imageFailure?this.onerror?.():this.onload?.());
    }});
    made.push(node);return node;
  }};
  return {document,made,drawn};
}

test('rasterizer refuses missing canvas context before image work and releases canvas storage', async () => {
  const env=rasterEnvironment({context:false});
  await assert.rejects(api('rasterizeTeachingPNG')(record(),{document:env.document}),/canvas.*unavailable/i);
  assert.equal(env.made.length,1);assert.equal(env.made[0].width,0);assert.equal(env.made[0].height,0);
});

test('rasterizer image decode, dimension, encoder and null-output failures release all resources', async () => {
  for(const options of [{imageFailure:true},{width:500},{encodeFailure:true},{nullBlob:true}]) {
    const env=rasterEnvironment(options);
    await assert.rejects(api('rasterizeTeachingPNG')(record(),{document:env.document}),/decode|dimension|encoder|encoding/i);
    const [canvas,image]=env.made;assert.equal(canvas.width,0);assert.equal(canvas.height,0);
    assert.equal(image.removed,true);assert.equal(image.onload,null);assert.equal(image.onerror,null);
  }
});

test('rasterizer is bounded by its timeout and abort signal, with no late image callbacks', async () => {
  const timeout=rasterEnvironment({hold:true});
  await assert.rejects(api('rasterizeTeachingPNG')(record(),{document:timeout.document,timeout:5}),/timed out/i);
  assert.equal(timeout.made[1].removed,true);assert.equal(timeout.made[0].width,0);
  const abort=rasterEnvironment({hold:true}),controller=new AbortController();
  const work=api('rasterizeTeachingPNG')(record(),{document:abort.document,signal:controller.signal});
  controller.abort();await assert.rejects(work,/canceled/i);
  assert.equal(abort.made[1].onload,null);assert.equal(abort.made[1].removed,true);assert.equal(abort.made[0].width,0);
});
