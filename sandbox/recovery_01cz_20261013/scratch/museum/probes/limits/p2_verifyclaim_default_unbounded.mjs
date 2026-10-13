// Q4 limits lens -- FINDING probe: the size/time bound is a property of startMuseum's private
// cachedFetch, not of the exported verification API. museum.mjs:111 exports
// verifyClaim(claim, fetcher=fetch); with its default (or any raw fetcher, as the integration test at
// tests/test_museum_frontend.mjs:150 does) it calls core.mjs verifiedBytes, which does
// `await response.arrayBuffer()` (core.mjs:24) with no byte cap, no AbortController and no deadline.
// No network: globalThis.fetch is replaced by an in-process stub before each call.
// Usage: node p2_verifyclaim_default_unbounded.mjs <path-to-cd32303-tree>
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const root = process.argv[2];
const museum = await import(pathToFileURL(path.join(root, 'docs/site/museum.mjs')));

const body = 'Pinned proof.';
const sha = createHash('sha256').update(body).digest('hex');
const c = 'a'.repeat(40);
const proof = {repository: 'd6g8k5htny-coder/Math-', path: 'proof.md', commit: c, blob: 'b'.repeat(40), bytes: Buffer.byteLength(body), sha256: sha,
  url: `https://raw.githubusercontent.com/d6g8k5htny-coder/Math-/${c}/proof.md`, html_url: `https://github.com/d6g8k5htny-coder/Math-/blob/${c}/proof.md`};
const claim = {id: 'x', proof, review: null, replay: {command: null, url: null, notice: 'n'}};

// A. Oversized response: 64 MiB streamed in 1 MiB chunks for a 13-byte pin.
{
  let streamed = 0, signalSeen = 'none';
  globalThis.fetch = async (_u, init) => {
    signalSeen = init && 'signal' in init ? String(init.signal) : 'no signal passed';
    let n = 0;
    return new Response(new ReadableStream({pull(ctl) { if (n++ >= 64) { ctl.close(); return; } streamed += 1 << 20; ctl.enqueue(new Uint8Array(1 << 20)); }}));
  };
  const t = Date.now();
  try { await museum.verifyClaim(claim); console.log('A: resolved (unexpected)'); }
  catch (e) { console.log(`A default fetcher: rejected "${e.message}" only after buffering ${streamed / (1 << 20)} MiB (pin.bytes=${proof.bytes}); init.signal: ${signalSeen}; ${Date.now() - t}ms`); }
  // Same response through boundedFetch (the cachedFetch path) for contrast.
  let streamed2 = 0, n2 = 0;
  const r = new Response(new ReadableStream({pull(ctl) { if (n2++ >= 64) { ctl.close(); return; } streamed2 += 1 << 20; ctl.enqueue(new Uint8Array(1 << 20)); }}));
  try { await museum.boundedFetch(proof.url, async () => r); } catch (e) { console.log(`A boundedFetch contrast: rejected "${e.message}" after ${streamed2 / (1 << 20)} MiB pulled`); }
}

// B. Stalled body: headers arrive, body never completes. boundedFetch's default deadline is 20 s;
// observe the default verifyClaim path for 25 s.
{
  globalThis.fetch = async () => new Response(new ReadableStream({pull() { return new Promise(() => {}); }}));
  const t = Date.now();
  const state = await Promise.race([
    museum.verifyClaim(claim).then(() => 'resolved', e => `rejected "${e.message}"`),
    new Promise(r => setTimeout(() => r('STILL PENDING'), 25000)),
  ]);
  console.log(`B default fetcher, stalled body: ${state} after ${Date.now() - t}ms (boundedFetch default deadline is 20000ms)`);
}
process.exit(0);
