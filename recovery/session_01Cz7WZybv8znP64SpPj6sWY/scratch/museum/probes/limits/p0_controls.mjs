// Q4 limits lens -- CONTROL probe (expected to PASS): what boundedFetch at cd32303 does enforce.
// No network: every fetcher is an in-process stub returning a WHATWG Response.
// Usage: node p0_controls.mjs <path-to-cd32303-tree>
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root = process.argv[2];
const museum = await import(pathToFileURL(path.join(root, 'docs/site/museum.mjs')));
globalThis.fetch = async () => { throw Error('network forbidden in probe'); };
const out = (k, v) => console.log(`${k}: ${v}`);
const settle = async p => { const t = Date.now(); try { const r = await p; return `resolved ok=${r.ok} bytes=${r.bytes.byteLength} after ${Date.now() - t}ms`; } catch (e) { return `rejected "${e.message}" after ${Date.now() - t}ms`; } };

// 1. headers never arrive (fetcher ignores the signal entirely)
let sig1;
out('C1 stalled headers, signal-ignoring fetcher, timeoutMs=150',
  await settle(museum.boundedFetch('u', async (_u, o) => { sig1 = o.signal; return new Promise(() => {}); }, 1024, 150)));
out('C1 signal.aborted', sig1.aborted);

// 2. declared Content-Length above cap is refused before reading the body
let pulled2 = 0;
const r2 = new Response(new ReadableStream({pull(c) { pulled2++; c.enqueue(new Uint8Array(10)); }}), {headers: {'content-length': '999999'}});
out('C2 content-length 999999 > cap 1024', await settle(museum.boundedFetch('u', async () => r2, 1024, 1000)));

// 3. Content-Length lies (says 10) but body streams past the cap -> streaming cap catches it
let pulled3 = 0;
const r3 = new Response(new ReadableStream({pull(c) { pulled3++; c.enqueue(new Uint8Array(256)); }}), {headers: {'content-length': '10'}});
out('C3 lying content-length=10, 256-byte chunks, cap 1024', await settle(museum.boundedFetch('u', async () => r3, 1024, 1000)));
out('C3 chunks pulled (cap/256 + 1 expected ~5)', pulled3);

// 4. non-numeric Content-Length does not bypass the streaming cap
const r4 = new Response(new ReadableStream({pull(c) { c.enqueue(new Uint8Array(512)); }}), {headers: {'content-length': 'abc'}});
out('C4 content-length "abc", cap 1024', await settle(museum.boundedFetch('u', async () => r4, 1024, 1000)));

// 5. the startMuseum refusal text for a timed-out source is not a "verified" display
out('C5 invalid bounds refused', await settle(museum.boundedFetch('u', async () => new Response('x'), 0, 10)));
