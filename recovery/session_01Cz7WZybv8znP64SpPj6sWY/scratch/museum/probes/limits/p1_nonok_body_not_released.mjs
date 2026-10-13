// Q4 limits lens -- FINDING probe: boundedFetch (museum.mjs:22) returns early on !response.ok
// without aborting its controller or cancelling the response body. Every other exit path
// (byte cap, deadline) calls cancel(); the non-OK path leaves the body stream open.
// No network: in-process stub fetcher.
// Usage: node p1_nonok_body_not_released.mjs <path-to-cd32303-tree>
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root = process.argv[2];
const museum = await import(pathToFileURL(path.join(root, 'docs/site/museum.mjs')));
globalThis.fetch = async () => { throw Error('network forbidden in probe'); };

for (const status of [404, 500, 503]) {
  let cancelled = false, pulls = 0, signal;
  // An error page whose body keeps streaming (e.g. a large/slow HTML error page from a CDN).
  const body = new ReadableStream({pull(c) { pulls++; c.enqueue(new Uint8Array(1024)); }, cancel() { cancelled = true; }}, {highWaterMark: 0});
  const response = new Response(body, {status});
  const result = await museum.boundedFetch('https://raw.githubusercontent.com/x', async (_u, o) => { signal = o.signal; return response; }, 4096, 200);
  await new Promise(r => setTimeout(r, 400)); // well past the 200 ms deadline
  console.log(`status=${status} result.ok=${result.ok} result.status=${result.status} | after deadline: signal.aborted=${signal.aborted} body.cancelled=${cancelled} body.locked=${response.body.locked} bodyUsed=${response.bodyUsed}`);
}

// Contrast: the byte-cap path in the same function does release the body.
{
  let cancelled = false, signal;
  const response = new Response(new ReadableStream({pull(c) { c.enqueue(new Uint8Array(1024)); }, cancel() { cancelled = true; }}));
  try { await museum.boundedFetch('u', async (_u, o) => { signal = o.signal; return response; }, 4096, 200); } catch (e) { console.log(`contrast (200, over cap): rejected "${e.message}" signal.aborted=${signal.aborted} body.cancelled=${cancelled}`); }
}
