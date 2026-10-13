// Feed builder-emitted D1 pointers (from v_builder.py) to the cd32303 browser guard. No network.
import fs from 'node:fs';
globalThis.fetch = async u => { throw Error('NETWORK BLOCKED ' + u); };
const S = '<SCRATCH>/museum';
const Q = `${S}/probes/verify_substitution_0/q1_3`;
const museum = await import(`${S}/cd32303/docs/site/museum.mjs`);
const index = fs.readFileSync(`${S}/probes/verify_substitution_0/math/PROOF_INDEX.md`, 'utf8');
const status = fs.readFileSync(`${S}/cd32303/STATUS.md`, 'utf8');
for (const tag of ['issues6', 'issues9999']) {
  const m = JSON.parse(fs.readFileSync(`${Q}/manifest_d1_${tag}.json`, 'utf8'));
  let r; try { museum.validateBoundManifest(m, index, status); r = 'ACCEPTED'; } catch (e) { r = 'REFUSED (' + e.message + ')'; }
  console.log(`frontend guard on builder-emitted D1 review_url ${m.claims[11].review.review_url}: ${r}`);
}
