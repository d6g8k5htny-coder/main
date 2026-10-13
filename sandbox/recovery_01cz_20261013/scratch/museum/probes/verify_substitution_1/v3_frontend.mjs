// V3b: feed the builder-emitted d5-pin claim (one-entry open_specs edit) to the cd32303 frontend guard.
import fs from 'node:fs';
const S='<SCRATCH>/museum', T=`${S}/cd32303`, H=`${S}/probes/verify_substitution_1`;
globalThis.fetch=async u=>{throw Error('NETWORK BLOCKED '+u);};
const museum=await import(`${T}/docs/site/museum.mjs`);
const m=JSON.parse(fs.readFileSync(`${T}/docs/site/museum.json`,'utf8'));
m.claims[12]=JSON.parse(fs.readFileSync(`${H}/v3_d5pin_claim.json`,'utf8'));
try{museum.validateBoundManifest(m,fs.readFileSync(`${H}/fixture/Math-/PROOF_INDEX.md`,'utf8'),fs.readFileSync(`${T}/STATUS.md`,'utf8'));console.log('validateBoundManifest ACCEPTS builder-emitted d5-pin claim; review.pointer_only =',m.claims[12].review.pointer_only??false,'-> renderClaim label "Byte-verified review"');}
catch(e){console.log('validateBoundManifest REFUSES:',e.message);}
