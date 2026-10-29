// Read-only export of one conditional route; never executes a source or fetches the network.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash,webcrypto} from 'node:crypto';
import {projectVerified,verifyProjection} from '../docs/site/conditionals.mjs';

function readBounded(filename,limit){
  const fd=fs.openSync(filename,'r');
  try{
    const stat=fs.fstatSync(fd);
    if(!stat.isFile()||stat.size>limit)throw Error('Input is not a bounded regular file');
    const raw=Buffer.alloc(limit+1);let total=0;
    while(total<raw.length){const n=fs.readSync(fd,raw,total,raw.length-total,null);if(n===0)break;total+=n;}
    if(total>limit)throw Error('Input exceeds byte limit');
    return raw.subarray(0,total);
  }finally{fs.closeSync(fd);}
}
function argumentsFor(argv){
  const options={root:path.resolve(fileURLToPath(new URL('..',import.meta.url)))};
  const seen=new Set();
  for(let i=0;i<argv.length;i+=2){
    const name=argv[i];
    if(!['--root','--source','--check'].includes(name)||seen.has(name)||!argv[i+1]||argv[i+1].startsWith('--'))
      throw Error('Use --source FILE [--root CHECKOUT] [--check PROJECTION.json]');
    options[name.slice(2)]=argv[i+1];seen.add(name);
  }
  if(!options.source)throw Error('--source FILE is required');
  return options;
}
async function main(){
  const options=argumentsFor(process.argv.slice(2)),site=path.join(options.root,'docs/site');
  const config=JSON.parse(readBounded(path.join(site,'config.json'),32*1024)),pin=config.museum_json;
  if(pin?.url!=='museum.json'||typeof pin.sha256!=='string'||!/^[0-9a-f]{64}$/.test(pin.sha256)
    ||!Number.isSafeInteger(pin.bytes)||pin.bytes<1||pin.bytes>256*1024)throw Error('Invalid museum manifest identity');
  let raw;
  try{raw=readBounded(path.join(site,'museum.json'),pin.bytes);}
  catch(error){throw Error('Museum manifest unreadable: '+error.message);}
  if(raw.length!==pin.bytes||createHash('sha256').update(raw).digest('hex')!==pin.sha256)
    throw Error('Museum manifest byte identity mismatch');
  const manifest=JSON.parse(raw);
  if(manifest.schema_version!==1||manifest.scientific_status_authority!==false||!Array.isArray(manifest.claims))
    throw Error('Invalid museum manifest declarations');
  const claims=manifest.claims.filter(c=>c?.id==='cumulative-transfer-correction');
  if(claims.length!==1)throw Error('Conditional claim is missing or repeated');
  const source=new Uint8Array(readBounded(options.source,3272));
  const projected=await projectVerified(source,claims[0].proof,webcrypto);
  if(options.check)await verifyProjection(JSON.parse(readBounded(options.check,64*1024)),source,claims[0].proof,webcrypto);
  process.stdout.write(JSON.stringify(projected,null,2)+'\n');
}
try{await main();}catch(error){process.stderr.write('REFUSED: '+error.message+'\n');process.exitCode=1;}
