import {parseStrictJSON} from "./reader-json.mjs?site-release=730b0b4d5c93a9e5aa5eee3feddd4070e9480db86af489c2982bfa0e39fca706";
// Reader-supplied reading notes carry identities, never scientific status.
// This module deliberately has no storage, network or DOM access.
export const SCHEMA='universal-law-reading-list/v1';
export const MAX_ENTRIES=50;
export const MAX_FILE_BYTES=131072;

const owns=(value,key)=>Object.prototype.hasOwnProperty.call(value,key);
const collections=new Set(['current','catalog']);
const repositories=new Set(['d6g8k5htny-coder/main','d6g8k5htny-coder/Math-']);
const hex40=/^[a-f0-9]{40}$/;
const hex64=/^[a-f0-9]{64}$/;
const controls=/[\u0000-\u001f\u007f-\u009f]/;
const commonKeys=['kind','collection','savedAt','observedAt'];
const sourceKeys=new Set([...commonKeys,'repository','commit','path','blob','sha256','bytes']);
const searchKeys=new Set([...commonKeys,'query','repository','path']);

function fail(message) { throw new TypeError(message); }

function record(value,allowed) {
  if(value===null||typeof value!=='object'||![Object.prototype,null].includes(Object.getPrototypeOf(value)))fail('Expected a plain JSON data record.');
  for(const key of Reflect.ownKeys(value)) {
    const descriptor=Object.getOwnPropertyDescriptor(value,key);
    if(typeof key!=='string'||!allowed.has(key)||!descriptor.enumerable||!owns(descriptor,'value'))fail('Unexpected or non-data property.');
  }
}

function text(value,label) {
  if(typeof value!=='string'||controls.test(value)||!value.isWellFormed())fail(`${label} must be well-formed text without control characters.`);
  return value;
}

function timestamp(value,label) {
  text(value,label);
  const match=/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,3}))?Z$/.exec(value);
  if(!match)fail(`${label} must be an ISO UTC timestamp.`);
  const date=new Date(value),canonical=`${match[1]}.${(match[2]||'').padEnd(3,'0')}Z`;
  if(!Number.isFinite(date.getTime())||date.toISOString()!==canonical)fail(`${label} must name a real UTC date and time.`);
  return value;
}

function gitPath(value) {
  text(value,'Source path');
  if(!value||value.includes('\\')||/^[a-z][a-z0-9+.-]*:/i.test(value)||value.split('/').some(part=>!part||part==='.'||part==='..'))fail('Source path must be a canonical relative Git path.');
  // Percent sequences are literal filename text. Decoding them would change
  // the imported Git identity and could introduce dot or slash segments.
  return value;
}

export function validateEntry(value) {
  // Check descriptors before reading kind, so getters never run here.
  record(value,new Set([...sourceKeys,...searchKeys]));
  const kind=value.kind;
  if(!owns(value,'kind')||!['source','search'].includes(kind))fail('Unknown reading-list entry kind.');
  record(value,kind==='source'?sourceKeys:searchKeys);
  for(const key of ['collection','savedAt','repository','path',kind==='source'?'commit':'query'])if(!owns(value,key))fail(`Missing ${key}.`);
  if(!collections.has(value.collection))fail('Unknown source collection.');
  const normalized={kind,collection:value.collection};
  if(kind==='source') {
    if(!repositories.has(value.repository))fail('Unknown source repository.');
    if(typeof value.commit!=='string'||!hex40.test(value.commit))fail('Source commit must be 40 lowercase hexadecimal digits.');
    normalized.repository=value.repository;normalized.commit=value.commit;normalized.path=gitPath(value.path);
  } else {
    for(const key of ['query','path']) {
      text(value[key],`Search ${key}`);
      if(value[key].length>500)fail(`Search ${key} exceeds 500 characters.`);
    }
    if(!['','main','Math-'].includes(value.repository))fail('Unknown search repository.');
    if(value.collection==='current'&&(value.repository!==''||value.path!==''))fail('Current searches do not have catalog repository or path filters.');
    normalized.query=value.query.trim();normalized.repository=value.repository;normalized.path=value.path.trim();
  }
  normalized.savedAt=timestamp(value.savedAt,'savedAt');
  if(owns(value,'observedAt'))normalized.observedAt=timestamp(value.observedAt,'observedAt');
  if(kind==='source') {
    for(const [key,pattern] of [['blob',hex40],['sha256',hex64]])if(owns(value,key)) {
      if(typeof value[key]!=='string'||!pattern.test(value[key]))fail(`Invalid ${key} identity.`);
      normalized[key]=value[key];
    }
    if(owns(value,'bytes')) {
      if(!Number.isSafeInteger(value.bytes)||value.bytes<0)fail('bytes must be a nonnegative safe integer.');
      normalized.bytes=value.bytes;
    }
  }
  return normalized;
}

function normalizedKey(entry) {
  // Tuple encoding avoids collisions from delimiters in literal paths/queries.
  return JSON.stringify(entry.kind==='source'?['source',entry.repository,entry.commit,entry.path]:['search',entry.collection,entry.query,entry.repository,entry.path]);
}

export function entryKey(entry) { return normalizedKey(validateEntry(entry)); }

function serialize(entries) {
  return JSON.stringify({schema:SCHEMA,entries},null,2)+'\n';
}

export function mergeEntries(existing,incoming) {
  if(!Array.isArray(existing)||!Array.isArray(incoming))fail('Reading-list entries must be arrays.');
  const merged=[],byKey=new Map();
  for(const value of [...existing,...incoming]) {
    const entry=validateEntry(value),key=normalizedKey(entry),previous=byKey.get(key);
    if(previous) {
      if(entry.kind==='source')for(const field of ['blob','sha256','bytes']) {
        if(owns(previous,field)&&owns(entry,field)&&previous[field]!==entry[field])fail(`Conflicting ${field} metadata for one source identity.`);
        if(!owns(previous,field)&&owns(entry,field))previous[field]=entry[field];
      }
    } else {
      if(merged.length>=MAX_ENTRIES)throw new RangeError(`Reading list cannot exceed ${MAX_ENTRIES} unique entries.`);
      merged.push(entry);byKey.set(key,entry);
    }
  }
  // The exported form, including its formatting, must fit the import ceiling.
  // Check after deduplication so an additive no-op never spends capacity.
  if(new TextEncoder().encode(serialize(merged)).byteLength>MAX_FILE_BYTES)throw new RangeError('Reading list exceeds the 128 KiB export/import limit.');
  return merged;
}

export function parseReadingList(input) {
  if(typeof input!=='string')fail('Reading-list JSON must be text.');
  if(new TextEncoder().encode(input).byteLength>MAX_FILE_BYTES)throw new RangeError('Reading-list file exceeds 128 KiB.');
  const data=parseStrictJSON(input);
  record(data,new Set(['schema','entries']));
  if(!owns(data,'schema')||!owns(data,'entries')||data.schema!==SCHEMA)fail('Unsupported reading-list schema.');
  return {schema:SCHEMA,entries:mergeEntries([],data.entries)};
}

export function readingListJSON(entries) {
  return serialize(mergeEntries([],entries));
}

export function sourceHref(value) {
  const entry=validateEntry(value);
  if(entry.kind!=='source')fail('A source entry is required.');
  const path=entry.path.split('/').map(part=>encodeURIComponent(part).replace(/[!'()*]/g,char=>'%'+char.charCodeAt(0).toString(16).toUpperCase())).join('/');
  return `https://github.com/${entry.repository}/blob/${entry.commit}/${path}`;
}

export function searchHref(value,baseURL) {
  const entry=validateEntry(value);
  if(entry.kind!=='search')fail('A search entry is required.');
  const url=new URL('workspace.html',baseURL);
  url.search='';
  if(entry.collection==='current') {
    url.hash='library-current';
    if(entry.query)url.searchParams.set('source-q',entry.query);
  } else {
    url.hash='inventory';
    for(const [key,value] of [['q',entry.query],['repository',entry.repository],['path',entry.path]])if(value)url.searchParams.set(key,value);
  }
  return url.href;
}
