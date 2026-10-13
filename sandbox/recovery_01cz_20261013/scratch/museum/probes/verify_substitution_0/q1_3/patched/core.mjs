// Read-only public source validation. No acceptance decisions or write APIs.
export const hex40 = /^[0-9a-f]{40}$/;
export const hex64 = /^[0-9a-f]{64}$/;
export function safeSourceURL(row) {
  if (!['Math-', 'main'].includes(row.repository) || !hex40.test(row.commit) ||
      !hex64.test(row.sha256) || typeof row.path !== 'string' ||
      row.path.split('/').some(p => !p || p === '.' || p === '..') ||
      /[\\\x00-\x1f]/.test(row.path)) throw new Error('Invalid public source identity');
  return `https://github.com/d6g8k5htny-coder/${row.repository}/blob/${row.commit}/${row.path.split('/').map(encodeURIComponent).join('/')}`;
}
export async function verifiedBytes(pin, fetcher = fetch) {
  if (!pin || !hex64.test(pin.sha256) || !Number.isSafeInteger(pin.bytes) || pin.bytes < 0)
    throw new Error('Missing source digest or byte count');
  const target = pin.url || pin.raw_url;
  if (typeof target !== 'string' || (!/^https:\/\/raw\.githubusercontent\.com\/d6g8k5htny-coder\/(main|Math-|query-)\/[0-9a-f]{40}\//.test(target) && !/^(?:\.\.\/public-math\/sources(?:-\d{2})?\.json|status\.json|observations\.json)$/.test(target)))
    throw new Error('Source must be a pinned public URL or a declared local data file');
  if (target.startsWith('https:')) {
    if (!/^d6g8k5htny-coder\/(main|Math-|query-)$/.test(pin.repository) || !hex40.test(pin.commit) || typeof pin.path !== 'string' || pin.path.split('/').some(p=>!p||p==='.'||p==='..')) throw new Error('Invalid displayed source identity');
    const expected = `https://raw.githubusercontent.com/${pin.repository}/${pin.commit}/${pin.path.split('/').map(encodeURIComponent).join('/')}`;
    if (target !== expected) throw new Error('Displayed identity does not match source URL');
  }
  const response = await fetcher(target, {credentials:'omit', redirect:'error'});
  if (!response.ok) throw new Error(`Source unavailable (${response.status})`);
  const data = new Uint8Array(await response.arrayBuffer());
  if (data.byteLength !== pin.bytes) throw new Error('Source byte count mismatch');
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)), x=>x.toString(16).padStart(2,'0')).join('');
  if (digest !== pin.sha256) throw new Error('Source SHA-256 mismatch');
  return data;
}
export async function verifiedJSON(pin, fetcher = fetch) {
  return JSON.parse(new TextDecoder('utf-8', {fatal:true}).decode(await verifiedBytes(pin, fetcher)));
}
export function validateStatus(status) {
  if (status.scientific_status_authority !== false || !Array.isArray(status.sections)) throw new Error('Invalid status export');
  const keys = new Set();
  for (const section of status.sections) {
    if (!['accept','amend','engineering'].includes(section.key) || keys.has(section.key) || !Array.isArray(section.rows) || !Array.isArray(section.headers)) throw new Error('Unrecognized status section');
    keys.add(section.key);
    if (section.rows.length !== section.count || status.counts[section.key] !== section.count || section.rows.some(row=>!Array.isArray(row)||row.length!==section.headers.length||row.some(c=>typeof c!=='string'))) throw new Error('Status count/schema mismatch');
  }
  if (keys.size !== 3) throw new Error('Incomplete status export');
  return status;
}
export function validateCoefficient(data) {
  if(data.scientific_acceptance!==false || data.object!=='SIDE24-COEFFICIENT-D23-20260924-v1') throw new Error('Unexpected coefficient scope');
  for(const d of ['2','3']) {
    const {lower,upper}=data.dimensions[d];
    if(!/^0\.\d{20}$/.test(lower)||!/^0\.\d{20}$/.test(upper)||BigInt(lower.replace('.',''))>=BigInt(upper.replace('.',''))) throw new Error('Invalid exact decimal interval');
  }
  return data;
}
