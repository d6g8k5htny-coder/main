// Local rasterization of the existing validated, captioned teaching SVG.
// Callers supply those serializers' output; this module does not validate proofs
// or fetch sources. SVG and readable citations remain the accessible alternatives.
const MAX_PNG_BYTES = 24 * 1024 * 1024;
const MAX_TEXT_BYTES = 64 * 1024;
const SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
const encoder = new TextEncoder();
const crcTable = Array.from({length: 256}, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = (c >>> 1) ^ ((c & 1) ? 0xedb88320 : 0);
  return c >>> 0;
});
function refuse(message) { throw new TypeError(`PNG refused: ${message}`); }
function crc(bytes) {
  let c = 0xffffffff;
  for (const b of bytes) c = (c >>> 8) ^ crcTable[(c ^ b) & 255];
  return (c ^ 0xffffffff) >>> 0;
}
function dimensions(size) {
  if (!size || !Number.isInteger(size.width) || !Number.isInteger(size.height)
      || size.width < 1 || size.height < 1 || size.width > 2400 || size.height > 2400) {
    refuse('unsupported dimensions');
  }
  return size;
}
function figureText(record) {
  if (!record || !record.metadata || typeof record.metadata !== 'object' || Array.isArray(record.metadata)
      || record.metadata.mode !== 'teaching_model' || record.metadata.verification !== 'not_performed') {
    refuse('unsupported figure metadata');
  }
  let metadata;
  try { metadata = JSON.stringify(record.metadata); } catch { refuse('unsupported figure metadata'); }
  if (typeof record.citation !== 'string' || !record.citation.trim()
      || /[\u0000\ud800-\udfff]/u.test(record.citation)) refuse('unsupported citation text');
  const encoded = [encoder.encode(metadata), encoder.encode(record.citation)];
  if (encoded.some(bytes => bytes.length > MAX_TEXT_BYTES)) refuse('figure metadata or citation exceeds limit');
  return encoded;
}
function textChunk(keyword, text) {
  const key = encoder.encode(keyword);
  // iTXt: keyword NUL, no compression, method0, empty language NUL,
  // empty translated keyword NUL, followed by UTF-8 text.
  const data = new Uint8Array(key.length + 5 + text.length);
  data.set(key); data.set(text, key.length + 5);
  const result = new Uint8Array(data.length + 12), view = new DataView(result.buffer);
  view.setUint32(0, data.length);
  result.set([105, 84, 88, 116], 4); result.set(data, 8);
  view.setUint32(result.length - 4, crc(result.subarray(4, -4)));
  return result;
}

export function addFigurePNGMetadata(input, record, size) {
  dimensions(size);
  const [metadata, citation] = figureText(record);
  if (!(input instanceof Uint8Array) || input.length < 57 || input.length > MAX_PNG_BYTES
      || SIGNATURE.some((b, i) => input[i] !== b)) refuse('invalid PNG signature or size');
  const view = new DataView(input.buffer, input.byteOffset, input.byteLength);
  let p = 8, count = 0, imageData = false, endedImageData = false, palette = false, colorType;
  let end = -1, imageBytes = 0;
  while (p < input.length) {
    if (++count > 256 || input.length - p < 12) refuse('truncated or excessive PNG chunks');
    const length = view.getUint32(p);
    if (length > MAX_PNG_BYTES || length > input.length - p - 12) refuse('unbounded or truncated PNG chunk');
    const type = String.fromCharCode(...input.subarray(p + 4, p + 8));
    if (!/^[A-Za-z]{2}[A-Z][A-Za-z]$/.test(type)) refuse('invalid PNG chunk type');
    if (view.getUint32(p + 8 + length) !== crc(input.subarray(p + 4, p + 8 + length))) refuse('PNG chunk CRC mismatch');
    if (count === 1 && type !== 'IHDR' || count > 1 && type === 'IHDR') refuse('misplaced PNG header chunk');
    if (type === 'IHDR') {
      if (length !== 13 || view.getUint32(p + 8) !== size.width || view.getUint32(p + 12) !== size.height) {
        refuse('PNG dimensions do not match the prepared figure');
      }
      const depth = input[p + 16]; colorType = input[p + 17];
      const depths = {0: [1, 2, 4, 8, 16], 2: [8, 16], 3: [1, 2, 4, 8], 4: [8, 16], 6: [8, 16]};
      if (!depths[colorType]?.includes(depth) || input[p + 18] !== 0 || input[p + 19] !== 0 || input[p + 20] > 1) {
        refuse('unsupported PNG header');
      }
    } else if (type === 'PLTE') {
      if (palette || imageData || !length || length > 768 || length % 3 || [0, 4].includes(colorType)) refuse('invalid PNG palette');
      palette = true;
    } else if (type === 'IDAT') {
      if (endedImageData || colorType === 3 && !palette) refuse('misplaced PNG image-data chunk');
      imageData = true; imageBytes += length;
    } else if (type === 'IEND') {
      if (length || !imageData || !imageBytes || p + 12 !== input.length) refuse('missing image data or invalid terminal PNG chunk');
      end = p; break;
    } else {
      if (['tEXt', 'zTXt', 'iTXt'].includes(type)) refuse('preexisting PNG text metadata');
      if (type[0] === type[0].toUpperCase()) refuse('unsupported critical PNG chunk');
      if (imageData) endedImageData = true;
    }
    p += length + 12;
  }
  if (end < 0) refuse('missing terminal PNG chunk');
  const additions = [textChunk('Universal Law figure', metadata), textChunk('Description', citation),
    textChunk('Universal Law raster', encoder.encode(JSON.stringify({schema: 'universal-law/teaching-png/v1',
      width: size.width, height: size.height, scale: 2, pixels: 'browser-rendered', mathematical_verification: 'not_performed'})))];
  const output = new Uint8Array(input.length + additions.reduce((n, b) => n + b.length, 0));
  output.set(input.subarray(0, end));
  let offset = end;
  for (const bytes of additions) { output.set(bytes, offset); offset += bytes.length; }
  output.set(input.subarray(end), offset);
  return output;
}

function validateCapture(record) {
  figureText(record);
  if (typeof record.svg !== 'string' || record.svg.length > 1024 * 1024
      || record.width !== 600 || ![535, 582, 478, 496].includes(record.height)) refuse('unsupported captioned SVG capture');
  return JSON.stringify([record.svg, record.metadata, record.citation, record.width, record.height]);
}

export function rasterizeTeachingPNG(record, {signal, document = globalThis.document, timeout = 15000} = {}) {
  validateCapture(record);
  if (!document || typeof document.createElement !== 'function' || typeof Blob !== 'function') {
    return Promise.reject(new Error('PNG preparation is unavailable in this browser.'));
  }
  if (!Number.isFinite(timeout) || timeout < 1 || timeout > 15000) throw new RangeError('PNG preparation limit must be 1–15000 milliseconds');
  return new Promise((resolve, reject) => {
    let canvas, image, timer, finished = false;
    const abort = () => finish(new Error('PNG preparation was canceled.'));
    function finish(error, value) {
      if (finished) return;
      finished = true; clearTimeout(timer); signal?.removeEventListener('abort', abort);
      if (image) { image.onload = null; image.onerror = null; image.removeAttribute('src'); }
      if (canvas) { canvas.width = 0; canvas.height = 0; }
      if (error) reject(error); else resolve(value);
    }
    if (signal?.aborted) { finish(new Error('PNG preparation was canceled.')); return; }
    signal?.addEventListener('abort', abort, {once: true});
    timer = setTimeout(() => finish(new Error('PNG preparation timed out. Please try again.')), timeout);
    try {
      canvas = document.createElement('canvas');
      const width = record.width * 2, height = record.height * 2;
      canvas.width = width; canvas.height = height;
      const context = canvas.getContext?.('2d');
      if (!context || typeof canvas.toBlob !== 'function') throw new Error('PNG canvas encoding is unavailable in this browser.');
      image = document.createElement('img');
      image.onerror = () => finish(new Error('The captioned SVG image could not be decoded.'));
      image.onload = () => {
        if (finished) return;
        try {
          if (image.naturalWidth !== record.width || image.naturalHeight !== record.height) throw new Error('Decoded SVG dimensions do not match.');
          context.drawImage(image, 0, 0, width, height);
          canvas.toBlob(async blob => {
            if (finished) return;
            try {
              if (!blob || blob.type !== 'image/png' || blob.size < 1 || blob.size > MAX_PNG_BYTES) throw new Error('PNG image encoding returned no supported image.');
              const bytes = new Uint8Array(await blob.arrayBuffer());
              if (finished) return;
              finish(null, new Blob([addFigurePNGMetadata(bytes, record, {width, height})], {type: 'image/png'}));
            } catch (error) { finish(error); }
          }, 'image/png');
        } catch (error) { finish(error); }
      };
      // The page CSP permits data images, not blob images. The supplied SVG has
      // already passed the existing feature's resource/geometry validator.
      image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(record.svg)}`;
    } catch (error) { finish(error); }
  });
}

export function downloadFigurePNG(blob, filename, document = globalThis.document, urls = globalThis.URL) {
  let url, link;
  try {
    url = urls.createObjectURL(blob);
    link = document.createElement('a'); link.href = url; link.download = filename;
    document.body.append(link); link.click();
  } finally {
    link?.remove();
    if (url) setTimeout(() => urls.revokeObjectURL(url), 1000);
  }
}

export function createFigurePNGController({capture, render = rasterizeTeachingPNG, download,
  now = () => new Date().toISOString(), onChange = () => {}}) {
  let ready = false, pending = false, revision = 0, active = null;
  let status = 'PNG export becomes available after the diagram is drawn.';
  const state = () => ({ready, pending, status});
  const notify = () => onChange(state());
  return {
    state,
    refresh(available) {
      const canceled = pending;
      revision++; active?.abort(); active = null; pending = false; ready = !!available;
      status = !ready ? 'PNG export is unavailable. The SVG and readable figure citation remain below.'
        : canceled ? 'Figure changed; the previous PNG was canceled. Download again for the current figure.'
        : 'PNG includes the caption, source metadata and citation. It is a teaching figure.';
      notify();
    },
    async export() {
      if (!ready || pending) return false;
      const started = ++revision, abort = new AbortController(); active = abort;
      pending = true; status = 'Preparing the captioned PNG…'; notify();
      try {
        const timestamp = now(), original = capture(timestamp), fingerprint = validateCapture(original);
        const blob = await render(original, {signal: abort.signal});
        if (started !== revision || abort.signal.aborted) return false;
        if (!(blob instanceof Blob) || blob.type !== 'image/png' || !blob.size || blob.size > MAX_PNG_BYTES + 3 * MAX_TEXT_BYTES) {
          throw new Error('PNG encoding returned no supported image.');
        }
        if (validateCapture(capture(timestamp)) !== fingerprint) throw new Error('The displayed figure changed during PNG preparation; download refused.');
        download(blob);
        status = 'PNG download started. Caption, source metadata and citation are included; no mathematical verification was performed.';
        return true;
      } catch (error) {
        if (started === revision) status = `PNG could not complete: ${error instanceof Error ? error.message : 'image preparation failed'}. Try again, or use SVG and the readable citation.`;
        return false;
      } finally {
        if (started === revision) { active = null; pending = false; notify(); }
      }
    }
  };
}
