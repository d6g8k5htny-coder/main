// URL state for the existing teaching controls, not a scientific record.
const numeric = { s: [-3, 3, 10], R: [0, 2, 10], r: [0.05, 0.6, 100] };

export function readExploreState(search) {
  const params = new URLSearchParams(search);
  const state = { s: -2, R: 1, r: 0.5, region: 'annulus', objects: [1, 2, 3] };
  const invalid = [];
  for (const [key, fallback] of Object.entries(state)) {
    if (!params.has(key)) continue;
    const values = params.getAll(key);
    const raw = values[0];
    let value = fallback;
    let valid = values.length === 1;
    if (key in numeric) {
      const [min, max, scale] = numeric[key];
      const number = Number(raw);
      valid &&= raw.length <= 32 && /^-?(?:\d+(?:\.\d*)?|\.\d+)$/.test(raw)
        && Number.isFinite(number) && number >= min && number <= max
        && Math.abs(number * scale - Math.round(number * scale)) < 1e-8;
      value = Math.round(number * scale) / scale;
    } else if (key === 'region') {
      valid &&= raw === 'annulus' || raw === 'remote';
      value = raw;
    } else {
      const parts = raw === '' ? [] : raw.split(',');
      valid &&= parts.length <= 3 && parts.every(x => /^[123]$/.test(x))
        && new Set(parts).size === parts.length;
      value = parts.map(Number).sort();
    }
    if (valid) state[key] = value;
    else invalid.push(key);
  }
  return { state, invalid };
}

export function exploreStateURL(href, state) {
  const url = new URL(href);
  for (const key of ['s', 'R', 'r', 'region', 'objects']) {
    url.searchParams.set(key, key === 'objects' ? state.objects.join(',') : String(state[key]));
  }
  return url.href;
}
