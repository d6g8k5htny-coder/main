// NON-CERTIFYING educational arithmetic: faithful first predecessor port.
// Source: d6g8k5htny-coder/main@032267c259162728229e3ddd14d08fe1c3f9a392,
// docs/site/explore-models.mjs, pinModel; source SHA-256:
// fdbf01dd7235f0b35e1f7dad30fbd70f07d30a690b83ff9a87b3506390a7159b.
// The source defaults and finite-value error classification are retained.
// Unrelated annulus/remote geometry is excluded from this pure cubic module.
// Exposed rCubed and affine normalization are absent from this predecessor.

function finite(value, name) {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return value;
}

export function cubicModel({r = 0.5, k = 1, b = 1} = {}) {
  for (const [name, value] of Object.entries({r, k, b})) finite(value, name);
  if (r <= 0 || k <= 0) throw new RangeError('r and k must be positive');
  const gap = k * r ** 3;
  const lower = b - gap;
  if (!Number.isFinite(gap) || gap <= 0 || !Number.isFinite(lower) || lower >= b) {
    throw new RangeError('The derived scales must remain representable');
  }
  return {r, k, b, pins: [-r / 2, r / 2], gap, heightWindow: [lower, b]};
}
