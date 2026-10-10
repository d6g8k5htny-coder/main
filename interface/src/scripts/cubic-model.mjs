// NON-CERTIFYING cubic arithmetic with explicit affine teaching coordinates.
// Source: d6g8k5htny-coder/main@032267c259162728229e3ddd14d08fe1c3f9a392,
// docs/site/explore-models.mjs, pinModel; source SHA-256:
// fdbf01dd7235f0b35e1f7dad30fbd70f07d30a690b83ff9a87b3506390a7159b.
// Unrelated annulus/remote geometry is excluded from this pure cubic module.
// Strict admission and nominal normalization are new teaching definitions from
// docs/plans/2026-10-09-canvas-model-contract.md, not a continuum sampling law.

export function cubicModel(input) {
  if (input === null || typeof input !== 'object' || Array.isArray(input)) {
    throw new TypeError('A cubic-parameter record is required');
  }
  for (const name of ['r', 'k', 'b']) {
    if (!Object.hasOwn(input, name)) throw new TypeError(`Own ${name} is required`);
  }
  const {r, k, b} = input;
  for (const [name, value] of Object.entries({r, k, b})) {
    if (typeof value !== 'number') throw new TypeError(`${name} must be a primitive Number`);
    if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  }
  if (r <= 0 || k <= 0) throw new RangeError('r and k must be positive');
  const rCubed = r ** 3;
  const gap = k * rCubed;
  const pins = [-r / 2, r / 2];
  const lower = b - gap;
  if (!Number.isFinite(rCubed) || rCubed <= 0 || !Number.isFinite(gap) || gap <= 0
      || !pins.every(Number.isFinite) || pins[0] === pins[1]
      || !Number.isFinite(lower) || lower >= b) {
    throw new RangeError('The derived scales must remain representable');
  }
  return {
    r, k, b, rCubed, pins, gap, heightWindow: [lower, b],
    // These nominal coordinates express X=x/r and Y=(y-b)/gap. They do not
    // assert exact recovery by dividing already-rounded Number endpoints.
    normalized: {
      xDivisor: r, yOrigin: b, yDivisor: gap,
      pins: [-0.5, 0.5], heightWindow: [-1, 0],
    },
  };
}
