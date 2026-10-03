// NON-CERTIFYING educational arithmetic. These functions illustrate source
// definitions; they do not simulate a field or estimate a theorem's constants.
import { annulusModel, p15Model, remoteModel } from './geometry.mjs?site-release=59ae2f18050fb2edc3d40738450b9137e3570730af2ff09026589b086d78c404';

function finite(value, name) {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return value;
}

export function coneModel(s, radius) {
  finite(s, 's'); finite(radius, 'radius');
  if (radius < 0) throw new RangeError('radius must be nonnegative');
  const eigenvalues = [s - radius, s + radius].map(value => value === 0 ? 0 : value);
  const determinant = eigenvalues[0] * eigenvalues[1];
  const square = determinant ** 2;
  [...eigenvalues, square].forEach(value => finite(value, 'derived value'));
  const kind = eigenvalues.includes(0) ? 'flat'
    : eigenvalues[1] < 0 ? 'peak' : eigenvalues[0] > 0 ? 'bowl' : 'saddle';
  return { s, radius, eigenvalues, determinant, kind, coneWeight: kind === 'peak' ? square : 0 };
}

export function pinModel({ r = 0.5, k = 1, b = 1, A = 2, B = 4, rho = 3, L = 24 } = {}) {
  for (const [name, value] of Object.entries({ r, k, b })) finite(value, name);
  if (r <= 0 || k <= 0) throw new RangeError('r and k must be positive');
  const remote = remoteModel({ L, rho });
  const annular = annulusModel({ A, B, dimension: 2 });
  const gap = k * r ** 3;
  const lower = b - gap;
  if (!Number.isFinite(gap) || gap <= 0 || !Number.isFinite(lower) || lower >= b || !Number.isFinite(B * r))
    throw new RangeError('The derived scales must remain representable');
  return {
    r, k, b, pins: [-r / 2, r / 2], gap, heightWindow: [lower, b],
    annulus: [A * r, B * r], remoteRadius: rho,
    containsRemote(distance, height) { return remote.contains({ distance, height, b, k, r }); },
    containsAnnulus(x, y) { return annular.containsPhysical({ x, y, r }); }
  };
}

export function paletteModel(selection) { return p15Model(selection); }
