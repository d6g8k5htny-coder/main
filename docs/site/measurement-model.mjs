// Synthetic educational arithmetic for EXPERIMENT.md §1, SIDE24 in d = 2.
// No field is generated and no empirical or continuum claim is tested here.
const AREA = 576;
const labels = {
  count: 'Aggregate count N', realizations: 'Realizations m',
  lower: 'Lower edge a', upper: 'Upper edge b',
};

function invalid(field, message) {
  throw Object.assign(new RangeError(message), { field });
}

export function parseMeasurementInputs(inputs) {
  const result = {};
  for (const field of Object.keys(labels)) {
    const text = inputs[field];
    if (typeof text !== 'string' || !text.trim())
      invalid(field, `Enter ${labels[field].toLowerCase()}.`);
    if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(text.trim()))
      invalid(field, `${labels[field]} must be a decimal number.`);
    result[field] = Number(text);
    if (!Number.isFinite(result[field]))
      invalid(field, `${labels[field]} must be finite.`);
    const [mantissa, exponent = '0'] = text.trim().toLowerCase().split('e');
    const digits = mantissa.replace(/[+.-]/g, '');
    const nonzero = /[1-9]/.test(digits);
    if (nonzero && result[field] === 0)
      invalid(field, `${labels[field]} is too small to represent.`);
    if (nonzero && (field === 'count' || field === 'realizations')) {
      // Check integrality in the entered decimal before binary rounding can
      // turn a fractional count such as 576.00000000000001 into 576.
      const decimalPlaces = (mantissa.split('.')[1] || '').length;
      const trailingZeros = digits.length - digits.replace(/0+$/, '').length;
      if (decimalPlaces - Number(exponent) > trailingZeros)
        invalid(field, `${labels[field]} must be a whole number.`);
    }
  }
  return result;
}

export function measurementModel({ count, realizations, lower, upper }) {
  if (!Number.isSafeInteger(count) || count < 0)
    invalid('count', 'Aggregate count N must be a nonnegative safe integer.');
  if (!Number.isSafeInteger(realizations) || realizations <= 0)
    invalid('realizations', 'Realizations m must be a positive safe integer, including zero-count realizations.');
  if (!Number.isFinite(lower) || lower <= 0)
    invalid('lower', 'Lower edge a must be finite and greater than zero.');
  if (!Number.isFinite(upper) || upper <= lower)
    invalid('upper', 'Upper edge b must be finite and greater than a.');
  const exposure = AREA * realizations;
  if (!Number.isSafeInteger(exposure))
    invalid('realizations', 'Use fewer realizations: 576 × m must remain a safe integer.');
  const width = upper - lower;
  const mass = count === 0 ? 0 : count / exposure;
  const density = mass / width;
  if (!Number.isFinite(width) || width <= 0 || !Number.isFinite(density) ||
      (count > 0 && (mass <= 0 || density <= 0)))
    invalid('upper', 'These values exceed the numeric range of this example. Use less extreme bin edges.');
  return { count: count || 0, realizations, lower, upper, area: AREA, exposure, width, mass, density };
}
