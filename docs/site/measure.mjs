import { measurementModel, parseMeasurementInputs } from './measurement-model.mjs?site-release=5b0d24cadc9e63ca73d710a1fe5ee9eef854d164e438a52e20b15d8d070c9e15';

const byId = id => document.getElementById(id);
const fields = Object.fromEntries(['count', 'realizations', 'lower', 'upper'].map(name => [name, byId(`measure-${name}`)]));
const form = byId('measure-form');
const error = byId('measure-error');
const status = byId('measure-status');
const format = value => value === 0 ? '0' : String(Number(value.toPrecision(6)));

function read() {
  return parseMeasurementInputs(Object.fromEntries(Object.entries(fields).map(([name, input]) => [name, input.value])));
}

function showError(problem, focus) {
  for (const input of Object.values(fields)) input.removeAttribute('aria-invalid');
  const input = fields[problem.field];
  if (input) input.setAttribute('aria-invalid', 'true');
  error.textContent = problem.message;
  error.hidden = false;
  for (const name of ['mass', 'density', 'width', 'exposure']) byId(`measure-${name}`).textContent = '—';
  byId('measure-formula').textContent = 'Enter valid inputs to calculate.';
  status.textContent = 'Results are unavailable until the inputs are valid.';
  if (focus && input) input.focus();
}

function render(result) {
  for (const input of Object.values(fields)) input.removeAttribute('aria-invalid');
  error.hidden = true;
  error.textContent = '';
  for (const name of ['mass', 'density', 'width', 'exposure']) byId(`measure-${name}`).textContent = format(result[name]);
  byId('measure-formula').textContent = `${format(result.count)} ÷ ${format(result.exposure)} ≈ ${format(result.mass)}; ${format(result.mass)} ÷ ${format(result.width)} ≈ ${format(result.density)}`;
  status.textContent = `${format(result.count)} synthetic counts across ${format(result.realizations)} realizations give a bin mass of approximately ${format(result.mass)} per unit area. Dividing by bin width ${format(result.width)} gives a density of approximately ${format(result.density)}.`;
}

function calculate(focusError = false) {
  try { render(measurementModel(read())); }
  catch (problem) { showError(problem, focusError); }
}

form.addEventListener('submit', event => { event.preventDefault(); calculate(true); });
for (const input of Object.values(fields)) input.addEventListener('input', () => calculate());
byId('measure-reset').addEventListener('click', () => { form.reset(); calculate(); });
byId('measure-wider').addEventListener('click', () => {
  try {
    const current = measurementModel(read());
    const wider = measurementModel({ ...current, upper: current.lower + 2 * current.width });
    fields.upper.value = String(wider.upper);
    render(wider);
  } catch (problem) { showError(problem, true); }
});

calculate();
byId('measure-controls').disabled = false;
byId('measure-loading').hidden = true;
