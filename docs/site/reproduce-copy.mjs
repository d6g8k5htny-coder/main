// Copies visible source-pinned commands only. It never executes a calculation.
export function wireCommandCopy({button, block, status, clipboard}) {
  const fallback = 'Select the command block above and copy it manually. No commands have been run.';
  if (typeof clipboard?.writeText !== 'function') {
    status.textContent = fallback;
    return;
  }
  button.addEventListener('click', async () => {
    button.disabled = true;
    status.textContent = 'Copying commands…';
    try {
      await clipboard.writeText(block.textContent);
      status.textContent = 'Copied all commands. They have not run; paste them into your own terminal when ready.';
    } catch {
      status.textContent = fallback;
    } finally {
      button.disabled = false;
    }
  });
  button.disabled = false;
}

if (typeof document !== 'undefined') {
  wireCommandCopy({button:document.getElementById('copy-commands'), block:document.getElementById('reproduction-commands'), status:document.getElementById('copy-commands-status'), clipboard:globalThis.navigator?.clipboard});
}
