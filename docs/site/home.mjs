// Preserve bookmarked technical sections after introducing the public home.
export function legacyDestination(hash) {
  return new Set(['#board', '#coefficient', '#inventory', '#contribute']).has(hash)
    ? `workspace.html${hash}` : null;
}
if (typeof window !== 'undefined') {
  const followLegacy = () => {
    const destination = legacyDestination(window.location.hash);
    if (destination) window.location.replace(destination);
  };
  followLegacy();
  window.addEventListener('hashchange', followLegacy);
}
