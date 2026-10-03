import {wireReferenceForm} from './source-reference.mjs?site-release=b308d748b0db6346d512b5a5eaeadbc70a2436121186c76fd10f8daa6405a91e';
const byId = id => document.getElementById(id);
wireReferenceForm({form:byId('reference-form'), repository:byId('reference-repository'), commit:byId('reference-commit'), path:byId('reference-path'), sha256:byId('reference-sha256'), output:byId('reference-output'), json:byId('reference-json'), link:byId('reference-link'), status:byId('reference-status'), actions:byId('reference-actions'), copyText:byId('reference-copy'), copyJSON:byId('reference-copy-json'), share:byId('reference-share'), clipboard:globalThis.navigator?.clipboard, location:globalThis.location, history:globalThis.history, events:globalThis.window});
byId('reference-builder').hidden = false;
