import {wireReferenceForm} from './source-reference.mjs?site-release=22f1fb8018b60697fa0bbad3681c49a2c849baebc843c15c9ea45aafef80ddd0';
const byId = id => document.getElementById(id);
wireReferenceForm({form:byId('reference-form'), repository:byId('reference-repository'), commit:byId('reference-commit'), path:byId('reference-path'), sha256:byId('reference-sha256'), output:byId('reference-output'), json:byId('reference-json'), link:byId('reference-link'), status:byId('reference-status'), actions:byId('reference-actions'), copyText:byId('reference-copy'), copyJSON:byId('reference-copy-json'), share:byId('reference-share'), clipboard:globalThis.navigator?.clipboard, location:globalThis.location, history:globalThis.history, events:globalThis.window});
byId('reference-builder').hidden = false;
