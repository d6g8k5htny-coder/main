import {wireReferenceForm} from './source-reference.mjs?site-release=fd455ab79541be5b46f2a045e4120ae0fa3c429b3b6df1dbf73cdaff65924e58';
const byId = id => document.getElementById(id);
wireReferenceForm({form:byId('reference-form'), repository:byId('reference-repository'), commit:byId('reference-commit'), path:byId('reference-path'), sha256:byId('reference-sha256'), output:byId('reference-output'), json:byId('reference-json'), link:byId('reference-link'), status:byId('reference-status'), actions:byId('reference-actions'), copyText:byId('reference-copy'), copyJSON:byId('reference-copy-json'), share:byId('reference-share'), clipboard:globalThis.navigator?.clipboard, location:globalThis.location, history:globalThis.history, events:globalThis.window});
byId('reference-builder').hidden = false;
