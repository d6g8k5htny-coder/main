import {wireReferenceForm} from './source-reference.mjs';
const byId = id => document.getElementById(id);
wireReferenceForm({form:byId('reference-form'), repository:byId('reference-repository'), commit:byId('reference-commit'), path:byId('reference-path'), output:byId('reference-output'), link:byId('reference-link'), status:byId('reference-status')});
byId('reference-builder').hidden = false;
