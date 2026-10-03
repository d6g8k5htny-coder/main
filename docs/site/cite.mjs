import {wireReferenceForm} from './source-reference.mjs?site-release=8b15350e744dac57c6fab662a1ce7e5bb6ea730f009649479e49a4786b240b10';
const byId = id => document.getElementById(id);
wireReferenceForm({form:byId('reference-form'), repository:byId('reference-repository'), commit:byId('reference-commit'), path:byId('reference-path'), output:byId('reference-output'), link:byId('reference-link'), status:byId('reference-status')});
byId('reference-builder').hidden = false;
